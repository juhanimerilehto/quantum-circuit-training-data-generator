#!/usr/bin/env python3
"""
Quantum Circuit Generator
Reads categories from CSV, generates unique OpenQASM 2.0 circuits via the xAI Grok API.

Key features:
- Hardened system prompt with explicit qelib1.inc gate allowlist (no QASM 3.0 leakage)
- Inline QASM validation via qiskit -- invalid circuits rejected before storage
- SHA-256 hash-based deduplication -- no duplicate circuits
- Chunked batch generation: large variant counts split into <=15 per API call
- Auto-fix pass for common Grok QASM 3.0 mistakes before validation
"""
import json
import csv
import hashlib
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

# ---------------------------------------------------------------------------
# Hardened system prompt with explicit qelib1.inc gate allowlist
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a quantum circuit expert. Generate diverse, unique, valid OpenQASM 2.0 circuits. "
    "Use ONLY gates defined in qelib1.inc: "
    "h, x, y, z, s, t, sdg, tdg, id, "
    "rx, ry, rz, u1, u2, u3, "
    "cx, cz, cy, ch, crz, cu1, cu3, ccx, cswap, swap, "
    "measure, reset, barrier. "
    "NEVER use p(), cp(), crx(), cry(), sx(), ecr(), rzz(), rxx() or ANY QASM 3.0 gate. "
    "Classical conditionals must use: if(creg==integer) gate qubit; "
    "Gate calling convention is: gate(param) qubit; NOT gate(param, qubit); "
    "Return ONLY valid JSON with plain ASCII text."
)

# ---------------------------------------------------------------------------
# Gate definitions to inject when Grok still slips in non-qelib1 gates
# ---------------------------------------------------------------------------
GATE_DEFS = {
    'p':     'gate p(lambda) q { U(0,0,lambda) q; }',
    'cp':    'gate cp(lambda) c,t { p(lambda/2) c; cx c,t; p(-lambda/2) t; cx c,t; p(lambda/2) t; }',
    'crx':   'gate crx(theta) c,t { ry(pi/2) t; cx c,t; ry(-pi/2) t; rz(-theta/2) t; cx c,t; rz(theta/2) t; }',
    'cry':   'gate cry(theta) c,t { ry(theta/2) t; cx c,t; ry(-theta/2) t; cx c,t; }',
    'swap':  'gate swap a,b { cx a,b; cx b,a; cx a,b; }',
    'cswap': 'gate cswap c,a,b { cx b,a; ccx c,a,b; cx b,a; }',
}
GATE_DEF_ORDER = ['p', 'cp', 'crx', 'cry', 'swap', 'cswap']


def _try_load(qasm: str) -> bool:
    """Return True if QASM parses cleanly via qiskit."""
    try:
        from qiskit.qasm2 import loads
        loads(qasm)
        return True
    except Exception:
        return False


def _fix_qasm(qasm: str) -> str:
    """Best-effort auto-fix for common Grok QASM 3.0 mistakes."""

    def fix_call(m):
        gate, inner = m.group(1), m.group(2)
        parts = [p.strip() for p in inner.split(',')]
        params, qubits = [], []
        for p in parts:
            if re.search(r'q\[|^q$', p):
                qubits.append(p)
            else:
                params.append(p)
        if params and qubits:
            return f"{gate}({','.join(params)}) {','.join(qubits)}"
        return m.group(0)
    qasm = re.sub(r'\b(\w+)\(([^)]+)\)', fix_call, qasm)

    def fix_bare(m):
        gate, param, q0, q1 = m.group(1), m.group(2), m.group(3), m.group(4)
        return f'{gate}({param}) q[{q0}],q[{q1}]'
    qasm = re.sub(r'\b(crx|cry|crz)\s+([\w/*+\-.pi]+)\s+(\d+)\s+(\d+)', fix_bare, qasm)

    def fix_3arg(m):
        gate, args_str = m.group(1), m.group(2)
        args = [a.strip() for a in args_str.split(',')]
        if len(args) == 3:
            try:
                int(args[1]); int(args[2])
                return f'{gate}({args[0]}) q[{args[1]}],q[{args[2]}]'
            except ValueError:
                pass
        return f'{gate}({args[0]})'
    qasm = re.sub(r'\b(crz|crx|cry)\(([^)]+)\)', fix_3arg, qasm)

    def fix_cu1(m):
        args = [a.strip() for a in m.group(1).split(',')]
        if len(args) == 3:
            try:
                int(args[1]); int(args[2])
                return f'cu1({args[0]}) q[{args[1]}],q[{args[2]}]'
            except ValueError:
                pass
        return f'cu1({args[0]})'
    qasm = re.sub(r'\bcu1\(([^)]+)\)', fix_cu1, qasm)

    def fix_rot2(m):
        gate, args_str = m.group(1), m.group(2)
        return f'{gate}({args_str.split(",")[0].strip()})'
    qasm = re.sub(r'\b(ry|rz)\(([^)]+,[^)]+)\)', fix_rot2, qasm)

    qasm = re.sub(r'if\s*\(\s*c\[(\d+)\]\s*==\s*(\d+)\s*\)', r'if(c==\2)', qasm)
    qasm = re.sub(r'(\w+)\s+(q\[\d+\])\s+\[c\[\d+\]\]\s*;', r'if(c==1) \1 \2;', qasm)
    qasm = re.sub(r'\bsgd\b', 'sdg', qasm)

    needed = []
    for gate in GATE_DEF_ORDER:
        if re.search(rf'\b{re.escape(gate)}\b', qasm) and f'gate {gate}' not in qasm:
            needed.append(GATE_DEFS[gate])
    if needed:
        qasm = qasm.replace(
            'include "qelib1.inc";',
            'include "qelib1.inc";\n' + '\n'.join(needed),
            1
        )

    return qasm


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class CircuitGenerator:
    CHUNK_SIZE = 15  # max variants per single API call

    def __init__(self, categories_file="data/categories.csv",
                 output_file="output/master_circuits.json",
                 hash_db="output/circuit_hashes.json"):
        self.categories_file = categories_file
        self.output_file = output_file
        self.hash_db = hash_db

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        self.hashes = self._load_hashes()
        self.circuits = self._load_circuits()

        try:
            from qiskit.qasm2 import loads  # noqa
            self._validate = True
            print("Inline QASM validation: enabled (qiskit)")
        except ImportError:
            self._validate = False
            print("Inline QASM validation: disabled (pip install qiskit)")

        print(f"Loaded {len(self.hashes)} existing circuit hashes")
        print(f"Loaded {len(self.circuits)} existing circuits")

    def _load_hashes(self):
        if Path(self.hash_db).exists():
            with open(self.hash_db, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_hashes(self):
        with open(self.hash_db, 'w') as f:
            json.dump(list(self.hashes), f, indent=2)

    def _load_circuits(self):
        if Path(self.output_file).exists():
            with open(self.output_file, 'r') as f:
                return json.load(f)
        return []

    def _save_circuits(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.circuits, f, indent=2)
        self._save_hashes()

    def _hash_circuit(self, qasm):
        normalized = ''.join(qasm.split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _add_circuit(self, description, qasm, category, metadata=None):
        """Validate (+ auto-fix), deduplicate, and store. Returns True if added."""
        if self._validate:
            if not _try_load(qasm):
                fixed = _fix_qasm(qasm)
                if not _try_load(fixed):
                    return False
                qasm = fixed

        circuit_hash = self._hash_circuit(qasm)
        if circuit_hash in self.hashes:
            return False

        self.hashes.add(circuit_hash)
        circuit = {"description": description, "qasm": qasm,
                   "category": category, "hash": circuit_hash}
        if metadata:
            circuit.update(metadata)
        self.circuits.append(circuit)
        return True

    def _clean_json(self, text):
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    text = part
                    break
        text = text.replace("json", "", 1).strip()
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace('\ufeff', '')
        return text.strip()

    def _max_tokens(self, qubits: int, chunk: int) -> int:
        qubits = int(qubits)
        if qubits <= 2:
            base = 3000
        elif qubits <= 4:
            base = 4000
        elif qubits <= 6:
            base = 6000
        else:
            base = 8000
        extra = max(0, chunk - 5) * 600
        return min(base + extra, 16000)

    def _generate_chunk(self, category_data, chunk_size, max_retries=3):
        category    = category_data['category']
        subcategory = category_data['subcategory']
        qubits      = category_data['qubits']
        desc        = category_data['description_template']
        special     = category_data['special_params']
        max_tokens  = self._max_tokens(qubits, chunk_size)

        prompt = f"""Generate {chunk_size} unique quantum circuits for this category:

Category: {category}
Subcategory: {subcategory}
Qubits: {qubits}
Description template: {desc}
Special parameters: {special}

Requirements:
- Generate {chunk_size} DIFFERENT circuits (vary gates, sequences, parameters)
- All circuits must be {qubits} qubits
- OpenQASM 2.0 only -- use ONLY qelib1.inc gates
- Use plain ASCII only in descriptions
- Each circuit must be UNIQUE and VALID

Return ONLY this JSON format:
{{
  "circuits": [
    {{
      "description": "specific description of this variant",
      "qasm": "OPENQASM 2.0;\\ninclude \\"qelib1.inc\\";\\nqreg q[{qubits}];\\ncreg c[{qubits}];\\n...\\nmeasure q -> c;"
    }}
  ]
}}"""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="grok-4.20-0309-reasoning",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=max_tokens,
                    timeout=120
                )

                content = self._clean_json(response.choices[0].message.content)
                data = json.loads(content)

                if "circuits" not in data:
                    raise ValueError("No 'circuits' key in response")

                unique_count = 0
                for circuit in data["circuits"]:
                    if self._add_circuit(
                        circuit["description"],
                        circuit["qasm"],
                        category,
                        {
                            "subcategory": subcategory,
                            "qubits": int(qubits),
                            "source": "grok_generated"
                        }
                    ):
                        unique_count += 1

                return unique_count, len(data["circuits"])

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      Retry {attempt+2}/{max_retries}: {e}")
                    time.sleep(2)
                else:
                    print(f"      Failed: {e}")
                    return 0, 0

        return 0, 0

    def generate_batch(self, category_data):
        variants = int(category_data['variants'])
        total_unique = 0
        total_generated = 0
        remaining = variants

        while remaining > 0:
            chunk = min(remaining, self.CHUNK_SIZE)
            u, g = self._generate_chunk(category_data, chunk)
            total_unique += u
            total_generated += g
            remaining -= chunk
            if remaining > 0:
                time.sleep(0.3)

        return total_unique, total_generated

    def _load_categories(self):
        categories = []
        with open(self.categories_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                categories.append(row)
        return categories

    def generate_all(self, start_from=0, save_every=10):
        categories = self._load_categories()

        print(f"\n{'='*70}")
        print(f"QUANTUM CIRCUIT GENERATOR")
        print(f"{'='*70}")
        print(f"Categories file:   {self.categories_file}")
        print(f"Output file:       {self.output_file}")
        print(f"Total categories:  {len(categories)}")
        print(f"Starting from:     {start_from}")
        print(f"Existing circuits: {len(self.circuits)}")
        print(f"Chunk size:        {self.CHUNK_SIZE}")
        print(f"{'='*70}\n")

        total_generated = 0
        total_duplicates = 0

        for i, category_data in enumerate(categories[start_from:], start=start_from):
            category = category_data['category']
            variants = int(category_data['variants'])
            qubits   = category_data['qubits']
            chunks   = (variants + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

            print(f"[{i+1}/{len(categories)}] {category} "
                  f"({variants} variants, {qubits}q, {chunks} chunk(s))")

            unique, total = self.generate_batch(category_data)
            duplicates = total - unique

            total_generated += unique
            total_duplicates += duplicates

            print(f"  unique/valid: {unique}/{total} (dup+invalid: {duplicates})")
            print(f"  total circuits: {len(self.circuits)}")

            if (i + 1) % save_every == 0:
                self._save_circuits()
                print(f"  saved ({len(self.circuits)} circuits)")

            time.sleep(0.3)

        self._save_circuits()

        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total unique+valid: {len(self.circuits)}")
        print(f"Generated this run: {total_generated}")
        print(f"Dup/invalid:        {total_duplicates}")
        print(f"Saved to:           {self.output_file}")
        print(f"Hash database:      {self.hash_db}")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Quantum Circuit Generator")
    parser.add_argument("--categories", default="data/categories.csv",
                        help="Path to categories CSV (default: data/categories.csv)")
    parser.add_argument("--output", default="output/master_circuits.json",
                        help="Path to output JSON (default: output/master_circuits.json)")
    parser.add_argument("--hashes", default="output/circuit_hashes.json",
                        help="Path to hash DB (default: output/circuit_hashes.json)")
    parser.add_argument("--start", type=int, default=0,
                        help="Category index to start from (default: 0)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save progress every N categories (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=15,
                        help="Max variants per API call (default: 15)")
    args = parser.parse_args()

    generator = CircuitGenerator(
        categories_file=args.categories,
        output_file=args.output,
        hash_db=args.hashes
    )
    generator.CHUNK_SIZE = args.chunk_size
    generator.generate_all(start_from=args.start, save_every=args.save_every)


if __name__ == "__main__":
    main()
