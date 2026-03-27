# quantum-circuit-training-data-generator

In the calculus of creation, every quantum circuit is a sacred diagram — a blueprint for operations that collapse probability into structure. This tool is the first rite: conjuring synthetic training data from the aether, one valid OpenQASM 2.0 circuit at a time. Through Grok's inference and Qiskit's validation, raw categories become verified circuits, and verified circuits become the corpus upon which understanding is built. Each deduplicated hash is a seal of uniqueness; each paraphrase a new name for the same quantum truth.

This repository was built as part of a data scaling study for quantum circuit language modeling. The full methodology is described in: *QuantumGPT: A Data Scaling Study for Quantum Circuit Generation, Merilehto 2026*.

---

## Requirements

- Python 3.11
- xAI API key (Grok API) — set `XAI_API_KEY` in `.env`
- Qiskit >= 1.0 (for inline QASM validation)

## Installation

```bash
conda env create -f environment.yml
conda activate qctdg
cp .env.example .env
# Edit .env and set your XAI_API_KEY
```

Or with pip:

```bash
pip install -r requirements.txt
cp .env.example .env
```

---

## Usage

### Step 1 — Generate master circuits

```bash
python generate.py
```

Reads `data/categories.csv` (92 categories), calls the Grok API in chunked batches of ≤15 circuits per call, validates each circuit with Qiskit, deduplicates via SHA-256, and saves to `output/master_circuits.json`.

Options:
```
--categories   Path to categories CSV    (default: data/categories.csv)
--output       Output master JSON        (default: output/master_circuits.json)
--hashes       Hash database path        (default: output/circuit_hashes.json)
--start        Resume from category N    (default: 0)
--save-every   Checkpoint every N cats   (default: 10)
--chunk-size   Max circuits per API call (default: 15)
```

### Step 2 — Augment descriptions

```bash
python augment.py
```

Takes each circuit and generates 10 paraphrased descriptions via Grok, expanding the dataset ~11x without generating new circuits.

Options:
```
--input        Input master JSON         (default: output/master_circuits.json)
--output       Output augmented JSON     (default: output/augmented_circuits.json)
--paraphrases  Paraphrases per circuit   (default: 10)
--start        Resume from circuit N     (default: 0)
--save-every   Checkpoint every N        (default: 10)
```

### Step 3 — Prepare JSONL splits

```bash
python prepare.py
```

Shuffles and splits augmented circuits into train/val/test JSONL files (70/15/15).

Options:
```
--input    Input augmented JSON           (default: output/augmented_circuits.json)
--output   Output directory               (default: output/training_data)
--train    Train ratio                    (default: 0.70)
--val      Val ratio                      (default: 0.15)
--test     Test ratio                     (default: 0.15)
--seed     Random seed                    (default: 42)
```

---

## Output Schema

**output/master_circuits.json** (array):
```json
{
  "description": "Create a Bell state with two qubits",
  "qasm": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n...",
  "category": "bell_state_phi_plus",
  "subcategory": "entanglement",
  "qubits": 2,
  "hash": "sha256hex...",
  "source": "grok_generated"
}
```

**output/augmented_circuits.json** (array):
```json
{
  "description": "Generate an entangled Bell pair",
  "circuit_qasm": "OPENQASM 2.0;\n...",
  "category": "bell_state_phi_plus",
  "source": "grok_generated",
  "original_hash": "sha256hex...",
  "variation": "paraphrase_3"
}
```

**output/training_data/augmented_train.jsonl** (one JSON per line):
```json
{"description": "...", "circuit_qasm": "...", "category": "...", "source": "..."}
```

---

## Pipeline Position

```
quantum-circuit-training-data-generator  (this repo)
    └── output/training_data/
            ├── augmented_train.jsonl
            ├── augmented_val.jsonl
            └── augmented_test.jsonl
                    |
                    v
            quantumgpt-training
```

---

## Hardware Notes

Developed and validated on:
- RTX 4070 12GB / Ryzen 9 5950X / 128GB RAM / Windows 11
- Python 3.11, CUDA 12.x, conda

The scripts run on CPU. GPU is not required for data generation. API calls are rate-limited at 0.3s between batches. A full generation run over 92 categories takes several hours depending on API latency.

---

## License

MIT
