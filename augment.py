#!/usr/bin/env python3
"""
Description Augmentation
Takes master circuits and generates paraphrased descriptions for training data expansion.

For each circuit, generates N paraphrases of the description while keeping the QASM
unchanged. This multiplies dataset size by (N+1)x without requiring additional API calls
to generate new circuits.
"""
import json
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")


class DescriptionAugmenter:
    def __init__(self, input_file="output/master_circuits.json",
                 output_file="output/augmented_circuits.json",
                 paraphrases_per_circuit=10):
        self.input_file = input_file
        self.output_file = output_file
        self.paraphrases_per_circuit = paraphrases_per_circuit

        with open(input_file, 'r') as f:
            self.circuits = json.load(f)

        print(f"Loaded {len(self.circuits)} unique circuits")

    def _get_qasm(self, circuit):
        """Handles both 'qasm' (master) and 'circuit_qasm' (augmented) key names."""
        return circuit.get('qasm') or circuit.get('circuit_qasm', '')

    def _clean_json(self, text):
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    text = part
                    break
        text = text.replace("json", "", 1).strip()
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace('\ufeff', '')
        return text.strip()

    def _generate_paraphrases(self, description, qasm, category, max_retries=3):
        prompt = f"""Generate {self.paraphrases_per_circuit} different paraphrases of this quantum circuit description.

Original description: "{description}"
Category: {category}

Requirements:
- {self.paraphrases_per_circuit} DIFFERENT ways to describe the same circuit
- Vary vocabulary, sentence structure, technical level
- Keep meaning identical - same gates, same operations
- Use plain ASCII only (no unicode, no special symbols)
- Make them natural and varied

Examples of variation:
- "Create Bell state" -> "Generate entangled Bell pair", "Prepare two-qubit Bell state"
- "Apply Hadamard gate" -> "Use H gate", "Implement Hadamard operation"

Return ONLY this JSON:
{{
  "paraphrases": [
    "first paraphrase",
    "second paraphrase"
  ]
}}"""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="grok-4.20-0309-reasoning",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at paraphrasing quantum circuit descriptions. Generate diverse, natural variations. Return ONLY valid JSON with plain ASCII text."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.9,
                    max_tokens=1500,
                    timeout=60
                )

                content = self._clean_json(response.choices[0].message.content)
                data = json.loads(content)

                if "paraphrases" not in data:
                    raise ValueError("No 'paraphrases' key")

                paraphrases = data["paraphrases"]

                if len(paraphrases) < self.paraphrases_per_circuit:
                    print(f"      Only got {len(paraphrases)}/{self.paraphrases_per_circuit}")

                return paraphrases[:self.paraphrases_per_circuit]

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      Retry {attempt+2}/{max_retries}: {e}")
                    time.sleep(2)
                else:
                    print(f"      Failed -- using original as fallback")
                    return [description] * self.paraphrases_per_circuit

        return [description] * self.paraphrases_per_circuit

    def augment_all(self, start_from=0, save_every=10):
        augmented = []

        print(f"\n{'='*70}")
        print(f"DESCRIPTION AUGMENTATION")
        print(f"{'='*70}")
        print(f"Input:           {self.input_file}")
        print(f"Output:          {self.output_file}")
        print(f"Circuits:        {len(self.circuits)}")
        print(f"Paraphrases:     {self.paraphrases_per_circuit}")
        print(f"Expected output: {len(self.circuits) * (self.paraphrases_per_circuit + 1):,}")
        print(f"Starting from:   {start_from}")
        print(f"{'='*70}\n")

        for i, circuit in enumerate(self.circuits[start_from:], start=start_from):
            print(f"[{i+1}/{len(self.circuits)}] {circuit['description'][:60]}...")

            qasm = self._get_qasm(circuit)

            augmented.append({
                "description":   circuit['description'],
                "circuit_qasm":  qasm,
                "category":      circuit['category'],
                "source":        circuit.get('source', 'unknown'),
                "original_hash": circuit.get('hash', ''),
                "variation":     "original"
            })

            print(f"    Generating {self.paraphrases_per_circuit} paraphrases...")
            paraphrases = self._generate_paraphrases(
                circuit['description'], qasm, circuit['category']
            )

            for j, para in enumerate(paraphrases, 1):
                augmented.append({
                    "description":   para,
                    "circuit_qasm":  qasm,
                    "category":      circuit['category'],
                    "source":        circuit.get('source', 'unknown'),
                    "original_hash": circuit.get('hash', ''),
                    "variation":     f"paraphrase_{j}"
                })

            print(f"    {len(paraphrases) + 1} variations | total so far: {len(augmented):,}")

            if (i + 1) % save_every == 0:
                with open(self.output_file, 'w') as f:
                    json.dump(augmented, f, indent=2)
                print(f"    Saved ({len(augmented):,} samples)")

            time.sleep(0.3)

        with open(self.output_file, 'w') as f:
            json.dump(augmented, f, indent=2)

        print(f"\n{'='*70}")
        print(f"AUGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"Unique circuits:  {len(self.circuits)}")
        print(f"Total samples:    {len(augmented):,}")
        print(f"Multiplier:       {len(augmented)/len(self.circuits):.1f}x")
        print(f"Saved to:         {self.output_file}")

        avg_desc = sum(len(s['description']) for s in augmented) / len(augmented)
        avg_qasm = sum(len(s['circuit_qasm']) for s in augmented) / len(augmented)
        est_tokens = ((avg_desc + avg_qasm + 50) / 4) * len(augmented)
        print(f"\nEstimated training corpus:")
        print(f"  Avg description: {avg_desc:.0f} chars")
        print(f"  Avg QASM:        {avg_qasm:.0f} chars")
        print(f"  Estimated tokens:{est_tokens:,.0f}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Description Augmentation")
    parser.add_argument("--input",       default="output/master_circuits.json",
                        help="Input master circuits JSON (default: output/master_circuits.json)")
    parser.add_argument("--output",      default="output/augmented_circuits.json",
                        help="Output augmented JSON (default: output/augmented_circuits.json)")
    parser.add_argument("--paraphrases", type=int, default=10,
                        help="Paraphrases per circuit (default: 10)")
    parser.add_argument("--start",       type=int, default=0,
                        help="Circuit index to start from (default: 0)")
    parser.add_argument("--save-every",  type=int, default=10,
                        help="Save every N circuits (default: 10)")
    args = parser.parse_args()

    augmenter = DescriptionAugmenter(
        input_file=args.input,
        output_file=args.output,
        paraphrases_per_circuit=args.paraphrases
    )
    augmenter.augment_all(start_from=args.start, save_every=args.save_every)


if __name__ == "__main__":
    main()
