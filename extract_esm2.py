"""Extract frozen ESM-2 protein embeddings using a REAL local checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
from importlib.metadata import version
import numpy as np
import torch
from regsignhg.data import digest

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV gene,sequence with an explicitly selected protein isoform per gene")
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--long-sequence", choices=["error", "chunk"], default="error")
    a = p.parse_args()
    import esm
    if not Path(a.checkpoint).is_file(): raise FileNotFoundError(a.checkpoint)
    with open(a.input, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"gene", "sequence"}.issubset(reader.fieldnames or []): raise ValueError("Expected gene,sequence")
        rows = [(r["gene"].strip(), r["sequence"].strip().upper()) for r in reader]
    if not rows or len({g for g, s in rows}) != len(rows): raise ValueError("Empty input or duplicate gene IDs")
    for gene, sequence in rows:
        if not gene or not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZUO"):
            raise ValueError(f"Invalid protein amino-acid sequence for {gene}")
        if len(sequence) > 1022 and a.long_sequence == "error":
            raise ValueError(f"{gene} exceeds 1022 residues; explicitly choose --long-sequence chunk or define another policy")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(a.checkpoint)
    if not model.__class__.__name__.startswith("ESM2"): raise ValueError("Checkpoint must be an ESM-2 model")
    model = model.eval().to(a.device); converter = alphabet.get_batch_converter()
    features = []
    for gene, sequence in rows:
        pooled, count = None, 0
        for offset in range(0, len(sequence), 1022):
            segment = sequence[offset:offset + 1022]
            _, _, tokens = converter([(gene, segment)])
            result = model(tokens.to(a.device), repr_layers=[model.num_layers], return_contacts=False)
            residues = result["representations"][model.num_layers][0, 1:len(segment) + 1]
            value = residues.sum(0).cpu()
            pooled = value if pooled is None else pooled + value; count += len(segment)
        features.append((pooled / count).numpy())
    meta = {"encoder": "ESM-2", "checkpoint": Path(a.checkpoint).name, "checkpoint_sha256": digest(a.checkpoint),
            "label_independent": True, "pooling": "final-layer residue mean excluding special tokens; length-weighted across chunks",
            "long_sequence": a.long_sequence, "max_chunk_residues": 1022, "fair_esm_version": version("fair-esm"),
            "input_csv_sha256": digest(a.input), "source_code_sha256": digest(__file__), "device": a.device}
    with open(a.output, "xb") as stream:
        np.savez_compressed(stream, ids=np.asarray([g for g, s in rows]), features=np.stack(features).astype(np.float32), metadata=json.dumps(meta))

if __name__ == "__main__": main()

