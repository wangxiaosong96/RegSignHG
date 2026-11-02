"""Generate synthetic fixtures ONLY; never a biomedical benchmark."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

def make_toy_data(directory, seed=7):
    out = Path(directory); out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    drugs, genes = [f"TOY_D{i:02d}" for i in range(16)], [f"TOY_G{i:02d}" for i in range(12)]
    dlatent, glatent = rng.normal(size=(16, 4)), rng.normal(size=(12, 4))
    candidate = np.array([(d, g) for d in range(16) for g in range(12)])
    selected = candidate[rng.permutation(len(candidate))[:160]]
    scores = (dlatent[selected[:, 0]] * glatent[selected[:, 1]]).sum(1)
    signs = np.where(scores >= np.median(scores), 1, -1)
    with open(out / "interactions.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["drug", "gene", "sign"])
        for (d, g), sign in zip(selected, signs): writer.writerow([drugs[d], genes[g], int(sign)])
    for filename, ids, values in [("drug_features.npz", drugs, dlatent @ rng.normal(size=(4, 10))),
                                  ("gene_features.npz", genes, glatent @ rng.normal(size=(4, 14)))]:
        meta = {"encoder": "synthetic", "checkpoint": "none synthetic fixture", "label_independent": True,
                "generation_seed": seed, "purpose": "software smoke test only"}
        np.savez_compressed(out / filename, ids=np.asarray(ids), features=np.asarray(values, dtype=np.float32), metadata=json.dumps(meta))
    provenance = {"dataset_name": "TOY_SYNTHETIC", "source_version": "synthetic_v1", "source_files": ["make_toy_data.py"],
                  "processing_description": "Random latent arrays and deterministic signed pairs for software validation only", "synthetic": True}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", default="data/toy"); p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(); print(make_toy_data(a.out, a.seed))

