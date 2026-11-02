"""Convert genuine existing NPY embeddings to ID-aligned NPZ; never infer their encoder."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--array", required=True); p.add_argument("--ids", required=True, help="CSV with id in the EXACT NPY row order")
    p.add_argument("--metadata", required=True, help="JSON recording actual encoder, checkpoint and label_independent")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with open(a.ids, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "id" not in (reader.fieldnames or []): raise ValueError("ID file needs an id column")
        ids = [r["id"].strip() for r in reader]
    features = np.load(a.array, allow_pickle=False).astype(np.float32)
    meta = json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    if not all(x in meta for x in ["encoder", "checkpoint", "label_independent"]): raise ValueError("Incomplete feature provenance")
    if features.ndim != 2 or len(features) != len(ids) or len(set(ids)) != len(ids) or not all(ids) or not np.isfinite(features).all():
        raise ValueError("Invalid embeddings or row-to-ID mapping")
    with open(a.output, "xb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids), features=features, metadata=json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__": main()

