from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from regsignhg.data import load_dataset, read_interactions, digest

def main():
    p = argparse.ArgumentParser(description="Audit real CSV counts, identity alignment and input checksums; no training")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir"); group.add_argument("--interactions")
    p.add_argument("--output"); p.add_argument("--allow-synthetic", action="store_true")
    a = p.parse_args()
    if a.data_dir:
        result = load_dataset(a.data_dir, a.allow_synthetic).manifest
    else:
        rows = read_interactions(a.interactions)
        result = {"drugs": len({r[0] for r in rows}), "genes": len({r[1] for r in rows}),
                  "positive_edges": sum(s == 1 for d, g, s in rows), "negative_edges": sum(s == -1 for d, g, s in rows),
                  "edges": len(rows), "md5": digest(a.interactions, "md5"), "sha256": digest(a.interactions),
                  "note": "CSV counts only; feature provenance is not verified by this mode"}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__": main()

