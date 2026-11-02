from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import itertools
import json
from regsignhg.data import read_interactions, digest

def main():
    p = argparse.ArgumentParser(description="Compare source CSV bytes and signed entity records, not just hashes")
    p.add_argument("csv_files", nargs="+"); p.add_argument("--output"); p.add_argument("--fail-on-identical", action="store_true")
    a = p.parse_args()
    if len(a.csv_files) < 2: p.error("Provide at least two CSV files")
    entries = [{"path": f, "sha256": digest(f), "md5": digest(f, "md5"), "records": set(read_interactions(f))} for f in a.csv_files]
    pairs = []
    for x, y in itertools.combinations(entries, 2):
        pairs.append({"left": x["path"], "right": y["path"], "byte_identical": Path(x["path"]).read_bytes() == Path(y["path"]).read_bytes(),
                      "same_signed_records": x["records"] == y["records"], "shared_signed_records": len(x["records"] & y["records"]),
                      "note": "Entity strings must use canonical biological IDs; separately remapped integer IDs are not comparable"})
    result = {"files": [{k: v for k, v in x.items() if k != "records"} for x in entries], "comparisons": pairs}
    text = json.dumps(result, indent=2)
    if a.output: Path(a.output).write_text(text, encoding="utf-8")
    print(text)
    if a.fail_on_identical and any(x["same_signed_records"] for x in pairs): raise SystemExit(2)

if __name__ == "__main__": main()

