from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_interactions(path):
    records, seen = [], set()
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"drug", "gene", "sign"}.issubset(reader.fieldnames or []):
            raise ValueError("interactions.csv requires drug,gene,sign columns")
        for line, row in enumerate(reader, 2):
            drug, gene, sign = (row[k].strip() for k in ["drug", "gene", "sign"])
            if not drug or not gene or sign not in ["1", "+1", "-1"]:
                raise ValueError(f"Invalid entity or sign at CSV line {line}")
            if (drug, gene) in seen:
                raise ValueError(f"Duplicate or conflicting pair at line {line}: {drug}/{gene}; resolve before splitting")
            seen.add((drug, gene))
            records.append((drug, gene, int(sign)))
    if not records or len({r[2] for r in records}) < 2:
        raise ValueError("The signed dataset must contain both signs")
    return records

def read_feature_file(path, required_ids, expected_encoder, synthetic=False):
    with np.load(path, allow_pickle=False) as f:
        if not {"ids", "features", "metadata"}.issubset(f.files):
            raise ValueError(f"{path}: NPZ requires ids,features,metadata")
        ids = f["ids"]
        values = np.asarray(f["features"], dtype=np.float32)
        meta = json.loads(str(f["metadata"].item()))
    if ids.ndim != 1 or ids.dtype.kind not in "US":
        raise ValueError("Feature IDs must be a 1-D string array, never pickle/object")
    ids = ids.astype(str).tolist()
    if len(set(ids)) != len(ids) or values.ndim != 2 or len(ids) != len(values) or values.shape[1] < 1:
        raise ValueError("Invalid feature shape or duplicate feature IDs")
    if not np.isfinite(values).all():
        raise ValueError("Features contain NaN or infinity")
    if not isinstance(meta, dict) or not meta.get("checkpoint") or meta.get("label_independent") is not True:
        raise ValueError("Feature metadata must record checkpoint and label_independent=true")
    if not synthetic and meta.get("encoder") != expected_encoder:
        raise ValueError(f"Expected {expected_encoder} features; got {meta.get('encoder')}. Do not relabel DGI/identity features")
    positions = {name: i for i, name in enumerate(ids)}
    missing = set(required_ids) - positions.keys()
    if missing:
        raise ValueError(f"Missing features for {len(missing)} entities, e.g. {sorted(missing)[:3]}")
    return values[[positions[name] for name in required_ids]], meta

@dataclass
class Dataset:
    drugs: list
    genes: list
    pairs: np.ndarray
    labels: np.ndarray
    drug_features: np.ndarray
    gene_features: np.ndarray
    manifest: dict

def load_dataset(directory, allow_synthetic=False):
    directory = Path(directory)
    paths = {name: directory / name for name in ["interactions.csv", "drug_features.npz", "gene_features.npz", "provenance.json"]}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = json.loads(paths["provenance.json"].read_text(encoding="utf-8"))
    for key in ["dataset_name", "source_version", "source_files", "processing_description"]:
        if not provenance.get(key):
            raise ValueError(f"provenance.json requires a real {key}")
    if not isinstance(provenance.get("synthetic"), bool):
        raise ValueError("provenance.synthetic must be explicitly true or false")
    synthetic = provenance["synthetic"]
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic smoke-test data requires --allow-synthetic")
    records = read_interactions(paths["interactions.csv"])
    drugs, genes = sorted({r[0] for r in records}), sorted({r[1] for r in records})
    dmap, gmap = ({v: i for i, v in enumerate(names)} for names in [drugs, genes])
    pairs = np.asarray([(dmap[d], gmap[g]) for d, g, s in records], dtype=np.int64)
    labels = np.asarray([int(s == 1) for d, g, s in records], dtype=np.int64)
    dx, dm = read_feature_file(paths["drug_features.npz"], drugs, "Uni-Mol", synthetic)
    gx, gm = read_feature_file(paths["gene_features.npz"], genes, "ESM-2", synthetic)
    manifest = {
        "provenance": provenance, "drug_feature_metadata": dm, "gene_feature_metadata": gm,
        "counts": {"drugs": len(drugs), "genes": len(genes), "positive_edges": int(labels.sum()),
                   "negative_edges": int((labels == 0).sum()), "edges": len(labels)},
        "files": {name: {"bytes": path.stat().st_size, "md5": digest(path, "md5"), "sha256": digest(path)} for name, path in paths.items()},
    }
    return Dataset(drugs, genes, pairs, labels, dx, gx, manifest)

def make_splits(labels, folds, split_seed, val_ratio):
    if np.bincount(labels, minlength=2).min() < folds:
        raise ValueError("Insufficient examples per sign for stratified folds")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    result = []
    for number, (development, test) in enumerate(cv.split(np.zeros(len(labels)), labels), 1):
        train, val = train_test_split(development, test_size=val_ratio,
                                      stratify=labels[development], random_state=split_seed + number)
        result.append({"fold": number, "train": train.tolist(), "validation": val.tolist(), "test": test.tolist()})
    return result

