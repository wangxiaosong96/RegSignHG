"""Use the official UniMolRepr API, explicitly loading and hashing real local weights."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import inspect
import json
from importlib.metadata import version
import random
import numpy as np
import torch
from regsignhg.data import digest

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV drug,smiles")
    p.add_argument("--checkpoint", required=True, help="A checkpoint compatible with the installed unimolv1 model")
    p.add_argument("--output", required=True); p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--remove-hydrogens", action="store_true")
    a = p.parse_args()
    from rdkit import Chem
    from unimol_tools import UniMolRepr
    if not Path(a.checkpoint).is_file(): raise FileNotFoundError(a.checkpoint)
    with open(a.input, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"drug", "smiles"}.issubset(reader.fieldnames or []): raise ValueError("Expected drug,smiles")
        rows = [(r["drug"].strip(), r["smiles"].strip()) for r in reader]
    if not rows or len({d for d, s in rows}) != len(rows) or any(not d for d, s in rows):
        raise ValueError("Empty input, empty IDs or duplicate drug IDs")
    smiles = []
    for drug, text in rows:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None or molecule.GetNumAtoms() == 0: raise ValueError(f"Invalid SMILES: {drug}")
        smiles.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    adapter = UniMolRepr(data_type="molecule", remove_hs=a.remove_hydrogens,
                         model_name="unimolv1", use_cuda=a.device == "cuda", use_ddp=False)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if "model" in state: state = state["model"]
    elif "state_dict" in state: state = state["state_dict"]
    adapter.model.load_state_dict(state, strict=True)
    adapter.model.eval()
    result = adapter.get_repr(smiles, return_atomic_reprs=False)
    features = np.asarray(result["cls_repr"], dtype=np.float32)
    if features.ndim != 2 or len(features) != len(rows) or not np.isfinite(features).all():
        raise ValueError("Uni-Mol returned invalid representations; no failed molecules are silently removed")
    meta = {"encoder": "Uni-Mol", "model_name": "unimolv1", "checkpoint": Path(a.checkpoint).name,
            "checkpoint_sha256": digest(a.checkpoint), "label_independent": True,
            "pooling": "CLS molecular representation", "canonicalization": "RDKit isomeric canonical SMILES, all fragments retained",
            "conformers": "installed unimol_tools upstream defaults; seed 42", "remove_hs": a.remove_hydrogens,
            "unimol_tools_version": version("unimol_tools"), "input_csv_sha256": digest(a.input),
            "adapter_source_sha256": digest(inspect.getfile(UniMolRepr)), "source_code_sha256": digest(__file__), "device": a.device}
    with open(a.output, "xb") as stream:
        np.savez_compressed(stream, ids=np.asarray([d for d, s in rows]), features=features, metadata=json.dumps(meta))

if __name__ == "__main__": main()

