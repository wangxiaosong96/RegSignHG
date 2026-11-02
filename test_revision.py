from pathlib import Path
from copy import deepcopy
import csv
import json
import tempfile
import unittest
import numpy as np
import torch
from regsignhg.config import load_config
from regsignhg.data import load_dataset, make_splits, read_interactions
from regsignhg.model import RegSignHG, build_graph
from regsignhg.losses import positive_view, corrupted_view, graph_contrastive, dropout_edges
from regsignhg.engine import run_training, evaluate_checkpoint, train_one
from scripts.make_toy_data import make_toy_data

ROOT = Path(__file__).resolve().parents[1]

class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = make_toy_data(Path(self.temp.name) / "data")

    def tearDown(self): self.temp.cleanup()

    def test_synthetic_data_requires_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, "allow-synthetic"):
            load_dataset(self.path)

    def test_feature_row_permutation_is_aligned_by_id(self):
        before = load_dataset(self.path, True)
        file = self.path / "drug_features.npz"
        with np.load(file, allow_pickle=False) as f:
            ids, values, meta = f["ids"].copy(), f["features"].copy(), f["metadata"].copy()
        order = np.arange(len(ids))[::-1]
        np.savez_compressed(file, ids=ids[order], features=values[order], metadata=meta)
        after = load_dataset(self.path, True)
        np.testing.assert_array_equal(before.drug_features, after.drug_features)

    def test_duplicate_pair_cannot_cross_folds(self):
        file = self.path / "interactions.csv"
        with open(file, newline="", encoding="utf-8") as s: rows = list(csv.reader(s))
        with open(file, "a", newline="", encoding="utf-8") as s: csv.writer(s).writerow(rows[1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_interactions(file)

    def test_zero_label_is_not_silently_a_negative_sign(self):
        file = self.path / "interactions.csv"
        file.write_text("drug,gene,sign\nd,g,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            read_interactions(file)

    def test_splits_are_disjoint_cover_test_set_once_and_fixed(self):
        data = load_dataset(self.path, True)
        a = make_splits(data.labels, 5, 11, .2)
        self.assertEqual(a, make_splits(data.labels, 5, 11, .2))
        tests = []
        for split in a:
            tr, va, te = (set(split[k]) for k in ["train", "validation", "test"])
            self.assertFalse(tr & va or tr & te or va & te)
            self.assertEqual(tr | va | te, set(range(len(data.labels))))
            tests.extend(te)
        self.assertEqual(sorted(tests), list(range(len(data.labels))))

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(23)
        cfg = load_config(ROOT / "configs/smoke.yaml")
        cfg["model"]["dropout"] = 0.0
        dx, gx = torch.randn(9, 10), torch.randn(7, 14)
        pairs = torch.tensor([(d, g) for d in range(9) for g in range(7)])
        y = (pairs[:, 0] + pairs[:, 1]) % 2
        pos, neg = build_graph(pairs, y)
        model = RegSignHG(9, 7, 10, 14, cfg["model"])
        return cfg, model, dx, gx, pairs, y, pos, neg

    def test_decoder_has_two_probabilities_and_prototypes_learn(self):
        cfg, model, dx, gx, pairs, y, pos, neg = self.fixture()
        logits, encoded = model(dx, gx, pos, neg, pairs)
        self.assertEqual(tuple(logits.shape), (63, 2))
        torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(63))
        torch.nn.functional.cross_entropy(logits, 1 - y).backward()
        norm = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if "prototypes" in n and p.grad is not None)
        self.assertGreater(norm, 1e-10)
        for channels in encoded["incidences"]:
            for H in channels:
                self.assertTrue(bool(((H > 0).sum(1) <= cfg["model"]["top_r"]).all()))

    def test_graph_contrast_has_real_negative_and_nonzero_gradient(self):
        cfg, model, dx, gx, pairs, y, pos, neg = self.fixture()
        a = positive_view(model, dx, gx, pos, neg, cfg["augmentation"])["all"]
        b = positive_view(model, dx, gx, pos, neg, cfg["augmentation"])["all"]
        negative = corrupted_view(model, dx, gx, pos, neg, cfg["augmentation"])["all"]
        self.assertGreater(float((negative.mean(0) - b.mean(0)).abs().sum()), 1e-7)
        loss = graph_contrastive(a, b, [negative], .2)
        loss.backward()
        total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        self.assertGreater(total, 1e-7)

    def test_positive_edge_dropout_never_introduces_new_edges(self):
        cfg, model, dx, gx, pairs, y, pos, neg = self.fixture()
        for edges in [pos, neg]:
            dropped = dropout_edges(edges, .5)
            original = set(map(tuple, edges.T.tolist()))
            self.assertTrue(set(map(tuple, dropped.T.tolist())) <= original)

    def test_no_hyper_is_actual_pairwise_architecture(self):
        cfg, model, dx, gx, pairs, y, pos, neg = self.fixture()
        cfg["model"]["mode"] = "pairwise"
        pairwise = RegSignHG(9, 7, 10, 14, cfg["model"])
        self.assertFalse(any("hypergraph" in n or "prototypes" in n for n, p in pairwise.named_parameters()))
        logits, encoded = pairwise(dx, gx, pos, neg, pairs)
        self.assertEqual(tuple(logits.shape), (63, 2)); self.assertEqual(encoded["intents"], [[], []])

class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_dir = make_toy_data(self.root / "data")
        self.cfg = load_config(ROOT / "configs/smoke.yaml")
        self.cfg["dataset"]["data_dir"] = str(self.data_dir)
        self.cfg["train"].update(folds=2, epochs=2, patience=2)

    def tearDown(self): self.temp.cleanup()

    def test_checkpoint_reloads_exactly_and_csv_metrics_match(self):
        out = self.root / "run"
        result = run_training(self.cfg, out, allow_synthetic=True)
        self.assertEqual(result["run_count"], 2)
        folder = out / "seed_7/fold_1"
        metrics, probabilities, ids = evaluate_checkpoint(folder / "best.pt", self.data_dir, allow_synthetic=True)
        expected = json.loads((folder / "metrics.json").read_text())
        for name in metrics: self.assertAlmostEqual(metrics[name], expected[name], places=12)
        with open(folder / "test_predictions.csv", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        np.testing.assert_allclose(probabilities, [float(r["p_activation"]) for r in rows], rtol=1e-9, atol=1e-9)
        self.assertEqual(ids, [int(r["row_index"]) for r in rows])
        for filename in ["environment.json", "resolved_config.json", "data_audit.json", "all_splits.json", "runs.csv", "summary.json"]:
            self.assertTrue((out / filename).is_file())

    def test_heldout_label_changes_do_not_change_training_or_checkpoint(self):
        data = load_dataset(self.data_dir, True)
        split = make_splits(data.labels, 2, 2026, .2)[0]
        altered = deepcopy(data); altered.labels[split["test"]] = 1 - altered.labels[split["test"]]
        first, second = self.root / "first", self.root / "second"
        train_one(data, split, self.cfg, 7, first, torch.device("cpu"))
        train_one(altered, split, self.cfg, 7, second, torch.device("cpu"))
        a = torch.load(first / "best.pt", weights_only=True); b = torch.load(second / "best.pt", weights_only=True)
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        for name in a["state_dict"]: torch.testing.assert_close(a["state_dict"][name], b["state_dict"][name], rtol=0, atol=0)

if __name__ == "__main__": unittest.main()
