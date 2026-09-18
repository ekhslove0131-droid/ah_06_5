from __future__ import annotations

from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if torch is not None:
    from torch_models import (
        Autoencoder64,
        ClassSpecificFM,
        TokenMLP,
        pack_token_rows,
        train_autoencoder,
        train_fm,
        train_token_mlp,
    )


@unittest.skipUnless(torch is not None, "torch is validated in its dedicated runtime")
class TorchModelTests(unittest.TestCase):
    def test_autoencoder_has_exact_256_64_256_bottleneck(self):
        model = Autoencoder64(input_width=20)
        x = torch.randn(7, 20)
        reconstructed = model(x)
        encoded = model.encode(x)
        self.assertEqual(reconstructed.shape, (7, 20))
        self.assertEqual(encoded.shape, (7, 64))
        self.assertEqual(model.architecture(), [20, 256, 64, 256, 20])
        reconstructed.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_class_specific_fm_produces_26_logits_and_gradients(self):
        model = ClassSpecificFM(symbolic_width=12, missing_width=4, numeric_width=3, classes=26, rank=8)
        symbolic = torch.randn(5, 12)
        missing = torch.randint(0, 2, (5, 4)).float()
        numeric = torch.randn(5, 3)
        logits = model(symbolic, missing, numeric)
        self.assertEqual(logits.shape, (5, 26))
        logits.sum().backward()
        self.assertIsNotNone(model.factors.grad)
        self.assertEqual(tuple(model.factors.shape), (26, 12, 8))

    def test_token_mlp_preserves_pool_contract_and_empty_rows(self):
        batch, tokens = 4, 6
        categorical = {
            "gene": torch.randint(0, 10, (batch, tokens)),
            "kind": torch.randint(0, 5, (batch, tokens)),
            "aa_from": torch.randint(0, 4, (batch, tokens)),
            "aa_to": torch.randint(0, 4, (batch, tokens)),
        }
        position = torch.randn(batch, tokens, 2)
        mask = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ], dtype=torch.bool)
        numeric = torch.randn(batch, 41)
        for pool in ("mean_max", "mean_sqrt_normalized_sum"):
            model = TokenMLP(
                vocab_sizes={"gene": 10, "kind": 5, "aa_from": 4, "aa_to": 4},
                embedding_dim=32, pool=pool, dropout=0.3, numeric_width=41, classes=26,
            )
            logits, pooled = model(categorical, position, mask, numeric, return_pooled=True)
            self.assertEqual(logits.shape, (batch, 26))
            self.assertEqual(pooled.shape, (batch, 64))
            self.assertTrue(torch.equal(pooled[2], torch.zeros_like(pooled[2])))
            logits.square().mean().backward()

    def test_training_helpers_return_finite_probabilities_on_synthetic_cpu(self):
        torch.manual_seed(42)
        X = torch.randn(18, 10).numpy().astype("float32")
        ae = train_autoencoder(X, device="cpu", epochs=2, batch_rows=6, learning_rate=0.001, weight_decay=0.0001, seed=42)
        self.assertTrue(torch.isfinite(ae.encode(torch.from_numpy(X))).all())

        payload = {
            "symbolic": torch.randn(18, 7).numpy().astype("float32"),
            "missing": torch.randint(0, 2, (18, 3)).numpy().astype("float32"),
            "numeric41": torch.randn(18, 4).numpy().astype("float32"),
        }
        y = torch.tensor([index % 3 for index in range(18)]).numpy()
        weights = torch.ones(18).numpy()
        fm = train_fm(payload, y, weights, classes=3, rank=4, device="cpu", epochs=2, batch_rows=6, learning_rate=0.001, weight_decay=0.001, seed=42)
        logits = fm(*(torch.from_numpy(payload[name]) for name in ("symbolic", "missing", "numeric41")))
        self.assertTrue(torch.isfinite(logits).all())

        rows = [[{"gene": 1, "kind": 1, "aa_from": 0, "aa_to": 2, "position": 0.5, "unknown_position": 0.0}], []] * 9
        packed = pack_token_rows(rows)
        self.assertEqual(tuple(packed["mask"].shape), (18, 1))
        token_model = train_token_mlp(
            rows, torch.randn(18, 4).numpy().astype("float32"), y, weights,
            vocab_sizes={"gene": 2, "kind": 2, "aa_from": 2, "aa_to": 2},
            embedding_dim=8, pool="mean_max", dropout=0.0, numeric_width=4,
            classes=3, device="cpu", epochs=2, batch_rows=6,
            learning_rate=0.001, weight_decay=0.001, seed=42,
        )
        categorical = {name: value for name, value in packed.items() if name in {"gene", "kind", "aa_from", "aa_to"}}
        token_logits = token_model(categorical, packed["position"], packed["mask"], torch.randn(18, 4))
        self.assertTrue(torch.isfinite(token_logits).all())


if __name__ == "__main__":
    unittest.main()
