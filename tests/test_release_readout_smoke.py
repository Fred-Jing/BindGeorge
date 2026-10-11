"""Data-free CPU checks for the primary single/pair readout interfaces."""

import io
import unittest

import torch

from scripts.smoke_test_multimodal_v2 import inputs_for_variant, make_batch
from training.multimodal_model import PairConditionedBinderClassifier


class ReleaseReadoutSmokeTests(unittest.TestCase):
    def check_variant(self, variant):
        batch = make_batch()
        model = PairConditionedBinderClassifier(
            variant=variant,
            projection_dropout=0.0,
            attention_dropout=0.0,
            classifier_dropout=0.0,
        )
        inputs = inputs_for_variant(batch, variant)
        logits = model(**inputs)
        self.assertEqual(logits.shape, (3,))
        self.assertTrue(torch.isfinite(logits).all())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([1.0, 0.0, 1.0])
        )
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

        # Check the export contract without loading or writing a real checkpoint.
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = PairConditionedBinderClassifier(
            variant=variant,
            projection_dropout=0.0,
            attention_dropout=0.0,
            classifier_dropout=0.0,
        )
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(**inputs), restored(**inputs), rtol=0, atol=0)

    def test_single(self):
        self.check_variant("af3_single")

    def test_pair(self):
        self.check_variant("af3_pair")

    def test_single_pair(self):
        self.check_variant("af3_single_pair")


if __name__ == "__main__":
    unittest.main()
