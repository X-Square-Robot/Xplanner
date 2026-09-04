from __future__ import annotations

import unittest

import torch

from scripts.train.v10_continuous_v5.loss_mask_v5 import (
    IGNORE_INDEX,
    add_mask_markers,
    apply_token_mask_and_remove_markers,
    contains_mask_marker,
)


class _ContextSensitiveTokenizer:
    """Small tokenizer double whose ordinary tokens depend on left context."""

    def __init__(self) -> None:
        self._added: dict[str, int] = {}

    def add_special_tokens(self, values, replace_additional_special_tokens=True):
        del replace_additional_special_tokens
        before = len(self._added)
        for token in values["additional_special_tokens"]:
            self._added.setdefault(token, 1000 + len(self._added))
        return len(self._added) - before

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self._added:
            return [self._added[text]]
        # Standalone ordinary strings intentionally have a different encoding
        # from the hand-built context sequence below.
        return [ord(character) + 10 for character in text]


class LossMaskV5Test(unittest.TestCase):
    def test_registered_markers_are_removed_and_value_is_masked(self) -> None:
        text = '{"execution_decision":"Continue"}'
        value_start = text.index('"Continue"')
        value_end = value_start + len('"Continue"')
        marked, markers = add_mask_markers(
            text,
            [(value_start, value_end, "/execution_decision")],
        )
        self.assertTrue(contains_mask_marker(marked))

        tokenizer = _ContextSensitiveTokenizer()
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    markers[0].start_text,
                    markers[0].end_text,
                ]
            },
            replace_additional_special_tokens=False,
        )
        start_id = tokenizer.encode(markers[0].start_text)[0]
        end_id = tokenizer.encode(markers[0].end_text)[0]
        input_ids = torch.tensor([41, start_id, 51, 52, end_id, 61])
        labels = input_ids.clone()

        output_ids, output_labels, stats = apply_token_mask_and_remove_markers(
            input_ids, labels, tokenizer, markers
        )

        self.assertEqual(output_ids.tolist(), [41, 51, 52, 61])
        self.assertEqual(output_labels.tolist(), [41, IGNORE_INDEX, IGNORE_INDEX, 61])
        self.assertEqual(stats["mask_paths"], 1)
        self.assertEqual(stats["masked_value_tokens"], 2)
        self.assertEqual(stats["removed_marker_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
