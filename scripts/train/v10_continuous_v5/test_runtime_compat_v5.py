from __future__ import annotations

import unittest

from scripts.train.v10_continuous.runtime import (
    _image_token_id_from_model,
    _needs_image_grid_compatibility,
)


class ImageGridCompatibilityTest(unittest.TestCase):
    def test_b30z_is_always_covered(self) -> None:
        self.assertTrue(_needs_image_grid_compatibility("2.9.1+cu128", (10, 3)))

    def test_dev13_torch_210_cuda_reduction_is_covered(self) -> None:
        self.assertTrue(_needs_image_grid_compatibility("2.10.0+cu128", (8, 0)))

    def test_unaffected_build_is_not_patched(self) -> None:
        self.assertFalse(_needs_image_grid_compatibility("2.9.1+cu128", (8, 0)))

    def test_image_token_is_found_through_deepspeed_like_wrapper(self) -> None:
        class Value:
            pass

        base = Value()
        base.config = Value()
        base.config.text_config = Value()
        base.config.text_config.image_token_id = 248056
        middle = Value()
        middle.config = Value()  # A non-HF wrapper config must not stop search.
        middle.module = base
        outer = Value()
        outer.module = middle
        self.assertEqual(_image_token_id_from_model(outer), 248056)

    def test_image_token_wrapper_cycle_fails_safely(self) -> None:
        class Value:
            pass

        value = Value()
        value.module = value
        self.assertIsNone(_image_token_id_from_model(value))


if __name__ == "__main__":
    unittest.main()
