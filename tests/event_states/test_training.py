from __future__ import annotations

import unittest

from x_planner.data.event_states.training import _generation_authorization, _strip_release_arguments


class TrainV5GenerationGateTest(unittest.TestCase):
    def test_full_generation_needs_no_partial_override(self) -> None:
        result = _generation_authorization(
            {"partial": False, "content_digest": "a" * 64}, []
        )
        self.assertFalse(result["partial"])
        self.assertFalse(result["partial_authorized"])

    def test_partial_generation_fails_closed_without_both_pins(self) -> None:
        manifest = {"partial": True, "content_digest": "b" * 64}
        with self.assertRaises(PermissionError):
            _generation_authorization(manifest, [])
        with self.assertRaisesRegex(ValueError, "expected_content_digest"):
            _generation_authorization(
                manifest, ["--allow-partial-generation"]
            )

    def test_partial_generation_accepts_exact_digest_and_strips_private_args(self) -> None:
        digest = "c" * 64
        arguments = [
            "--snapshot", "/tmp/snapshot",
            "--allow-partial-generation",
            "--expected-content-digest", digest,
            "--max_steps", "30",
        ]
        result = _generation_authorization(
            {"partial": True, "content_digest": digest}, arguments
        )
        self.assertTrue(result["partial_authorized"])
        self.assertEqual(
            _strip_release_arguments(arguments),
            ["--snapshot", "/tmp/snapshot", "--max_steps", "30"],
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            _generation_authorization(
                {"partial": True, "content_digest": digest},
                [
                    "--allow-partial-generation",
                    "--expected-content-digest=" + "d" * 64,
                ],
            )


if __name__ == "__main__":
    unittest.main()
