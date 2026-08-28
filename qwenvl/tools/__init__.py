# Copyright (c) 2026
"""Standalone command-line utilities for Qwen3.5-VL SFT.

* ``python -m qwenvl.tools.precompute_lengths`` -- warm the length-balanced
  packing cache offline (before any model is loaded);
* ``python -m qwenvl.tools.smoke_test_packing``  -- GPU correctness check for the
  neat-packing patch (no cross-document leakage + finite forward/backward).
"""
