from __future__ import annotations

import unittest

import numpy as np

from prism.cli.common import (
    build_key_value_table,
    ensure_mutually_exclusive,
    resolve_numpy_dtype,
    resolve_prior_source,
)


class CliCommonTests(unittest.TestCase):
    def test_resolve_numpy_dtype(self) -> None:
        self.assertEqual(resolve_numpy_dtype("float32"), np.dtype(np.float32))
        self.assertEqual(resolve_numpy_dtype("float64"), np.dtype(np.float64))
        with self.assertRaisesRegex(ValueError, r"--dtype must be one of: float32, float64"):
            resolve_numpy_dtype("int32")

    def test_resolve_prior_source(self) -> None:
        self.assertEqual(resolve_prior_source("global"), "global")
        self.assertEqual(resolve_prior_source("LABEL"), "label")
        with self.assertRaisesRegex(ValueError, r"--prior-source must be one of: global, label"):
            resolve_prior_source("checkpoint")

    def test_mutually_exclusive_helper(self) -> None:
        ensure_mutually_exclusive(("--gene", None), ("--genes", None))
        ensure_mutually_exclusive(("--gene", ["GeneA"]), ("--genes", None))
        with self.assertRaisesRegex(ValueError, r"--gene and --genes are mutually exclusive"):
            ensure_mutually_exclusive(("--gene", ["GeneA"]), ("--genes", "genes.txt"))

    def test_build_key_value_table(self) -> None:
        table = build_key_value_table(title="Example", values={"A": 1, "B": 2})
        self.assertEqual(table.title, "Example")
        self.assertEqual(len(table.columns), 2)
        self.assertEqual(len(table.rows), 2)


if __name__ == "__main__":
    unittest.main()
