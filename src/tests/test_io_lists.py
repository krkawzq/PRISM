from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from prism.io import (
    GeneListSpec,
    read_gene_list_spec,
    read_string_list,
    read_string_list_spec,
    write_string_list_spec,
)


class IoListsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_read_gene_list_legacy_training_json(self) -> None:
        path = self.root / "legacy_gene_list.json"
        path.write_text(
            json.dumps(
                {
                    "gene_names": ["GeneA", "GeneA", "GeneB"],
                    "method": "legacy",
                    "top_k": 2,
                    "gene_indices": [0, 1, 2],
                }
            ),
            encoding="utf-8",
        )

        spec = read_gene_list_spec(path)

        self.assertEqual(spec.gene_names, ["GeneA", "GeneB"])
        self.assertEqual(spec.method, "legacy")
        self.assertEqual(spec.metadata["top_k"], 2)
        self.assertEqual(spec.metadata["gene_indices"], [0, 1, 2])

    def test_read_string_list_supports_json_array_and_object(self) -> None:
        array_path = self.root / "labels_array.json"
        array_path.write_text(json.dumps(["ctrl", "stim", "ctrl"]), encoding="utf-8")
        object_path = self.root / "labels_object.json"
        write_string_list_spec(
            object_path,
            spec=read_string_list_spec(array_path),
        )

        self.assertEqual(read_string_list(array_path), ["ctrl", "stim"])
        self.assertEqual(read_string_list(object_path), ["ctrl", "stim"])

    def test_gene_list_json_error_message_is_normalized(self) -> None:
        path = self.root / "broken_gene_list.json"
        path.write_text("{", encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            r"invalid gene-list file .* malformed JSON",
        ):
            read_gene_list_spec(path)

    def test_gene_list_accepts_generic_items_field(self) -> None:
        path = self.root / "generic_items.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "string_list",
                    "items": ["GeneA", "GeneB", "GeneA"],
                }
            ),
            encoding="utf-8",
        )

        spec = read_gene_list_spec(path)

        self.assertEqual(spec.gene_names, ["GeneA", "GeneB"])
        self.assertEqual(spec.kind, "string_list")


if __name__ == "__main__":
    unittest.main()
