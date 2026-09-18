from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from partition_contract import build_partitions, verify_partition_bundle, PartitionContractError


def synthetic_contract():
    classes = [f"C{i:02d}" for i in range(26)]
    ids, y, groups, outer = [], [], [], []
    for class_index, label in enumerate(classes):
        for repeat in range(15):
            ids.append(f"ID-{class_index:02d}-{repeat:02d}")
            y.append(label)
            groups.append(f"G-{class_index:02d}-{repeat:02d}")
            outer.append(repeat % 5)
    return (
        np.asarray(ids), np.asarray(y), np.asarray(groups),
        np.asarray(outer, dtype=np.int16), classes,
    )


class PartitionTests(unittest.TestCase):
    def test_nested_partitions_are_deterministic_grouped_and_complete(self):
        ids, y, groups, outer, classes = synthetic_contract()
        first = build_partitions(ids, y, groups, outer, classes)
        second = build_partitions(ids, y, groups, outer, classes)
        verify_partition_bundle(first, ids, y, groups, outer, classes)
        self.assertEqual(first, second)
        self.assertEqual(first["outer_seed"], 43)
        self.assertEqual([row["inner_seed"] for row in first["outer"]], [2026091400+i for i in range(5)])
        self.assertEqual(first["outer"][2]["inner"][1]["subinner_seed"], 2026091521)
        for outer_row in first["outer"]:
            valid = set(outer_row["valid_indices"])
            train = set(outer_row["train_indices"])
            self.assertFalse(train & valid)
            self.assertEqual(train | valid, set(range(len(ids))))
            self.assertEqual({y[index] for index in valid}, set(classes))
            inner_valid = []
            for inner_row in outer_row["inner"]:
                self.assertFalse(set(inner_row["train_indices"]) & set(inner_row["valid_indices"]))
                inner_valid.extend(inner_row["valid_indices"])
                sub_valid = [index for part in inner_row["subinner"] for index in part["valid_indices"]]
                self.assertEqual(sorted(sub_valid), sorted(inner_row["train_indices"]))
            self.assertEqual(sorted(inner_valid), sorted(outer_row["train_indices"]))

    def test_duplicate_id_crossed_group_and_reordered_classes_fail_closed(self):
        ids, y, groups, outer, classes = synthetic_contract()
        duplicate = ids.copy(); duplicate[1] = duplicate[0]
        with self.assertRaisesRegex(PartitionContractError, "ID"):
            build_partitions(duplicate, y, groups, outer, classes)
        crossed = groups.copy(); crossed[1] = crossed[0]
        with self.assertRaisesRegex(PartitionContractError, "group"):
            build_partitions(ids, y, crossed, outer, classes)
        with self.assertRaisesRegex(PartitionContractError, "class order"):
            build_partitions(ids, y, groups, outer, list(reversed(classes)))

    def test_missing_class_or_tampered_receipt_is_rejected(self):
        ids, y, groups, outer, classes = synthetic_contract()
        missing = y.copy(); missing[missing == classes[-1]] = classes[-2]
        with self.assertRaisesRegex(PartitionContractError, "class"):
            build_partitions(ids, missing, groups, outer, classes)
        bundle = build_partitions(ids, y, groups, outer, classes)
        tampered = copy.deepcopy(bundle)
        tampered["outer"][0]["valid_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(PartitionContractError, "receipt"):
            verify_partition_bundle(tampered, ids, y, groups, outer, classes)


if __name__ == "__main__":
    unittest.main()
