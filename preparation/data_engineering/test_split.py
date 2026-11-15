import unittest

from split import aligned_size, check_splits, fast_reference, last_box, quota, training_order, validation_selection


class SplitChecks(unittest.TestCase):
    def test_batch_alignment_does_not_require_power_of_two(self):
        self.assertEqual(aligned_size(10000, 128), 9984)
        self.assertEqual(aligned_size(10240, 128), 10240)
        self.assertEqual(9984//128, 78)

    def test_validation_difficulty_quotas(self):
        counts = {"easy": 4496, "medium": 21496, "hard": 7244}
        self.assertEqual(quota(512, counts, counts), {"easy": 69, "medium": 331, "hard": 112})

    def test_rare_cell_minimum_and_capacity(self):
        q = quota(10, {"common": 999, "rare": 1}, {"common": 20, "rare": 1}, minimum=1)
        self.assertEqual(q, {"common": 9, "rare": 1})
        with self.assertRaises(ValueError):
            quota(23, {"common": 999, "rare": 1}, {"common": 20, "rare": 1})

    def test_reference_requires_actual_terminal_answer(self):
        text = r"An intermediate value is \boxed{3}. Finally \boxed{\textbf{(D)}\ \frac{1}{2}}."
        self.assertEqual(last_box(text), r"\textbf{(D)}\ \frac{1}{2}")
        self.assertEqual(fast_reference({"solution": text, "answer": "0.5"}), "last_box_equivalent")
        self.assertIsNone(fast_reference({"solution": text, "answer": "3"}))

    def rows(self):
        return [{"raw_id": f"r{i}", "sample_id": f"s{i}", "duplicate_group_id": f"g{i}", "raw_index": i+100,
                 "problem": f"Find x for instance {i}.", "answer": "2", "solution": r"Result \boxed{2}.",
                 "difficulty": "easy" if i < 20 else "medium", "domain": "algebra" if i % 3 else "geometry"}
                for i in range(60)]

    def test_training_prefixes_are_nested_and_source_order_independent(self):
        rows = self.rows()
        a = training_order(rows, 42)
        b = training_order(list(reversed(rows)), 42)
        self.assertEqual([r["raw_id"] for r in a], [r["raw_id"] for r in b])
        self.assertTrue(set(r["raw_id"] for r in a[:20]) <= set(r["raw_id"] for r in a[:40]))
        self.assertEqual([r["sampling_rank"] for r in a], list(range(60)))

    def test_validation_is_frozen_disjoint_and_keeps_all_source_rows(self):
        source = self.rows()
        validation, _ = validation_selection(source, 12, 42, {100, 101})
        again, _ = validation_selection(list(reversed(source)), 12, 42, {100, 101})
        self.assertEqual([r["raw_id"] for r in validation], [r["raw_id"] for r in again])
        self.assertFalse({100, 101} & {r["raw_index"] for r in validation})
        held = {r["raw_id"] for r in validation}
        pool = training_order([r for r in source if r["raw_id"] not in held], 42)
        check_splits(source, validation, pool, pool[:20], pool[20:])
        with self.assertRaises(AssertionError):
            check_splits(source, validation, pool + [validation[0]], pool[:20], pool[20:])


if __name__ == "__main__":
    unittest.main()
