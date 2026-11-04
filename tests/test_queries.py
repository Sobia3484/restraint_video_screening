import unittest
from pathlib import Path

from vidscreen.queries import load_queries

ROOT = Path(__file__).resolve().parents[1]


class QueryFileTests(unittest.TestCase):
    def test_25_unique_english_queries(self):
        qs = load_queries(ROOT / "config" / "queries.csv")
        self.assertEqual(len(qs), 25)
        self.assertEqual(len({q.id for q in qs}), 25)
        self.assertEqual(len({q.file_stem for q in qs}), 25)
        self.assertEqual(qs[0].file_stem, "Q001_safe_restraint_techniques_for_infants_and_young_children")
        self.assertTrue(all(q.language == "en" for q in qs))
        self.assertTrue(all(q.text.isascii() for q in qs))

    def test_ids_sequential(self):
        qs = load_queries(ROOT / "config" / "queries.csv")
        self.assertEqual([q.id for q in qs], [f"Q{i:03d}" for i in range(1, 26)])

    def test_every_query_has_a_subtopic(self):
        qs = load_queries(ROOT / "config" / "queries.csv")
        self.assertTrue(all(q.subtopic for q in qs))


if __name__ == "__main__":
    unittest.main()
