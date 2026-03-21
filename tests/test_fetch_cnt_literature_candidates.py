import unittest

from scripts.fetch_cnt_literature_candidates import (
    classify_source,
    normalize_doi,
    relation_union,
    upsert_rows,
)


class FetchCntLiteratureCandidatesTests(unittest.TestCase):
    def test_normalize_doi_accepts_url(self):
        self.assertEqual(
            normalize_doi("https://doi.org/10.1016/J.CARBON.2020.10.066"),
            "10.1016/j.carbon.2020.10.066",
        )

    def test_relation_union_keeps_defined_order(self):
        left = "process_to_morphology;morphology_to_performance"
        right = "process_to_performance;process_to_morphology"
        self.assertEqual(
            relation_union(left, right),
            "process_to_morphology;morphology_to_performance;process_to_performance",
        )

    def test_classify_source_prefers_crossref_pdf(self):
        item = {
            "URL": "https://publisher.example/article",
            "link": [{"URL": "https://publisher.example/paper.pdf", "content-type": "application/pdf"}],
        }
        source_url, status = classify_source(item, "10.1000/abc", "", do_verify=False)
        self.assertEqual(source_url, "https://publisher.example/paper.pdf")
        self.assertEqual(status, "direct_pdf_candidate")

    def test_upsert_rows_merges_relations_and_prefers_better_status(self):
        rows = [
            {
                "priority": "P1",
                "bucket": "02_形貌-性能",
                "title": "Doc A",
                "year": "2020",
                "journal": "Carbon",
                "doi": "10.1000/a",
                "relation_targets": "morphology_to_performance",
                "notes": "note-a",
                "source_url": "https://publisher.example/a",
                "download_status": "publisher_page_only",
            },
            {
                "priority": "P0",
                "bucket": "03_工艺-性能",
                "title": "Doc A Updated",
                "year": "2021",
                "journal": "Carbon",
                "doi": "https://doi.org/10.1000/a",
                "relation_targets": "process_to_performance",
                "notes": "note-b",
                "source_url": "https://publisher.example/a.pdf",
                "download_status": "direct_pdf_candidate",
            },
        ]
        merged = upsert_rows(rows)
        self.assertEqual(len(merged), 1)
        row = merged[0]
        self.assertEqual(row["priority"], "P0")
        self.assertEqual(row["doi"], "10.1000/a")
        self.assertEqual(row["source_url"], "https://publisher.example/a.pdf")
        self.assertEqual(row["download_status"], "direct_pdf_candidate")
        self.assertEqual(
            row["relation_targets"],
            "morphology_to_performance;process_to_performance",
        )


if __name__ == "__main__":
    unittest.main()
