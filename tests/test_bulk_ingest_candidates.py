import unittest

from scripts.bulk_ingest_candidates import (
    Candidate,
    build_download_url_candidates,
    build_pdf_filename,
    build_signatures,
    make_http_session,
    normalize_doi,
    normalize_title_key,
    normalize_url,
    should_skip_by_existing,
    status_allow_set,
)


class BulkIngestCandidatesTests(unittest.TestCase):
    def test_normalize_doi_and_url(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1000/ABC"), "10.1000/abc")
        self.assertEqual(
            normalize_url("HTTPS://Example.com/path/Paper.pdf?x=1"),
            "https://example.com/path/Paper.pdf?x=1",
        )

    def test_build_signatures_contains_doi_url_title(self):
        row = Candidate(
            priority="P0",
            bucket="b1",
            title="Aligned CNT Arrays for Conductivity",
            year="2024",
            journal="Carbon",
            doi="10.1000/test-doi",
            relation_targets="morphology_to_performance",
            notes="",
            source_url="https://example.com/paper.pdf",
            download_status="direct_pdf_candidate",
        )
        signatures = build_signatures(row)
        self.assertIn("doi:10.1000/test-doi", signatures)
        self.assertIn("url:https://example.com/paper.pdf", signatures)
        self.assertIn(f"title:{normalize_title_key(row.title)}", signatures)

    def test_build_pdf_filename_ends_with_pdf(self):
        row = Candidate(
            priority="P1",
            bucket="b2",
            title="A Very Long Title for Carbon Nanotube Paper",
            year="2023",
            journal="Nano Lett",
            doi="10.1000/xyz",
            relation_targets="process_to_morphology",
            notes="",
            source_url="https://example.com/a.pdf",
            download_status="direct_pdf_candidate",
        )
        filename = build_pdf_filename(row)
        self.assertTrue(filename.endswith(".pdf"))
        self.assertIn("2023_", filename)

    def test_should_skip_by_existing(self):
        row = Candidate(
            priority="P0",
            bucket="b1",
            title="Known Title",
            year="2024",
            journal="Carbon",
            doi="10.1000/known",
            relation_targets="process_to_performance",
            notes="",
            source_url="https://example.com/known.pdf",
            download_status="direct_pdf_candidate",
        )
        reason = should_skip_by_existing(
            row=row,
            existing_title_keys={normalize_title_key("Known Title")},
            existing_file_keys=set(),
            historical_signatures=set(),
            run_signatures=set(),
        )
        self.assertEqual(reason, "title_exists_in_kb")

    def test_status_allow_set(self):
        self.assertEqual(
            status_allow_set("direct_pdf_candidate,oa_pdf"),
            {"direct_pdf_candidate", "oa_pdf"},
        )
        self.assertTrue(status_allow_set(""))

    def test_make_http_session_disables_env_proxy(self):
        session = make_http_session()
        try:
            self.assertFalse(session.trust_env)
        finally:
            session.close()

    def test_build_download_candidates_with_no_doi_keeps_source_url(self):
        row = Candidate(
            priority="P1",
            bucket="b3",
            title="No DOI Paper",
            year="2020",
            journal="Carbon",
            doi="",
            relation_targets="",
            notes="",
            source_url="HTTPS://example.com/test.pdf?x=1",
            download_status="direct_pdf_candidate",
        )
        session = make_http_session()
        try:
            urls = build_download_url_candidates(row=row, session=session, timeout=5)
        finally:
            session.close()
        self.assertEqual(urls, ["https://example.com/test.pdf?x=1"])


if __name__ == "__main__":
    unittest.main()
