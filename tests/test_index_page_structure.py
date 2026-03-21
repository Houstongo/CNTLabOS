import unittest
from pathlib import Path

from lxml import html


class IndexPageStructureTests(unittest.TestCase):
    def setUp(self):
        self.index_path = Path(r"D:\CNTDATA\CNTA_ML_Project\index.html")
        self.tree = html.fromstring(self.index_path.read_text(encoding="utf-8"))

    def test_clean_page_is_not_nested_inside_ml_page(self):
        clean_page = self.tree.get_element_by_id("clean-page")
        ml_page = self.tree.get_element_by_id("ml-page")
        self.assertNotIn(ml_page, clean_page.iterancestors())

    def test_rag_page_is_not_nested_inside_ml_page(self):
        rag_page = self.tree.get_element_by_id("rag-page")
        ml_page = self.tree.get_element_by_id("ml-page")
        self.assertNotIn(ml_page, rag_page.iterancestors())

    def test_detail_panel_contains_runtime_fields_used_by_open_details(self):
        self.assertIsNotNone(self.tree.get_element_by_id("d-actual-temp"))
        self.assertIsNotNone(self.tree.get_element_by_id("d-pos-cm"))

    def test_detail_panel_contains_dynamic_delete_actions_mount(self):
        self.assertIsNotNone(self.tree.get_element_by_id("detail-delete-actions"))

    def test_data_page_contains_batch_action_mount(self):
        self.assertIsNotNone(self.tree.get_element_by_id("data-batch-toolbar"))

    def test_data_page_contains_select_all_checkbox(self):
        self.assertIsNotNone(self.tree.get_element_by_id("data-select-all"))


if __name__ == "__main__":
    unittest.main()
