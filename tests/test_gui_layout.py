from __future__ import annotations

import unittest
from pathlib import Path


class GuiLayoutTests(unittest.TestCase):
    def test_single_run_section_removed_from_source(self) -> None:
        source = Path(__file__).resolve().parents[1] / "gui_app.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn('Accordion("Single Run"', text)
        self.assertIn('Retry mode', text)


if __name__ == "__main__":
    unittest.main()
