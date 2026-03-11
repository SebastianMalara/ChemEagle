from __future__ import annotations

import unittest

from review_renderer import render_reaction_png


class ReviewRendererTests(unittest.TestCase):
    def test_render_reaction_png_returns_bytes_for_valid_reaction(self) -> None:
        payload = {
            "reactants": [{"smiles": "CCO", "label": "A"}],
            "products": [{"smiles": "CC=O", "label": "B"}],
        }
        rendered = render_reaction_png(payload)
        self.assertTrue(rendered.startswith(b"\x89PNG"))

    def test_render_reaction_png_handles_invalid_smiles(self) -> None:
        payload = {
            "reactants": [{"smiles": "None", "label": "A"}],
            "products": [{"smiles": "invalid", "label": "B"}],
        }
        rendered = render_reaction_png(payload)
        self.assertTrue(rendered.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
