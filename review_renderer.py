from __future__ import annotations

import io
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw


def _renderable_smiles(smiles: str) -> Optional[str]:
    text = (smiles or "").strip()
    if not text or text.lower() == "none":
        return None
    if Chem.MolFromSmiles(text) is not None:
        return text
    placeholder_text = text
    for token in ("[R]", "[R1]", "[R2]", "[R3]", "[R4]", "[PG]"):
        placeholder_text = placeholder_text.replace(token, "[*]")
    if Chem.MolFromSmiles(placeholder_text) is not None:
        return placeholder_text
    return None


def _valid_smiles_entries(entries: Iterable[dict]) -> List[Tuple[str, str]]:
    valid: List[Tuple[str, str]] = []
    for entry in entries:
        smiles = str(entry.get("smiles") or "").strip()
        label = str(entry.get("label") or "").strip()
        renderable = _renderable_smiles(smiles)
        if renderable is None:
            continue
        valid.append((renderable, label or smiles))
    return valid


def _render_panel(entries: Sequence[Tuple[str, str]], title: str) -> Image.Image:
    if not entries:
        image = Image.new("RGB", (360, 240), "white")
        draw = ImageDraw.Draw(image)
        draw.text((16, 16), title, fill="black")
        draw.text((16, 120), "No renderable structures", fill="gray")
        return image

    mols = []
    legends = []
    for smiles, label in entries:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(label or smiles)
    if not mols:
        return _render_panel([], title)

    grid = Draw.MolsToGridImage(
        mols,
        molsPerRow=max(1, min(3, len(mols))),
        subImgSize=(260, 220),
        legends=legends,
        returnPNG=False,
    )
    canvas = Image.new("RGB", (grid.width, grid.height + 40), "white")
    canvas.paste(grid, (0, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black")
    return canvas


def render_reaction_png(reaction: dict) -> bytes:
    reactants = _valid_smiles_entries(reaction.get("reactants", []))
    products = _valid_smiles_entries(reaction.get("products", []))
    left = _render_panel(reactants, "Reactants")
    right = _render_panel(products, "Products")

    arrow_width = 120
    height = max(left.height, right.height)
    image = Image.new("RGB", (left.width + right.width + arrow_width, height), "white")
    image.paste(left, (0, 0))
    image.paste(right, (left.width + arrow_width, 0))

    draw = ImageDraw.Draw(image)
    mid_y = height // 2
    start_x = left.width + 20
    end_x = left.width + arrow_width - 20
    draw.line((start_x, mid_y, end_x, mid_y), fill="black", width=4)
    draw.polygon(
        [
            (end_x, mid_y),
            (end_x - 18, mid_y - 10),
            (end_x - 18, mid_y + 10),
        ],
        fill="black",
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
