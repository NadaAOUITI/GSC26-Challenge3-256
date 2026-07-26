"""Generate Excalidraw-style architecture PNG for README."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture.png"
W, H = 1200, 380
STROKE = "#1e1e1e"

BOXES = [
    (40, 140, 190, 70, "#ffd8a8", "ZIP bundles", "train + judge"),
    (270, 140, 170, 70, "#d0bfff", "data_loader", "nested parquet"),
    (480, 140, 170, 70, "#a5d8ff", "features", "job aggregates"),
    (690, 140, 170, 70, "#b2f2bb", "model", "HistGBM"),
    (900, 140, 150, 70, "#b2f2bb", "predict", "merge row_ids"),
    (1080, 140, 120, 70, "#ffd8a8", "Kaggle", "submission.csv"),
    (480, 40, 240, 55, "#fff3bf", "CLI main.py", "inspect | train | eval | predict"),
]


def _font(size: int, bold: bool = False):
    paths = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
    if bold:
        paths = ["C:/Windows/Fonts/seguisb.ttf", *paths]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)
    draw.text((40, 20), "Challenge 03 — FRESCO Failure Prediction", fill=STROKE, font=_font(22, True))
    for x, y, w, h, color, title, subtitle in BOXES:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=color, outline=STROKE, width=2)
        draw.text((x + 10, y + 12), title, fill=STROKE, font=_font(14, True))
        draw.text((x + 10, y + 34), subtitle, fill="#343a40", font=_font(11))
    arrows = [((230, 175), (270, 175)), ((440, 175), (480, 175)), ((650, 175), (690, 175)), ((860, 175), (900, 175)), ((1050, 175), (1080, 175)), ((600, 95), (575, 140))]
    for start, end in arrows:
        draw.line([start, end], fill=STROKE, width=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
