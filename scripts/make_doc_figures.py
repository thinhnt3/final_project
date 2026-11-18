#!/usr/bin/env python3
"""Draw the SVG figures of docs/ from the committed result and graph files.

Run from the repository root:
    python -m scripts.make_doc_figures

Reads runs/benchmark/*.json, runs/graphsage/<run>/seed<n>/history.json, runs/explanations/
and data/processed/vg_coco/graph/*.json; writes docs/images/*.svg. Charts take every number
from those files. Diagrams carry fixed labels; the one illustrative figure (reranking with
assumed scores) says so on the image. Standard library plus PyYAML; no model is loaded.
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.core.config_parsing import read_mapping, require_field
from src.core.exceptions import AppError, DataError
from src.data.config import DataConfig
from src.data.json_io import read_json

log = logging.getLogger("make_doc_figures")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
DEFAULT_OUT = "docs/images"
# The running example of the docs: one test query and its correct image.
EXAMPLE_QUERY_ID = "coco-659101"
EXAMPLE_IMAGE_ID = 2350042
MAIN_RUN_MARKER = "-learned-"
VARIANT_RUN_MARKER = "-clip_image-"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'
LIGHT_BG, DARK_BG = "#fcfcfb", "#1a1a19"
WARNINGS: list[str] = []


# ----------------------------------------------------------------------------- colours
def _rgb(colour: str) -> tuple[int, int, int]:
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def _mix(top: str, base: str, share: float) -> str:
    """`share` of colour `top` laid over colour `base`."""
    pairs = zip(_rgb(top), _rgb(base), strict=True)
    return "#" + "".join(f"{round(a * share + b * (1 - share)):02x}" for a, b in pairs)


# Token -> (light value, dark value). Light values are written as attributes, dark ones as CSS.
T: dict[str, tuple[str, str]] = {
    "bg": (LIGHT_BG, DARK_BG),
    "band": ("#f3f2ee", "#222221"),
    "card": ("#ffffff", "#262624"),
    "ink": ("#0b0b0b", "#f0efec"),
    "ink2": ("#52514e", "#c3c2b7"),
    "muted": ("#898781", "#8f8d86"),
    "grid": ("#e1e0d9", "#2c2c2a"),
    "axis": ("#c3c2b7", "#4a4a46"),
    "border": ("#d6d5ce", "#3d3d39"),
    "blue": ("#2a78d6", "#3987e5"),
    "orange": ("#eb6834", "#d95926"),
    "aqua": ("#1baf7a", "#199e70"),
    "gray": ("#b3b1aa", "#5f5e59"),
    "gray2": ("#6f6d67", "#a9a79f"),
    "white": ("#ffffff", "#ffffff"),
}
for _name in ("blue", "orange", "aqua"):
    T[_name + "T"] = (_mix(T[_name][0], LIGHT_BG, 0.12), _mix(T[_name][1], DARK_BG, 0.26))
T["grayT"] = (_mix("#898781", LIGHT_BG, 0.10), _mix("#898781", DARK_BG, 0.20))


# ----------------------------------------------------------------------------- text metrics
_EM: dict[str, float] = dict.fromkeys("abdeghnopqu", 0.556)
_EM.update({"c": .5, "f": .278, "i": .222, "j": .222, "k": .5, "l": .222, "m": .833, "r": .333, "s": .5,
            "t": .278, "v": .5, "w": .722, "x": .5, "y": .5, "z": .5})
_EM.update(dict.fromkeys("ABEKPSVXY", 0.667))
_EM.update({"C": .722, "D": .722, "F": .611, "G": .778, "H": .722, "I": .278, "J": .5, "L": .556, "M": .833,
            "N": .722, "O": .778, "Q": .778, "R": .722, "T": .611, "U": .722, "W": .944, "Z": .611})
_EM.update(dict.fromkeys("0123456789", 0.556))
_EM.update({" ": .278, ".": .278, ",": .278, ":": .278, ";": .278, "(": .333, ")": .333, "-": .333,
            "—": 1.0, "–": .556, "→": 1.0, "←": 1.0, "·": .278, "≥": .584, "≤": .584, "×": .584, "±": .584,
            "α": .58, "%": .889, "“": .4, "”": .4, '"': .355, "'": .191, "/": .278, "+": .584, "=": .584,
            "_": .556, "<": .584, ">": .584, "✓": .8, "∩": .72, "①": .95, "②": .95, "③": .95, "④": .95,
            "−": .584, "…": 1.0, "↑": .6, "↓": .6, "?": .556, "!": .278, "&": .667, "#": .556, "*": .389,
            "đ": .556, "Đ": .722, "′": .25, "@": 1.0, "|": .26, "[": .3, "]": .3, "≈": .584, "≠": .584})


def text_width(text: str, size: float, weight: int = 400, mono: bool = False) -> float:
    """Rough rendered width of a string, slightly pessimistic for system sans faces."""
    if mono:
        return len(text) * size * 0.61
    total = 0.0
    for char in text:
        if char in _EM:
            total += _EM[char]
            continue
        decomposed = unicodedata.normalize("NFD", char)
        base = decomposed[0]
        width = _EM.get(base, 0.667 if base.isupper() else 0.56)
        if "̛" in decomposed:  # horn: ơ, ư
            width += 0.06
        total += width
    return total * size * (1.07 if weight >= 600 else 1.0) * 1.05


def check_fit(text: str, size: float, room: float, weight: int = 400, mono: bool = False, where: str = "") -> None:
    """Record a warning when a label is likely wider than the room it has."""
    width = text_width(text, size, weight, mono)
    if width > room:
        WARNINGS.append(f"{where}: '{text}' is ~{width:.0f}px, room {room:.0f}px")


def wrap(text: str, size: float, room: float, weight: int = 400) -> list[str]:
    """Greedy word wrap to the given width."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if not current or text_width(candidate, size, weight) <= room:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def vn(value: float, decimals: int = 2) -> str:
    """Vietnamese number format: comma for decimals, dot for thousands."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


# ----------------------------------------------------------------------------- svg builder
class Svg:
    """A figure under construction; colours follow the light/dark tokens of `T`."""

    def __init__(self, width: int, height: int, title: str, desc: str) -> None:
        self.w, self.h = width, height
        self.title, self.desc = title, desc
        self.parts: list[str] = []
        self.markers: set[str] = set()
        self.used: set[str] = set()

    def _paint(self, fill: str | None, stroke: str | None) -> str:
        attrs, classes = [], []
        if fill:
            attrs.append(f'fill="{T[fill][0]}"')
            classes.append(f"f-{fill}")
            self.used.add(fill)
        else:
            attrs.append('fill="none"')
        if stroke:
            attrs.append(f'stroke="{T[stroke][0]}"')
            classes.append(f"s-{stroke}")
            self.used.add(stroke)
        if classes:
            attrs.append(f'class="{" ".join(classes)}"')
        return " ".join(attrs)

    @staticmethod
    def _tip(tip: str | None) -> str:
        return f"<title>{html.escape(tip)}</title>" if tip else ""

    def rect(self, x: float, y: float, w: float, h: float, *, fill: str | None = None, stroke: str | None = None,
             rx: float = 0, sw: float = 1.0, tip: str | None = None) -> None:
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'{self._paint(fill, stroke)} stroke-width="{sw}">{self._tip(tip)}</rect>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str = "muted", sw: float = 1.0,
             extra: str = "") -> None:
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {self._paint(None, stroke)} '
            f'stroke-width="{sw}" stroke-linecap="round" {extra}/>'
        )

    def path(self, d: str, *, fill: str | None = None, stroke: str | None = None, sw: float = 1.0,
             tip: str | None = None) -> None:
        self.parts.append(
            f'<path d="{d}" {self._paint(fill, stroke)} stroke-width="{sw}" stroke-linejoin="round" '
            f'stroke-linecap="round">{self._tip(tip)}</path>'
        )

    def circle(self, cx: float, cy: float, radius: float, *, fill: str | None = None, stroke: str | None = None,
               sw: float = 1.0, tip: str | None = None) -> None:
        self.parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" {self._paint(fill, stroke)} stroke-width="{sw}">'
            f"{self._tip(tip)}</circle>"
        )

    def text(self, x: float, y: float, text: str, *, size: float = 13, fill: str = "ink", weight: int = 400,
             anchor: str = "start", mono: bool = False, italic: bool = False, tabular: bool = False,
             extra: str = "") -> None:
        family = f" font-family='{MONO}'" if mono else ""
        style = ' font-style="italic"' if italic else ""
        numbers = ' style="font-variant-numeric: tabular-nums"' if tabular else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"'
            f"{family}{style}{numbers} {self._paint(fill, None)} {extra}>{html.escape(text)}</text>"
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float, *, tok: str = "muted", sw: float = 1.5) -> None:
        self.markers.add(tok)
        self.line(x1, y1, x2, y2, stroke=tok, sw=sw, extra=f'marker-end="url(#ah-{tok})"')

    def background(self) -> None:
        self.rect(0.5, 0.5, self.w - 1, self.h - 1, fill="bg", stroke="border", rx=12)

    def heading(self, title: str, subtitle: str | None = None) -> None:
        self.text(28, 38, title, size=17, weight=600)
        check_fit(title, 17, self.w - 56, 600, where=self.title)
        if subtitle:
            self.text(28, 61, subtitle, size=13, fill="ink2")
            check_fit(subtitle, 13, self.w - 56, where=self.title)

    def save(self, path: Path) -> None:
        defs = [
            f'<marker id="ah-{tok}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="9" markerHeight="9" '
            f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.6 L9.4,5 L0,9.4 z" '
            f'fill="{T[tok][0]}" class="f-{tok}"/></marker>'
            for tok in sorted(self.markers)
        ]
        dark = [f"  .f-{tok} {{ fill: {T[tok][1]}; }} .s-{tok} {{ stroke: {T[tok][1]}; }}"
                for tok in sorted(self.used | self.markers)]
        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f"viewBox=\"0 0 {self.w} {self.h}\" font-family='{FONT}' role=\"img\" "
            'aria-labelledby="fig-title fig-desc">',
            f'<title id="fig-title">{html.escape(self.title)}</title>',
            f'<desc id="fig-desc">{html.escape(self.desc)}</desc>',
            "<style>\n@media (prefers-color-scheme: dark) {\n" + "\n".join(dark) + "\n}\n</style>",
            "<defs>" + "".join(defs) + "</defs>",
            *self.parts,
            "</svg>",
        ]
        path.write_text("\n".join(out) + "\n", encoding="utf-8")


def box(s: Svg, x: float, y: float, w: float, h: float, title: str | list[str], detail: str | list[str] | None = None,
        *, fill: str = "card", stroke: str = "border", tsize: float = 14, dsize: float = 12, rx: float = 10,
        pad: float = 12) -> None:
    """A rounded box with bold lines and optional secondary lines, centred vertically and horizontally."""
    s.rect(x, y, w, h, fill=fill, stroke=stroke, rx=rx, sw=1.2)
    titles = title if isinstance(title, list) else [title]
    details = detail if isinstance(detail, list) else ([detail] if detail else [])
    title_lh, detail_lh = tsize * 1.3, dsize * 1.38
    top = y + (h - len(titles) * title_lh - len(details) * detail_lh) / 2
    for line in titles:
        s.text(x + w / 2, top + title_lh / 2 + 0.36 * tsize, line, size=tsize, weight=600, anchor="middle")
        check_fit(line, tsize, w - 2 * pad + 8, 600, where=s.title)
        top += title_lh
    for line in details:
        s.text(x + w / 2, top + detail_lh / 2 + 0.36 * dsize, line, size=dsize, fill="ink2", anchor="middle")
        check_fit(line, dsize, w - 2 * pad + 8, where=s.title)
        top += detail_lh


def paragraph(s: Svg, x: float, y: float, text: str, room: float, *, size: float = 12.5, fill: str = "ink2") -> float:
    """Wrapped text block; returns the y below it."""
    for line in wrap(text, size, room):
        s.text(x, y, line, size=size, fill=fill)
        y += size * 1.45
    return y


def bar(s: Svg, x0: float, y: float, width: float, thick: float, *, fill: str, tip: str) -> None:
    """A horizontal bar growing from x0, square at the baseline and rounded at the data end."""
    if width <= 6:
        s.rect(x0, y, max(width, 2.5), thick, fill=fill, tip=tip)
        return
    s.path(
        f"M{x0},{y} h{width - 4:.1f} a4,4 0 0 1 4,4 v{thick - 8} a4,4 0 0 1 -4,4 h{-(width - 4):.1f} z",
        fill=fill, tip=tip,
    )


# ----------------------------------------------------------------------------- diagrams
def fig_pipeline(out: Path) -> None:
    s = Svg(1080, 508, "Toàn cảnh pipeline truy hồi",
            "Chuẩn bị trước: ảnh qua CLIP thành vector ảnh, đồ thị cảnh qua GraphSAGE thành vector đồ thị. "
            "Khi có truy vấn: CLIP văn bản, chọn top-k theo cosine CLIP, tính điểm đồ thị, trộn điểm theo alpha, "
            "trả danh sách ảnh kèm relation giải thích.")
    s.background()
    s.heading("Toàn cảnh: từ câu truy vấn đến danh sách ảnh đã xếp lại",
              "Phần trên làm một lần cho cả 10.000 ảnh rồi lưu ra đĩa. Phần dưới chạy mỗi khi có một câu truy vấn.")
    xs = [41, 211, 381, 551, 721, 891]
    bw, bh, gy, iy = 148, 54, 124, 202
    s.rect(20, 86, s.w - 40, 186, fill="band", rx=12)
    s.text(40, 110, "CHUẨN BỊ TRƯỚC · làm một lần", size=12, fill="ink2", weight=600, extra='letter-spacing="0.06em"')
    box(s, xs[1], gy, bw, bh, "Đồ thị cảnh", "nhãn Visual Genome")
    box(s, xs[2], gy, bw, bh, "GraphSAGE", "đã huấn luyện", fill="blueT", stroke="blue")
    box(s, xs[3], gy, bw, bh, "Vector đồ thị", "1 vector cho mỗi ảnh")
    box(s, xs[0], iy, bw, bh, "10.000 ảnh", "VG ∩ COCO")
    box(s, xs[1], iy, bw, bh, "CLIP ảnh", "đóng băng")
    box(s, xs[2], iy, bw, bh, "Vector CLIP ảnh", "512 số cho mỗi ảnh")
    for row_y, pairs in ((gy, ((1, 2), (2, 3))), (iy, ((0, 1), (1, 2)))):
        for a, b in pairs:
            s.arrow(xs[a] + bw + 2, row_y + bh / 2, xs[b] - 3, row_y + bh / 2)
    note_y = 146.0
    for line in ("Chỉ GraphSAGE được huấn luyện,", "trên 7.000 ảnh train.", "CLIP giữ nguyên, không học thêm.",
                 "Vector lưu ra đĩa, lúc truy vấn", "chỉ việc đọc lại."):
        s.text(742, note_y, line, size=12.5, fill="ink2")
        note_y += 19
    s.rect(724, 132, 4, 88, fill="blue", rx=2)
    s.rect(20, 302, s.w - 40, 134, fill="band", rx=12)
    s.text(40, 326, "KHI CÓ TRUY VẤN · chạy cho từng câu", size=12, fill="ink2", weight=600,
           extra='letter-spacing="0.06em"')
    by, bh2 = 348, 66
    labels = [("Câu truy vấn", "một câu tiếng Anh"), ("CLIP văn bản", "đóng băng"),
              ("① Chọn top-k", "theo cosine CLIP"), ("② Điểm đồ thị", "cos(truy vấn, đồ thị)"),
              ("③ Trộn điểm", "α·CLIP + (1−α)·đồ thị"), ("④ Trả kết quả", "ảnh + relation giải thích")]
    for x, (title, detail) in zip(xs, labels, strict=True):
        accent = title.startswith(("②", "③"))
        box(s, x, by, bw, bh2, title, detail, fill="blueT" if accent else "card", stroke="blue" if accent else "border",
            pad=8)
    for a in range(5):
        s.arrow(xs[a] + bw + 2, by + bh2 / 2, xs[a + 1] - 3, by + bh2 / 2)
    s.arrow(xs[2] + bw / 2, iy + bh + 2, xs[2] + bw / 2, by - 3)
    s.arrow(xs[3] + bw / 2, gy + bh + 2, xs[3] + bw / 2, by - 3)
    s.text(xs[2] + bw / 2 + 8, 290, "so với vector truy vấn", size=11.5, fill="ink2")
    s.text(xs[3] + bw / 2 + 8, 290, "đọc vector của k ứng viên", size=11.5, fill="ink2")
    s.text(41, 466, "Ví dụ thật trên tập test: truy vấn "
           "“Woman is just about to cut birthday cake as friends are around.”", size=12.5, fill="ink2")
    s.text(41, 486, f"→ ảnh đúng ({EXAMPLE_IMAGE_ID}) đứng hạng 1; "
           "relation giải thích đứng đầu: woman → cutting → cake.", size=12.5, fill="ink2")
    s.save(out / "pipeline-tong-quan.svg")


def fig_rerank_toy(out: Path) -> None:
    alpha = 0.8
    items: dict[str, tuple[str, str, float, float, bool]] = {
        "A": ("người đứng cạnh ngựa", "man → next to → horse", 0.31, 0.30, False),
        "B": ("người cưỡi ngựa", "man → riding → horse", 0.30, 0.45, True),
        "C": ("ngựa trên đồng cỏ", "horse → on → grass", 0.28, 0.25, False),
    }
    final = {key: alpha * item[2] + (1 - alpha) * item[3] for key, item in items.items()}
    before = sorted(items, key=lambda key: -items[key][2])
    after = sorted(items, key=lambda key: -final[key])
    s = Svg(1000, 452, "Ví dụ minh hoạ xếp hạng lại",
            "Ví dụ với số giả định: ba ảnh ứng viên cho truy vấn a man riding a horse. Theo CLIP, ảnh A đứng đầu. "
            "Sau khi cộng điểm đồ thị với alpha bằng 0,8, ảnh B (người cưỡi ngựa, ảnh đúng) lên hạng 1.")
    s.background()
    s.heading("Xếp hạng lại hoạt động thế nào (ví dụ minh hoạ, số giả định)",
              "Truy vấn: “a man riding a horse”. Với α = 0,8: điểm cuối = 0,8 × điểm CLIP + 0,2 × điểm đồ thị.")
    col_l, col_m, col_r = 28, 344, 676
    cw, ch, gap, top = 296, 82, 14, 116
    s.text(col_l, 102, "Thứ tự theo CLIP", size=14, weight=600)
    s.text(col_m, 102, "Tính điểm cuối", size=14, weight=600)
    s.text(col_r, 102, "Sau khi xếp lại", size=14, weight=600)

    def card(x: float, y: float, rank: int, key: str, score_label: str) -> None:
        name, graph, _, _, correct = items[key]
        s.rect(x, y, cw, ch, fill="blueT" if correct else "card", stroke="blue" if correct else "border", rx=10,
               sw=1.6 if correct else 1.2)
        s.text(x + 22, y + 50, str(rank), size=26, weight=700, anchor="middle")
        s.text(x + 46, y + 28, f"Ảnh {key} · {name}", size=14, weight=600)
        check_fit(f"Ảnh {key} · {name}", 14, cw - 56, 600, where=s.title)
        s.text(x + 46, y + 49, graph, size=12.5, fill="ink2")
        s.text(x + 46, y + 69, score_label, size=12.5, tabular=True)
        if correct:
            s.text(x + cw - 12, y + 69, "ảnh đúng ✓", size=12, weight=600, anchor="end")

    for i, key in enumerate(before):
        card(col_l, top + i * (ch + gap), i + 1, key, f"điểm CLIP {vn(items[key][2])}")
    for i, key in enumerate(after):
        card(col_r, top + i * (ch + gap), i + 1, key, f"điểm cuối {vn(final[key], 3)}")
    my = top + 8.0
    for key in sorted(items):
        _, _, clip, graph, correct = items[key]
        s.rect(col_m, my, 304, ch - 16, fill="card", stroke="blue" if correct else "border", rx=8,
               sw=1.6 if correct else 1.2)
        s.text(col_m + 14, my + 26, f"Ảnh {key}", size=13, weight=600)
        s.text(col_m + 290, my + 26, f"đồ thị {vn(graph)}", size=12.5, fill="ink2", anchor="end", tabular=True)
        formula = f"0,8 × {vn(clip)} + 0,2 × {vn(graph)} = {vn(final[key], 3)}"
        s.text(col_m + 14, my + 50, formula, size=13, tabular=True)
        check_fit(formula, 13, 280, where=s.title)
        my += ch + gap
    s.text(28, 416, "Ảnh B có quan hệ “man riding horse” khớp truy vấn nên điểm đồ thị cao nhất (0,45) "
           "và vượt lên ảnh A.", size=12.5, fill="ink2")
    s.text(28, 436, "Trong ví dụ này CLIP thấy người và ngựa ở cả A lẫn B nên điểm CLIP của hai ảnh gần nhau.",
           size=12.5, fill="ink2")
    s.save(out / "xep-hang-lai-minh-hoa.svg")


def fig_data_steps(out: Path) -> None:
    s = Svg(1080, 392, "Sáu bước dựng dữ liệu",
            "Tải dữ liệu gốc, ghép Visual Genome với COCO, chuẩn hoá nhãn theo synset, chia tập, xây đồ thị, "
            "nhúng CLIP. Mỗi bước ghi file cho bước sau đọc.")
    s.background()
    s.heading("Dữ liệu được dựng qua 6 bước; mỗi bước ghi file cho bước sau đọc",
              "Từ vựng đồ thị, IDF và mọi ngưỡng chỉ tính trên ảnh train. Ảnh test chỉ lấy từ phần test của bản chia "
              "Karpathy.")
    stages = [
        ("Tải dữ liệu gốc", ["Visual Genome:", "ảnh + đồ thị cảnh", "COCO: 5 caption/ảnh", "(bản chia Karpathy)",
                             "sha256 → MANIFEST"], "data/raw/"),
        ("Ghép VG ↔ COCO", ["108.077 ảnh VG", "51.498 có mã COCO", "50.918 sau khi bỏ", "mã COCO bị trùng"],
         "data/interim/"),
        ("Chuẩn hoá nhãn", ["đối tượng → synset", "bikes → bicycle.n.01", "94,1% có synset", "vị từ: viết thường"],
         "data/interim/"),
        ("Chia tập", ["49.381 ảnh hợp lệ", "chọn 10.000 ảnh:", "train 7.000", "val 1.500, test 1.500"],
         "data/splits/"),
        ("Xây đồ thị", ["từ vựng chỉ từ train", "1.778 khái niệm", "493 vị từ", "127.812 bộ ba"], "…/vg_coco/graph/"),
        ("Nhúng CLIP", ["10.000 ảnh", "50.000 caption", "2.271 chữ của đồ thị", "vector dài 1"], "…/embeddings/"),
    ]
    bw, gap, x0, y0, bh = 152, 21, 34, 126, 190
    for i, (title, lines, folder) in enumerate(stages):
        x = x0 + i * (bw + gap)
        cx = x + bw / 2
        s.circle(cx, 100, 13, fill="blue")
        s.text(cx, 105, str(i + 1), size=13, weight=700, anchor="middle", fill="white")
        if i < len(stages) - 1:
            s.arrow(cx + 17, 100, cx + bw + gap - 17, 100)
        s.rect(x, y0, bw, bh, fill="card", stroke="border", rx=10, sw=1.2)
        s.text(x + 14, y0 + 30, title, size=13.5, weight=600)
        check_fit(title, 13.5, bw - 22, 600, where=s.title)
        y = y0 + 60.0
        for line in lines:
            s.text(x + 14, y, line, size=12.5, fill="ink2")
            check_fit(line, 12.5, bw - 22, where=s.title)
            y += 22
        s.text(cx, y0 + bh + 24, folder, size=11.5, fill="ink2", anchor="middle", mono=True)
    s.text(34, s.h - 22, "Bước 1–5 do script dựng dữ liệu chạy (script này không nằm trong repo); "
           "kết quả JSON của chúng có trong repo. Bước 6: python -m scripts.embed_vg_coco.", size=12, fill="ink2")
    s.save(out / "du-lieu-6-buoc.svg")


def fig_funnel(out: Path, split_sizes: dict[str, int]) -> None:
    s = Svg(960, 470, "Phễu dữ liệu và cách chia tập",
            "Số ảnh còn lại sau từng bước lọc: 108.077, 51.498, 50.918, 49.381, 10.000. "
            "10.000 ảnh chia thành train 7.000, validation 1.500, test 1.500.")
    s.background()
    s.heading("Từ 108.077 ảnh Visual Genome còn 10.000 ảnh dùng cho thí nghiệm",
              "Số ảnh còn lại sau từng bước lọc (thanh xanh: phần được dùng), và cách chia 10.000 ảnh đó.")
    rows = [("Ảnh trong Visual Genome", 108077), ("Có mã ảnh COCO", 51498), ("Bỏ ảnh dùng chung một mã COCO", 50918),
            ("Còn ít nhất 1 quan hệ", 49381), ("Chọn vào thí nghiệm", sum(split_sizes.values()))]
    x0, x1, top, step, thick = 300, 860, 92, 38, 20
    scale = (x1 - x0) / 110000
    for i, (label, count) in enumerate(rows):
        y = top + i * step
        s.text(x0 - 14, y + 15, label, size=13, anchor="end")
        width = count * scale
        bar(s, x0, y, width, thick, fill="blue" if i == len(rows) - 1 else "gray", tip=f"{label}: {vn(count, 0)} ảnh")
        s.text(x0 + width + 8, y + 15, vn(count, 0), size=13, tabular=True)
    s.line(x0, top - 6, x0, top + (len(rows) - 1) * step + thick + 6, stroke="axis")
    seg_y = 318
    s.text(28, seg_y - 14, "Cách chia 10.000 ảnh (số trong thanh: số ảnh; mỗi ảnh có 5 caption = 5 câu truy vấn)",
           size=13.5, weight=600)
    total = sum(split_sizes.values())
    colours = {"train": "blue", "val": "orange", "test": "aqua"}
    roles = {"train": ("train", "học GraphSAGE, tính từ vựng đồ thị và IDF"),
             "val": ("validation", "chọn α, k và vòng huấn luyện tốt nhất"),
             "test": ("test", "chỉ chấm điểm một lần, sau khi đã chốt mọi tham số")}
    x = 28.0
    for name in ("train", "val", "test"):
        count = split_sizes[name]
        width = (s.w - 56) * count / total
        s.rect(x, seg_y, width - 2, 28, fill=colours[name], rx=4,
               tip=f"{roles[name][0]}: {vn(count, 0)} ảnh, {vn(5 * count, 0)} truy vấn")
        if text_width(vn(count, 0), 13, 600) + 16 < width:
            s.text(x + 10, seg_y + 19, vn(count, 0), size=13, weight=600, fill="white")
        x += width
    legend_y = seg_y + 56.0
    for name in ("train", "val", "test"):
        title, role = roles[name]
        s.rect(28, legend_y - 11, 12, 12, fill=colours[name], rx=3)
        s.text(48, legend_y, title, size=12.5, weight=600)
        s.text(140, legend_y, f"{role} · {vn(5 * split_sizes[name], 0)} truy vấn", size=12.5, fill="ink2")
        legend_y += 22
    s.save(out / "du-lieu-pheu.svg")


def fig_subgraph(
    out: Path, graph: dict[str, Any], concept_text: dict[int, str], predicate_text: dict[int, str]
) -> None:
    words = {concept_text[c] for c in graph["concept_ids"]}
    triples = {(concept_text[a], predicate_text[p], concept_text[b]) for a, p, b in graph["triplets"]}
    shown = {"shirt": 96, "woman": 230, "cake": 364, "candle": 498, "knife": 632, "girl": 776, "chair": 904}
    relations = {("woman", "wearing", "shirt"): 163, ("woman", "cutting", "cake"): 297, ("candle", "on", "cake"): 431,
                 ("girl", "on", "chair"): 840}
    if not set(shown) <= words or not set(relations) <= triples:
        raise DataError(f"image {EXAMPLE_IMAGE_ID} no longer has the objects and relations drawn in the figure")
    s = Svg(1000, 560, f"Đồ thị con của ảnh {EXAMPLE_IMAGE_ID}",
            "Đỉnh ảnh nối tới các đỉnh đối tượng shirt, woman, cake, candle, knife, girl, chair. Mỗi relation là một "
            "đỉnh phụ: woman → wearing → shirt, woman → cutting → cake, candle → on → cake, girl → on → chair.")
    s.background()
    s.heading(f"Đồ thị con của một ảnh thật (ảnh test {EXAMPLE_IMAGE_ID})",
              "Mỗi relation được lưu thành một đỉnh phụ mang nhãn, vì lớp GraphSAGE không đọc được nhãn trên cạnh.")
    oy, ry, ix, iy = 250, 360, 498, 116
    s.rect(ix - 82, iy - 24, 164, 48, fill="blueT", stroke="blue", rx=10, sw=1.6)
    s.text(ix, iy + 5, f"ảnh {EXAMPLE_IMAGE_ID}", size=14, weight=600, anchor="middle")
    for x in shown.values():
        s.arrow(ix + (x - ix) * 0.12, iy + 25, x, oy - 21, tok="gray", sw=1.2)
    s.text(575, 184, "chứa", size=12, fill="ink2", italic=True)
    for (subject, predicate, target), rel_x in relations.items():
        w = text_width(predicate, 13, 600) + 24
        s.rect(rel_x - w / 2, ry - 15, w, 30, fill="aquaT", stroke="aqua", rx=7, sw=1.4)
        s.text(rel_x, ry + 5, predicate, size=13, weight=600, anchor="middle")
        side = -1 if shown[subject] < rel_x else 1
        s.arrow(shown[subject] + side * -6, oy + 19, rel_x + side * w * 0.22, ry - 17, tok="ink2", sw=1.4)
        s.arrow(rel_x - side * w * 0.22, ry - 17, shown[target] - side * -6, oy + 21, tok="ink2", sw=1.4)
    for word, x in shown.items():
        w = text_width(word, 14, 600) + 30
        s.rect(x - w / 2, oy - 18, w, 36, fill="orangeT", stroke="orange", rx=18, sw=1.4)
        s.text(x, oy + 5, word, size=14, weight=600, anchor="middle")
    s.text(640, 88, "Một caption của ảnh này, dùng làm truy vấn:", size=12, fill="ink2")
    s.text(640, 108, "“Woman is just about to cut birthday cake", size=12.5, italic=True)
    s.text(640, 127, "as friends are around.”", size=12.5, italic=True)
    legend = [
        ("blueT", "blue", 10, "Đỉnh ảnh", "Vector ban đầu: một vector học được, dùng chung cho mọi ảnh (mô hình "
                                          "chính), hoặc vector CLIP của chính ảnh (biến thể)."),
        ("orangeT", "orange", 12, "Đỉnh đối tượng", "Vector ban đầu: vector CLIP văn bản của chữ trên đỉnh, "
                                                    "ví dụ “woman”. Một khái niệm là một đỉnh."),
        ("aquaT", "aqua", 7, "Đỉnh phụ (relation)", "Vector ban đầu: vector CLIP văn bản của vị từ, ví dụ "
                                                     "“cutting”. Cạnh đi chủ thể → đỉnh phụ → đối tượng."),
    ]
    lx = 28
    for fill, stroke, rx, name, text in legend:
        s.rect(lx, 420, 34, 22, fill=fill, stroke=stroke, rx=rx, sw=1.4)
        s.text(lx + 44, 436, name, size=13.5, weight=600)
        paragraph(s, lx, 464, text, 290, size=12)
        lx += 320
    s.text(28, s.h - 16, f"Ảnh thật có {len(graph['concept_ids'])} đối tượng và {len(graph['triplets'])} "
           f"relation; hình vẽ {len(shown)} đối tượng và {len(relations)} relation. "
           "Hai người phụ nữ trong ảnh gộp thành một đỉnh “woman”.", size=12, fill="ink2")
    s.save(out / f"do-thi-con-anh-{EXAMPLE_IMAGE_ID}.svg")


def _layer_node(s: Svg, center: tuple[float, float], label: str, kind: str, reached: bool) -> None:
    w = text_width(label, 13.5, 600) + (30 if kind == "img" else 26)
    rx = {"img": 9, "obj": 16, "rel": 7}[kind]
    fill = {"img": "blueT", "obj": "orangeT", "rel": "aquaT"}[kind] if reached else "card"
    stroke = {"img": "blue", "obj": "orange", "rel": "aqua"}[kind] if reached else "gray"
    s.rect(center[0] - w / 2, center[1] - 16, w, 32, fill=fill, stroke=stroke, rx=rx, sw=1.4)
    s.text(center[0], center[1] + 5, label, size=13.5, weight=600, anchor="middle", fill="ink" if reached else "muted")


def _layer_edge(s: Svg, a: tuple[float, float], b: tuple[float, float], used: bool, trim_a: float = 17,
                trim_b: float = 19) -> None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    s.arrow(a[0] + ux * trim_a, a[1] + uy * trim_a, b[0] - ux * trim_b, b[1] - uy * trim_b,
            tok="blue" if used else "gray", sw=2 if used else 1.3)


def fig_layers(out: Path) -> None:
    s = Svg(1000, 518, "Ba tầng GraphSAGE",
            "Sau tầng 1 đỉnh ảnh biết có woman và cake. Sau tầng 2 biết vai của mỗi đối tượng trong relation cutting. "
            "Sau tầng 3 biết woman và cake là hai đầu của cùng một relation cutting.")
    s.background()
    s.heading("Ba tầng GraphSAGE: sau mỗi tầng, đỉnh ảnh biết thêm một vòng hàng xóm",
              "Đồ thị rút gọn: ảnh → woman, ảnh → cake, woman → (cutting) → cake. Đỉnh tô màu: thông tin đã tới được "
              "đỉnh ảnh.")
    panels = [("Sau tầng 1", "Ảnh biết trong ảnh có woman và cake."),
              ("Sau tầng 2", "Ảnh biết vai: woman là chủ thể của một “cutting”, cake là đối tượng của một “cutting”."),
              ("Sau tầng 3", "Ảnh biết hai đầu thuộc cùng một relation: woman → cutting → cake.")]
    pw, ph, py = 304, 300, 82
    for p, (header, caption) in enumerate(panels):
        px = 24 + p * (pw + 20)
        s.rect(px, py, pw, ph, fill="band", rx=12)
        s.text(px + 16, py + 26, header, size=14, weight=600)
        img, woman = (px + pw / 2, py + 64.0), (px + 78.0, py + 142.0)
        cake, rel = (px + pw - 78.0, py + 142.0), (px + pw / 2, py + 214.0)
        _layer_edge(s, img, woman, True)
        _layer_edge(s, img, cake, True)
        if p == 0:
            _layer_edge(s, woman, rel, False)
            _layer_edge(s, rel, cake, False)
            _layer_node(s, rel, "cutting", "rel", False)
        elif p == 1:
            left, right = (px + 112.0, py + 230.0), (px + pw - 112.0, py + 230.0)
            _layer_edge(s, woman, left, True, trim_b=17)
            _layer_edge(s, right, cake, True, trim_a=17)
            _layer_node(s, left, "cutting", "rel", True)
            _layer_node(s, right, "cutting", "rel", True)
            s.text(px + pw / 2, py + 270, "chưa biết hai “cutting” này là một", size=11.5, fill="ink2", anchor="middle",
                   italic=True)
        else:
            _layer_edge(s, woman, rel, True)
            _layer_edge(s, rel, cake, True)
            _layer_node(s, rel, "cutting", "rel", True)
        _layer_node(s, img, "ảnh", "img", True)
        _layer_node(s, woman, "woman", "obj", True)
        _layer_node(s, cake, "cake", "obj", True)
        cy = py + ph + 22.0
        for line in wrap(caption, 12.5, pw - 8):
            s.text(px + 4, cy, line, size=12.5)
            cy += 18
    fy = 452
    s.line(28, fy - 18, s.w - 28, fy - 18, stroke="grid")
    s.text(28, fy, "Mỗi tầng = một SAGEConv theo chiều cạnh + một SAGEConv ngược chiều cạnh, cộng lại. "
           "Mỗi SAGEConv: h mới = W·h + W′·trung bình(h của hàng xóm).", size=12.5)
    s.text(28, fy + 21, "Có hướng nên “man riding horse” khác “horse riding man”. ReLU giữa các tầng, "
           "không có ReLU sau tầng 3 (vector CLIP có cả số âm).", size=12.5, fill="ink2")
    s.text(28, fy + 42, "Sau tầng 3: vector của đỉnh ảnh → tầng chiếu tuyến tính → chuẩn hoá độ dài 1 "
           "= vector đồ thị (512 số, cùng không gian với CLIP).", size=12.5, fill="ink2")
    s.save(out / "graphsage-3-tang.svg")


def fig_training(out: Path) -> None:
    s = Svg(1000, 466, "Huấn luyện bằng InfoNCE trong batch",
            "Đồ thị của ảnh i đi qua GraphSAGE thành vector g_i; caption của ảnh i đi qua CLIP văn bản đóng băng "
            "thành t_i. Bảng điểm g_i với t_j: ô đường chéo là cặp đúng, được kéo lên; các ô khác là cặp sai, bị đẩy "
            "xuống.")
    s.background()
    s.heading("Huấn luyện: kéo vector đồ thị của ảnh về gần vector caption của chính ảnh đó",
              "InfoNCE trong batch. CLIP đóng băng; chỉ GraphSAGE và tầng chiếu được cập nhật (3,41 triệu tham số).")
    n, cell, mx, my = 4, 50, 520, 182
    rows_y = [my + cell * i + cell / 2 for i in range(n)]
    for i, y in enumerate(rows_y):
        box(s, 34, y - 18, 140, 36, f"đồ thị ảnh {i + 1}", None, tsize=13, rx=8)
        s.arrow(176, y, 214, y)
    box(s, 218, my + 8, 160, cell * n - 16, ["GraphSAGE", "+ tầng chiếu"], ["được học"], fill="blueT", stroke="blue")
    for i, y in enumerate(rows_y):
        s.arrow(380, y, 466, y)
        s.text(488, y + 5, f"g{i + 1}", size=14, weight=600, anchor="middle")
    box(s, mx, 98, cell * n, 42, "CLIP văn bản", "đóng băng", tsize=13, dsize=11.5, rx=8)
    s.text(mx + cell * n / 2, 90, "caption của ảnh 1, 2, 3, 4", size=12, fill="ink2", anchor="middle")
    for j in range(n):
        x = mx + cell * j + cell / 2
        s.arrow(x, 142, x, 156)
        s.text(x, 174, f"t{j + 1}", size=14, weight=600, anchor="middle")
    for i in range(n):
        for j in range(n):
            x, y = mx + cell * j, my + cell * i
            if i == j:
                s.rect(x + 1, y + 1, cell - 2, cell - 2, fill="blue", rx=4, tip=f"cặp đúng g{i + 1}·t{j + 1}")
                s.text(x + cell / 2, y + cell / 2 + 6, "↑", size=17, weight=700, anchor="middle", fill="white")
            else:
                s.rect(x + 1, y + 1, cell - 2, cell - 2, fill="grayT", stroke="border", rx=4,
                       tip=f"cặp sai g{i + 1}·t{j + 1}")
                s.text(x + cell / 2, y + cell / 2 + 5, "↓", size=14, anchor="middle", fill="muted")
    tx, y = 752, 194.0
    s.rect(tx, y - 13, 14, 14, fill="blue", rx=3)
    y = paragraph(s, tx + 22, y, "Ô đậm: ảnh i với caption của chính nó. Điểm được kéo lên.", 210, fill="ink")
    y += 12
    s.rect(tx, y - 13, 14, 14, fill="grayT", stroke="border", rx=3)
    y = paragraph(s, tx + 22, y, "Ô nhạt: ảnh với caption của ảnh khác trong cùng batch. Điểm bị đẩy xuống.", 210,
                  fill="ink")
    y += 12
    paragraph(s, tx, y, "Mất mát tính theo cả hai chiều: theo hàng (ảnh → caption) và theo cột (caption → ảnh).", 232)
    s.text(34, 414, "Một batch: 256 ảnh train, mỗi ảnh một lần; mỗi vòng chọn ngẫu nhiên 1 trong 5 caption. "
           "Nhiệt độ 0,07, AdamW, tốc độ học 0,001.", size=12.5, fill="ink2")
    s.text(34, 435, "Sau mỗi vòng đo R@1 của riêng điểm đồ thị trên validation, giữ vòng tốt nhất (best.pt), "
           "dừng khi 5 vòng liền không tốt hơn. Khoảng 2 phút mỗi seed trên CPU.", size=12.5, fill="ink2")
    s.save(out / "huan-luyen-infonce.svg")


def fig_explain(out: Path, explanation: dict[str, Any], objects_gap: float) -> None:
    query = next((q for q in explanation["queries"] if q["query_id"] == EXAMPLE_QUERY_ID), None)
    if query is None or not query["results"][0]["is_relevant"]:
        raise DataError(f"explanation file has no correct top result for {EXAMPLE_QUERY_ID}")
    top = query["results"][0]
    relations = top["relations"]
    summary = "; bỏ ".join(
        f"{r['subject']} {r['predicate']} {r['object']} giảm {vn(r['score_drop'], 4)}" for r in relations
    )
    s = Svg(1000, 400, "Giải thích bằng cách bỏ từng relation", f"Ảnh đúng {top['image_id']}: bỏ {summary}.")
    s.background()
    s.heading("Giải thích: bỏ từng relation rồi xem điểm đồ thị giảm bao nhiêu",
              f"Truy vấn “{query['text']}” · ảnh đúng {top['image_id']} · seed {explanation['seed']}")
    steps = [f"Điểm đồ thị với đủ relation: {vn(top['graph_score'], 4)}", "Bỏ một relation (giữ hai đối tượng của nó)",
             "Cho đồ thị mới qua GraphSAGE, tính lại điểm", "Mức giảm = điểm đầy đủ − điểm mới",
             "Lặp lại với từng relation, xếp theo mức giảm"]
    y = 96.0
    for number, text in enumerate(steps, start=1):
        s.circle(44, y + 14, 12, fill="blue")
        s.text(44, y + 19, str(number), size=12.5, weight=700, anchor="middle", fill="white")
        s.text(66, y + 19, text, size=13)
        y += 42
    s.line(484, 92, 484, 300, stroke="grid")
    x0, x1, max_drop = 700, 930, 0.008
    s.text(512, 104, "Ba relation làm điểm giảm nhiều nhất", size=13.5, weight=600)
    for i, relation in enumerate(relations):
        y = 128 + i * 52.0
        label = f"{relation['subject']} → {relation['predicate']} → {relation['object']}"
        s.text(512, y + 15, label, size=13)
        width = (x1 - x0) * relation["score_drop"] / max_drop
        bar(s, x0, y, width, 20, fill="blue", tip=f"{label}: giảm {vn(relation['score_drop'], 4)}")
        s.text(x0 + width + 8, y + 15, vn(relation["score_drop"], 4), size=13, tabular=True)
    s.line(x0, 120, x0, 128 + 2 * 52 + 28, stroke="axis")
    s.text(512, 128 + 3 * 52 + 4, "Các relation còn lại của ảnh làm điểm giảm ít hơn.", size=12, fill="ink2")
    note = ("Relation đứng đầu, woman → cutting → cake, đúng là hành động mà truy vấn mô tả. Mức giảm rất nhỏ so "
            f"với điểm {vn(top['graph_score'], 4)}: phần lớn điểm đồ thị đến từ các đối tượng. Ablation cho kết quả "
            f"cùng chiều: bỏ hết relation, R@1 chỉ giảm {vn(objects_gap)} điểm.")
    paragraph(s, 28, 336, note, s.w - 56)
    s.save(out / "giai-thich-bo-tung-relation.svg")


def fig_commands(out: Path) -> None:
    rows: list[tuple[str, str, list[tuple[str, bool]]]] = [
        ("python -m scripts.embed_vg_coco", "Vector CLIP của ảnh, caption, chữ trên đồ thị",
         [("data/processed/vg_coco/embeddings/<model>/", False)]),
        ("python -m scripts.run_benchmark --method clip", "Baseline CLIP (và graph_lexical: so khớp từ)",
         [("runs/benchmark/clip.json", True), ("graph_lexical.json", True)]),
        ("python -m scripts.train_graphsage", "3 seed; ablation: thêm --config configs/retrieval/ablations/…",
         [("…/seed<n>/config.yaml, history.json, result.json", True), ("best.pt, last.pt, graph_vectors.*", False)]),
        ("python -m scripts.run_benchmark --method graphsage", "Chọn α, k trên validation, chấm test",
         [("runs/benchmark/graphsage-<run>.json", True)]),
        ("python -m scripts.explain_retrieval --sample 5", "Kết quả kèm relation giải thích",
         [("runs/explanations/<run>-seed<n>.json", True)]),
        ("python -m scripts.analyze_ablation", "So sánh ghép cặp các biến thể, khoảng 95% bootstrap",
         [("runs/analysis/ablation.json", False)]),
        ("python -m scripts.evaluate_fidelity", "Xoá nhóm giải thích so với nhóm ngẫu nhiên",
         [("runs/fidelity/full-joint-test.json", False)]),
        ("python -m scripts.evaluate_explanation_quality", "Sparsity, keep-only; xuất mẫu chấm validity",
         [("runs/explanation-quality/*.json", False)]),
    ]
    s = Svg(1000, 64 + 56 + len(rows) * 84 + 40, "Lệnh chạy và file sinh ra",
            "Tám lệnh theo thứ tự: embed_vg_coco, run_benchmark clip, train_graphsage, run_benchmark graphsage, "
            "explain_retrieval, analyze_ablation, evaluate_fidelity, evaluate_explanation_quality, cùng file mỗi "
            "lệnh ghi ra.")
    s.background()
    s.heading("Lệnh chạy và file mỗi lệnh sinh ra",
              "Chạy từ thư mục gốc, từ trên xuống. Thẻ xanh: loại file đã có bản commit trong repo; thẻ xám: chỉ có "
              "trên máy đã chạy.")
    s.text(28, 98, "Đầu vào có sẵn trong repo: data/processed/vg_coco/*.json, graph/*.json, data/splits/. "
           "Ảnh (1,4 GB) phải tải riêng vào data/processed/vg_coco/images/.", size=12.5, fill="ink2")
    top, step, cx, cw = 120, 84, 28, 470
    for i, (command, what, outputs) in enumerate(rows):
        y = top + i * step
        s.rect(cx, y, cw, 58, fill="card", stroke="border", rx=8, sw=1.2)
        s.text(cx + 14, y + 25, command, size=12.5, mono=True)
        check_fit(command, 12.5, cw - 24, mono=True, where=s.title)
        s.text(cx + 14, y + 45, what, size=12, fill="ink2")
        check_fit(what, 12, cw - 24, where=s.title)
        s.arrow(cx + cw + 4, y + 29, cx + cw + 46, y + 29)
        tag_y = y + 8.0
        for path, in_repo in outputs:
            width = text_width(path, 12, mono=True) + 20
            s.rect(cx + cw + 52, tag_y, width, 22, fill="blueT" if in_repo else "grayT",
                   stroke="blue" if in_repo else "gray", rx=6, sw=1.1)
            s.text(cx + cw + 62, tag_y + 15, path, size=12, mono=True)
            if cx + cw + 52 + width > s.w - 16:
                WARNINGS.append(f"commands: '{path}' overflows")
            tag_y += 26
        if i < len(rows) - 1:
            s.arrow(cx + 40, y + 60, cx + 40, y + step - 2, tok="gray", sw=1.3)
    s.text(28, s.h - 18, ".gitignore bỏ qua /runs/ và /private/: output mới chỉ nằm trên máy, file đã commit "
           "từ trước vẫn được theo dõi. Config mặc định báo cáo lên W&B; config ablation đã tắt.", size=12,
           fill="ink2")
    s.save(out / "lenh-va-file.svg")


# ----------------------------------------------------------------------------- charts
def _metric(report: dict[str, Any], kind: str, split: str) -> tuple[float, float | None]:
    """R@1 in percent and its standard deviation (None for single-run methods)."""
    value = report[kind][split]["recall@1"]
    if isinstance(value, dict):
        return value["mean"] * 100, value["std"] * 100
    return value * 100, None


def fig_results(out: Path, res: dict[str, dict[str, Any]]) -> None:
    clip, _ = _metric(res["clip"], "clip", "test")
    lexical, _ = _metric(res["lexical"], "reranked", "test")
    main_mean, main_std = _metric(res["main"], "reranked", "test")
    var_mean, var_std = _metric(res["variant"], "reranked", "test")
    rows: list[tuple[str, float, float | None, str]] = [
        ("CLIP ViT-B/32 (chỉ CLIP)", clip, None, "gray2"),
        ("CLIP + so khớp từ với đồ thị", lexical, None, "gray"),
        ("CLIP + GraphSAGE (mô hình chính)", main_mean, main_std, "blue"),
        ("CLIP + GraphSAGE, đỉnh ảnh mang vector CLIP", var_mean, var_std, "orange"),
    ]
    s = Svg(920, 352, "R@1 trên tập test",
            "; ".join(f"{label}: {vn(m)}" + (f" ± {vn(sd)}" if sd else "") for label, m, sd, _ in rows) + " (%).")
    s.background()
    s.heading("R@1 trên tập test: tỉ lệ truy vấn có ảnh đúng ở hạng 1",
              "1.500 ảnh, 7.500 truy vấn. Dòng GraphSAGE: trung bình 3 seed, vạch ngang = ± 1 độ lệch chuẩn. "
              "Đường dọc: mốc CLIP. Đơn vị: %.")
    x0, x1, lo, hi, top, step, label_x = 420, 880, 42, 51, 118, 48, 392

    def to_x(value: float) -> float:
        return x0 + (value - lo) * (x1 - x0) / (hi - lo)

    bottom = top + step * (len(rows) - 1) + 24
    for tick in (42, 44, 46, 48, 50):
        s.line(to_x(tick), top - 18, to_x(tick), bottom, stroke="grid")
        s.text(to_x(tick), bottom + 18, vn(tick, 0), size=12, fill="muted", anchor="middle", tabular=True)
    s.line(to_x(clip), 104, to_x(clip), bottom, stroke="gray2", sw=1.2)
    for i, (label, mean, std, tok) in enumerate(rows):
        y = top + i * step
        strong = 600 if tok == "blue" else 400
        s.text(label_x, y + 5, label, size=13.5, anchor="end", weight=strong)
        s.line(to_x(lo), y, to_x(hi), y, stroke="grid")
        if std:
            s.line(to_x(mean - std), y, to_x(mean + std), y, stroke=tok, sw=2)
            for edge in (mean - std, mean + std):
                s.line(to_x(edge), y - 6, to_x(edge), y + 6, stroke=tok, sw=2)
        value = vn(mean) + (f" ± {vn(std)}" if std else "")
        s.circle(to_x(mean), y, 7, fill=tok, stroke="bg", sw=2, tip=f"{label}: {value}")
        s.text(to_x(mean) + 14, y - 10, value, size=13, tabular=True, weight=strong)
    gain = main_mean - clip
    queries = res["clip"]["splits"]["test"]["queries"]
    s.text(28, s.h - 22, f"CLIP + GraphSAGE hơn CLIP {vn(gain)} điểm R@1: thêm khoảng "
           f"{round(gain / 100 * queries)} trên {vn(queries, 0)} truy vấn có ảnh đúng đứng đầu.", size=12.5,
           fill="ink2")
    s.save(out / "ket-qua-r1-test.svg")


def fig_graph_only(out: Path, res: dict[str, dict[str, Any]]) -> None:
    lexical, _ = _metric(res["lexical"], "graph_only", "test")
    main_mean, main_std = _metric(res["main"], "graph_only", "test")
    var_mean, var_std = _metric(res["variant"], "graph_only", "test")
    clip, _ = _metric(res["clip"], "clip", "test")
    random_r1 = 100 / res["clip"]["splits"]["test"]["gallery_images"]
    rows: list[tuple[str, float, float | None, str]] = [
        ("Chọn ngẫu nhiên", random_r1, None, "gray"),
        ("So khớp từ với đồ thị", lexical, None, "gray"),
        ("GraphSAGE (mô hình chính)", main_mean, main_std, "blue"),
        ("GraphSAGE, đỉnh ảnh mang vector CLIP", var_mean, var_std, "orange"),
        ("CLIP thuần (để so sánh)", clip, None, "gray2"),
    ]
    s = Svg(920, 360, "R@1 khi chỉ dùng điểm đồ thị",
            "; ".join(f"{label}: {vn(m)}" + (f" ± {vn(sd)}" if sd else "") for label, m, sd, _ in rows) + " (%).")
    s.background()
    s.heading("Khi chỉ dùng điểm đồ thị, không có CLIP: R@1 trên tập test",
              "Cùng 1.500 ảnh và 7.500 truy vấn. Đơn vị: %. Dòng cuối là CLIP thuần, để so sánh.")
    x0, x1, top, step, thick = 330, 840, 92, 44, 20
    scale = (x1 - x0) / 50
    axis_bottom = top + step * (len(rows) - 1) + thick + 6
    for tick in (0, 10, 20, 30, 40, 50):
        s.line(x0 + tick * scale, top - 8, x0 + tick * scale, axis_bottom, stroke="grid")
        s.text(x0 + tick * scale, axis_bottom + 18, vn(tick, 0), size=12, fill="muted", anchor="middle", tabular=True)
    for i, (label, mean, std, tok) in enumerate(rows):
        y = top + i * step
        strong = 600 if tok == "blue" else 400
        s.text(x0 - 14, y + 15, label, size=13.5, anchor="end", weight=strong)
        bar(s, x0, y, mean * scale, thick, fill=tok, tip=f"{label}: {vn(mean)}")
        end = x0 + max(mean * scale, 2.5)
        if std:
            a, b = x0 + (mean - std) * scale, x0 + (mean + std) * scale
            s.line(a, y + thick / 2, b, y + thick / 2, stroke="ink", sw=1.4)
            for edge in (a, b):
                s.line(edge, y + 5, edge, y + thick - 5, stroke="ink", sw=1.4)
            end = b
        s.text(end + 9, y + 15, vn(mean) + (f" ± {vn(std)}" if std else ""), size=13, tabular=True, weight=strong)
    s.line(x0, top - 8, x0, axis_bottom, stroke="axis")
    s.text(28, s.h - 20, f"GraphSAGE gấp {vn(main_mean / lexical, 1)} lần so khớp từ và khoảng "
           f"{round(main_mean / random_r1)} lần mức ngẫu nhiên, dù đỉnh ảnh không mang thông tin gì về điểm ảnh.",
           size=12.5, fill="ink2")
    s.save(out / "ket-qua-chi-do-thi.svg")


ABLATION_LABELS = {"full": "Đầy đủ", "objects_only": "Chỉ đối tượng", "drop_predicates": "Bỏ nhãn vị từ",
                   "random_edges": "Xáo trộn cạnh", "undirected": "Bỏ hướng"}


def _panel_domain(values: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Axis range and tick step covering every mean ± std with a little air."""
    low = min(m - sd for m, sd in values) - 0.2
    high = max(m + sd for m, sd in values) + 0.2
    tick = 0.5 if high - low <= 2.5 else 1.0
    return math.floor(low / tick) * tick, math.ceil(high / tick) * tick, tick


def fig_ablation(out: Path, reports: dict[str, dict[str, Any]]) -> None:
    names = [name for name in ABLATION_LABELS if name in reports]
    series: dict[int, list[tuple[float, float]]] = {1: [], 2: []}
    for name in names:
        for column, kind in ((1, "reranked"), (2, "graph_only")):
            mean, std = _metric(reports[name], kind, "test")
            series[column].append((mean, std or 0.0))
    s = Svg(1000, 404, "Kết quả ablation",
            "; ".join(f"{ABLATION_LABELS[n]}: sau khi trộn {vn(f[0])} ± {vn(f[1])}, chỉ đồ thị {vn(g[0])} ± "
                      f"{vn(g[1])}" for n, f, g in zip(names, series[1], series[2], strict=True)) + " (R@1 test, %).")
    s.background()
    s.heading("Ablation: R@1 trên tập test khi bỏ hoặc phá từng phần của đồ thị",
              "Mỗi biến thể được huấn luyện lại với 3 seed và chọn α, k riêng trên validation. Chấm: trung bình; vạch "
              "ngang: ± 1 độ lệch chuẩn. Đơn vị: %.")
    top, step, label_x = 140, 46, 214
    panels = [("Sau khi trộn với CLIP", 1, 262, 600), ("Chỉ điểm đồ thị, không có CLIP", 2, 650, 972)]
    for title, column, x0, x1 in panels:
        values = series[column]
        lo, hi, tick = _panel_domain(values)

        def to_x(value: float, lo: float = lo, hi: float = hi, x0: float = x0, x1: float = x1) -> float:
            return x0 + (value - lo) * (x1 - x0) / (hi - lo)

        s.text(x0, top - 38, title, size=13.5, weight=600)
        bottom = top + step * (len(names) - 1) + 22
        value = lo
        while value <= hi + 1e-9:
            s.line(to_x(value), top - 20, to_x(value), bottom, stroke="grid")
            s.text(to_x(value), bottom + 18, vn(value, 1), size=12, fill="muted", anchor="middle", tabular=True)
            value += tick
        full_mean = values[0][0]
        s.line(to_x(full_mean), top - 20, to_x(full_mean), bottom, stroke="blue", sw=1)
        for i, (mean, std) in enumerate(values):
            y = top + i * step
            tok = "blue" if i == 0 else "gray2"
            if std:
                s.line(to_x(mean - std), y, to_x(mean + std), y, stroke=tok, sw=2)
                for edge in (mean - std, mean + std):
                    s.line(to_x(edge), y - 5, to_x(edge), y + 5, stroke=tok, sw=2)
            label = f"{vn(mean)} ± {vn(std)}"
            s.circle(to_x(mean), y, 6.5, fill=tok, stroke="bg", sw=2, tip=f"{ABLATION_LABELS[names[i]]}: {label}")
            s.text(to_x(mean), y - 11, label, size=12, anchor="middle", tabular=True, weight=600 if i == 0 else 400)
    for i, name in enumerate(names):
        y = top + i * step
        s.text(label_x, y + 5, ABLATION_LABELS[name], size=13.5, anchor="end", weight=600 if i == 0 else 400)
    s.text(28, s.h - 22, "Đường dọc xanh: mức của bản đầy đủ. Chỉ chênh lệch với bản “chỉ đối tượng” có khoảng 95% "
           "không chứa 0 (xem tài liệu 4).", size=12.5, fill="ink2")
    s.save(out / "ket-qua-ablation.svg")



def fig_curves(out: Path, histories: dict[str, dict[str, list[dict[str, Any]]]], clip_val: float,
               lexical_val: float) -> None:
    s = Svg(920, 452, "Đường cong huấn luyện",
            "R@1 trên validation của riêng điểm đồ thị theo từng vòng, 3 seed cho mỗi biến thể. Mô hình chính dừng "
            "quanh 27%, biến thể mang vector CLIP dừng quanh 42%.")
    s.background()
    s.heading("Huấn luyện: R@1 của riêng điểm đồ thị trên validation, theo từng vòng",
              "Mỗi đường là một seed (13, 37, 97). Chấm tròn: vòng tốt nhất, được giữ lại làm mô hình. Đơn vị: %.")
    x0, x1, y0, y1 = 72, 880, 352, 92
    longest = max(len(history) for runs in histories.values() for history in runs.values())
    last_epoch = max(36, longest + 1)

    def to_x(epoch: float) -> float:
        return x0 + (epoch - 1) * (x1 - x0) / (last_epoch - 1)

    def to_y(value: float) -> float:
        return y0 - value * (y0 - y1) / 50

    for tick in range(0, 51, 10):
        s.line(x0, to_y(tick), x1, to_y(tick), stroke="grid" if tick else "axis")
        s.text(x0 - 10, to_y(tick) + 4, vn(tick, 0), size=12, fill="muted", anchor="end", tabular=True)
    for epoch in (1, *range(5, last_epoch + 1, 5)):
        s.text(to_x(epoch), y0 + 20, str(epoch), size=12, fill="muted", anchor="middle", tabular=True)
    s.text((x0 + x1) / 2, y0 + 40, "vòng huấn luyện (epoch)", size=12, fill="ink2", anchor="middle")
    for value, label in ((clip_val, f"CLIP thuần trên validation: {vn(clip_val)}"),
                         (lexical_val, f"so khớp từ, chỉ đồ thị: {vn(lexical_val)}")):
        s.line(x0, to_y(value), x1, to_y(value), stroke="gray2", sw=1.2)
        s.text(x1, to_y(value) - 7, label, size=11.5, fill="ink2", anchor="end")
    colours = {"main": "blue", "variant": "orange"}
    for kind, runs in histories.items():
        for seed, history in runs.items():
            points = " L".join(f"{to_x(r['epoch']):.1f},{to_y(r['val_recall@1'] * 100):.1f}" for r in history)
            s.path("M" + points, stroke=colours[kind], sw=2, tip=f"{kind}, seed {seed}")
        for seed, history in runs.items():
            # The best epoch is the last one that set a new best (`is_best` marks each new best).
            best = max((r for r in history if r["is_best"]), key=lambda r: r["epoch"])
            s.circle(to_x(best["epoch"]), to_y(best["val_recall@1"] * 100), 5, fill=colours[kind], stroke="bg", sw=2,
                     tip=f"seed {seed}: vòng tốt nhất {best['epoch']}, R@1 {vn(best['val_recall@1'] * 100)}")
    legend_y = s.h - 22
    first = "GraphSAGE, đỉnh ảnh học được (mô hình chính)"
    s.line(72, legend_y - 4, 100, legend_y - 4, stroke="blue", sw=2.5)
    s.text(108, legend_y, first, size=12.5)
    second_x = 108 + text_width(first, 12.5) + 30
    s.line(second_x, legend_y - 4, second_x + 28, legend_y - 4, stroke="orange", sw=2.5)
    s.text(second_x + 36, legend_y, "GraphSAGE, đỉnh ảnh mang vector CLIP (biến thể)", size=12.5)
    # Label the plateau of the main model with the mean best validation R@1 over its seeds.
    main_runs = list(histories["main"].values())
    plateau = sum(max(r["val_recall@1"] for r in history) for history in main_runs) / len(main_runs) * 100
    end = main_runs[0][-1]
    s.text(to_x(end["epoch"]) + 6, to_y(end["val_recall@1"] * 100) + 22, f"≈ {round(plateau)}%", size=12.5,
           weight=600)
    s.save(out / "duong-cong-huan-luyen.svg")


def fig_alpha(out: Path, lexical: dict[str, Any]) -> None:
    k = lexical["selected_on_validation"]["top_k"]
    points = sorted((p for p in lexical["validation_sweep"] if p["top_k"] == k), key=lambda p: p["alpha"])
    best = max(points, key=lambda p: (p["recall@1"], p["mrr"], p["alpha"]))
    first, last = points[0], points[-1]
    s = Svg(920, 446, "Quét alpha trên validation",
            f"R@1 trên validation theo alpha, phương pháp so khớp từ, k = {k}. Đỉnh tại alpha = {vn(best['alpha'])} "
            f"với R@1 {vn(best['recall@1'] * 100)}; alpha = 1 là CLIP thuần với {vn(last['recall@1'] * 100)}.")
    s.background()
    s.heading(f"Chọn α trên validation: R@1 theo α (phương pháp so khớp từ, k = {k})",
              "α = 1 là CLIP thuần; α = 0 là chỉ dùng điểm đồ thị bên trong k ứng viên. Đơn vị: %.")
    x0, x1, y0, y1 = 72, 880, 340, 100

    def to_x(alpha: float) -> float:
        return x0 + alpha * (x1 - x0)

    def to_y(value: float) -> float:
        return y0 - value * (y0 - y1) / 50

    for tick in range(0, 51, 10):
        s.line(x0, to_y(tick), x1, to_y(tick), stroke="grid" if tick else "axis")
        s.text(x0 - 10, to_y(tick) + 4, vn(tick, 0), size=12, fill="muted", anchor="end", tabular=True)
    for alpha in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        s.text(to_x(alpha), y0 + 20, vn(alpha, 1), size=12, fill="muted", anchor="middle", tabular=True)
    s.text((x0 + x1) / 2, y0 + 40, "α (trọng số của điểm CLIP)", size=12, fill="ink2", anchor="middle")
    s.path("M" + " L".join(f"{to_x(p['alpha']):.1f},{to_y(p['recall@1'] * 100):.1f}" for p in points), stroke="blue",
           sw=2)
    s.circle(to_x(best["alpha"]), to_y(best["recall@1"] * 100), 6, fill="blue", stroke="bg", sw=2)
    for point in (first, last):
        s.circle(to_x(point["alpha"]), to_y(point["recall@1"] * 100), 5, fill="gray2", stroke="bg", sw=2)
    s.text(to_x(best["alpha"]) - 12, to_y(best["recall@1"] * 100) - 14,
           f"chọn α = {vn(best['alpha'])}: {vn(best['recall@1'] * 100)}", size=13, weight=600, anchor="end")
    s.text(to_x(1.0) - 4, to_y(last["recall@1"] * 100) + 26, f"α = 1 (CLIP thuần): {vn(last['recall@1'] * 100)}",
           size=12.5, anchor="end")
    s.text(to_x(0.0) + 10, to_y(first["recall@1"] * 100) + 22,
           f"α = 0 (chỉ điểm đồ thị): {vn(first['recall@1'] * 100)}", size=12.5)
    s.text(28, s.h - 40, "Đỉnh hẹp: lệch α đi 0,05 là R@1 giảm khoảng 1–2 điểm, nên α được quét với bước 0,01.",
           size=12.5, fill="ink2")
    s.text(28, s.h - 20, "Lượt chạy GraphSAGE chỉ lưu α đã chọn cho từng seed, không lưu cả đường quét, "
           "nên hình dùng phương pháp so khớp từ làm ví dụ.", size=12.5, fill="ink2")
    s.save(out / "quet-alpha-validation.svg")


# ----------------------------------------------------------------------------- main
def _graphsage_reports(benchmark_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The main run (learned image node) and the variant (CLIP image node), found by run name."""
    reports = [read_json(path) for path in sorted(benchmark_dir.glob("graphsage-*.json"))]

    def pick(marker: str) -> dict[str, Any]:
        # Ablation runs carry a `-<variant>-a<seed>` suffix; only the unablated run of each kind is wanted.
        found = [r for r in reports if marker in r["run_name"] and not re.search(r"-a\d+$", r["run_name"])]
        if len(found) != 1:
            raise DataError(f"expected one GraphSAGE benchmark with '{marker}' in {benchmark_dir}, found {len(found)}")
        return found[0]

    return pick(MAIN_RUN_MARKER), pick(VARIANT_RUN_MARKER)


def run(config_path: str, out: Path) -> None:
    """Load the result files named by the configs and draw every figure."""
    raw = read_mapping(config_path)
    runs_dir = Path(require_field(raw, "runs_dir", str))
    processed = DataConfig.from_yaml(require_field(raw, "data_config", str)).paths.processed_dir
    benchmark_dir = runs_dir / "benchmark"
    main_report, variant_report = _graphsage_reports(benchmark_dir)
    ablations = {"full": main_report}
    for path in sorted(benchmark_dir.glob(f"graphsage-{main_report['run_name']}-*-a*.json")):
        report = read_json(path)
        ablations[report["run_name"][len(main_report["run_name"]) + 1:].rsplit("-a", 1)[0]] = report
    objects_gap = (
        _metric(main_report, "reranked", "test")[0] - _metric(ablations["objects_only"], "reranked", "test")[0]
        if "objects_only" in ablations else 0.0
    )
    res = {
        "clip": read_json(benchmark_dir / "clip.json"),
        "lexical": read_json(benchmark_dir / "graph_lexical.json"),
        "main": main_report,
        "variant": variant_report,
    }
    histories = {
        kind: {seed: read_json(runs_dir / "graphsage" / report["run_name"] / f"seed{seed}" / "history.json")
               for seed in report["seeds"]}
        for kind, report in (("main", main_report), ("variant", variant_report))
    }
    first_seed = next(iter(main_report["seeds"]))
    explanation = read_json(runs_dir / "explanations" / f"{main_report['run_name']}-seed{first_seed}.json")
    graphs = read_json(processed / "graph" / "image_graphs.json")
    concept_text = {c["concept_id"]: c["text"] for c in read_json(processed / "graph" / "concepts.json")}
    predicate_text = {p["predicate_id"]: p["text"] for p in read_json(processed / "graph" / "predicates.json")}
    example = next((g for g in graphs if g["vg_image_id"] == EXAMPLE_IMAGE_ID), None)
    if example is None:
        raise DataError(f"image {EXAMPLE_IMAGE_ID} is not in {processed / 'graph' / 'image_graphs.json'}")
    split_sizes = {name: sum(1 for g in graphs if g["split"] == name) for name in ("train", "val", "test")}

    out.mkdir(parents=True, exist_ok=True)
    drawers: list[Callable[[], None]] = [
        lambda: fig_pipeline(out),
        lambda: fig_rerank_toy(out),
        lambda: fig_data_steps(out),
        lambda: fig_funnel(out, split_sizes),
        lambda: fig_subgraph(out, example, concept_text, predicate_text),
        lambda: fig_layers(out),
        lambda: fig_training(out),
        lambda: fig_explain(out, explanation, objects_gap),
        lambda: fig_commands(out),
        lambda: fig_results(out, res),
        lambda: fig_graph_only(out, res),
        lambda: fig_curves(out, histories, res["clip"]["clip"]["val"]["recall@1"] * 100,
                           res["lexical"]["graph_only"]["val"]["recall@1"] * 100),
        lambda: fig_alpha(out, res["lexical"]),
        lambda: fig_ablation(out, ablations),
    ]
    for draw in drawers:
        draw()
    for warning in WARNINGS:
        log.warning("label may not fit: %s", warning)
    log.info("wrote %d figures to %s", len(drawers), out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help="directory for the SVG files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(args.config, args.out)
    except AppError:
        log.exception("drawing the figures failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
