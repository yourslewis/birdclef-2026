#!/usr/bin/env python3
"""Generate BirdCLEF v238 workflow and model-architecture diagrams.

Outputs PNG, SVG, and/or standalone HTML diagrams without Graphviz or browser deps.
The diagrams are intentionally static/readable, with a small amount of metadata
parsed from kaggle-kernels/v238-file-context-boost/script.py so labels stay useful
as v238 constants change.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "kaggle-kernels" / "v238-file-context-boost" / "script.py"
DEFAULT_OUTDIR = Path(__file__).resolve().parent


PALETTE = {
    "bg": "#0f172a",
    "panel": "#111827",
    "stroke": "#334155",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "amber": "#d97706",
    "purple": "#7c3aed",
    "pink": "#db2777",
    "red": "#dc2626",
    "slate": "#475569",
}


@dataclass(frozen=True)
class Node:
    id: str
    title: str
    lines: tuple[str, ...]
    x: int
    y: int
    w: int
    h: int
    color: str = "blue"


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""


@dataclass(frozen=True)
class Diagram:
    name: str
    title: str
    subtitle: str
    width: int
    height: int
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScriptFacts:
    pca_dim: str = "64"
    ensemble_weight: str = "0.60"
    protossm_d_model: str = "128"
    protossm_d_state: str = "16"
    protossm_layers: str = "2"
    protossm_seeds: str = "42, 137, 2026"
    quantile_mix_alpha: str = "0.50"


def _const(text: str, name: str, default: str) -> str:
    m = re.search(rf"^{re.escape(name)}\s*=\s*([^#\n]+)", text, flags=re.M)
    if not m:
        return default
    return m.group(1).strip().strip("'").strip('"')


def read_facts(script_path: Path) -> ScriptFacts:
    if not script_path.exists():
        return ScriptFacts()
    text = script_path.read_text(encoding="utf-8", errors="replace")
    return ScriptFacts(
        pca_dim=_const(text, "PROBE_PCA_DIM", "64"),
        ensemble_weight=_const(text, "PROTOSSM_ENSEMBLE_WEIGHT", "0.60"),
        protossm_d_model=_const(text, "PROTOSSM_D_MODEL", "128"),
        protossm_d_state=_const(text, "PROTOSSM_D_STATE", "16"),
        protossm_layers=_const(text, "PROTOSSM_N_LAYERS", "2"),
        protossm_seeds=_const(text, "PROTOSSM_SEEDS", "[42, 137, 2026]").strip("[]"),
        quantile_mix_alpha=_const(text, "QUANTILE_MIX_ALPHA", "0.50"),
    )


def build_workflow(facts: ScriptFacts) -> Diagram:
    nodes = (
        Node("input", "Kaggle inputs", ("taxonomy + sample submission", "train labels / train soundscapes", "hidden test soundscapes"), 70, 120, 250, 110, "slate"),
        Node("perch", "Perch inference", ("ONNX fast path when available", "TF SavedModel fallback", "embeddings + base logits"), 390, 120, 270, 110, "blue"),
        Node("meta", "Metadata + label prep", ("parse site/hour from filenames", "map Perch labels to BirdCLEF", "active/proxy/prior-only groups"), 730, 120, 290, 110, "cyan"),
        Node("priors", "Ecological priors", ("site, hour, site×hour tables", "global fallback probabilities", "fuse selected logits"), 210, 310, 270, 110, "green"),
        Node("features", "Probe features", (f"standardize → PCA-{facts.pca_dim}", "class prototypes + cosine sim", "raw/base/prior/family signals"), 570, 310, 300, 110, "purple"),
        Node("mlp", "Per-class MLP probes", ("train one small classifier", "only sufficiently represented classes", "tabular calibration branch"), 130, 505, 290, 110, "pink"),
        Node("ssm", "ProtoSSM branch", (f"d_model={facts.protossm_d_model}, state={facts.protossm_d_state}", f"layers={facts.protossm_layers}, seeds={facts.protossm_seeds}", "temporal/file sequence modeling"), 500, 505, 310, 110, "amber"),
        Node("ensemble", "Ensemble + calibration", (f"simple blend: ProtoSSM weight {facts.ensemble_weight}", "rank-average ensemble", f"quantile mix α={facts.quantile_mix_alpha}"), 875, 505, 315, 110, "red"),
        Node("post", "Post-process", ("per-class temperatures", "Gaussian temporal smoothing", "file-context boost + power calibration"), 330, 700, 330, 110, "green"),
        Node("submit", "submission.csv", ("clip/order to sample_submission", "write Kaggle artifact", "print final run summary"), 760, 700, 270, 110, "blue"),
    )
    edges = (
        Edge("input", "perch"), Edge("perch", "meta"), Edge("meta", "priors"), Edge("perch", "features"),
        Edge("priors", "features"), Edge("features", "mlp"), Edge("perch", "ssm"), Edge("priors", "ssm"),
        Edge("mlp", "ensemble", "probe scores"), Edge("ssm", "ensemble", "ProtoSSM scores"),
        Edge("ensemble", "post"), Edge("post", "submit"),
    )
    return Diagram(
        name="birdclef_v238_workflow",
        title="BirdCLEF v238 File-Context-Boost Workflow",
        subtitle="End-to-end path from Kaggle inputs to submission.csv",
        width=1260,
        height=900,
        nodes=nodes,
        edges=edges,
        notes=("Generated from docs/diagrams/generate_v238_diagrams.py", "Target: kaggle-kernels/v238-file-context-boost/script.py"),
    )


def build_architecture(facts: ScriptFacts) -> Diagram:
    nodes = (
        Node("audio", "60s soundscape audio", ("load OGG", "window/batch per file", "hidden test or dry-run train files"), 60, 110, 260, 105, "slate"),
        Node("perch", "Perch foundation model", ("Bird Vocalization Classifier", "ONNX Runtime preferred", "TF fallback"), 390, 110, 270, 105, "blue"),
        Node("embed", "Perch embeddings", ("semantic acoustic representation", "cached for train when mounted", "input to PCA/prototypes/SSM"), 250, 285, 290, 110, "cyan"),
        Node("base", "Perch/base logits", ("mapped to BirdCLEF labels", "raw score signal", "input to priors + SSM"), 605, 285, 285, 110, "cyan"),
        Node("prior", "Metadata prior branch", ("site/hour/site×hour tables", "class group-specific fusion", "ecological calibration"), 935, 285, 285, 110, "green"),
        Node("pca", "PCA + prototype bank", (f"PCA-{facts.pca_dim} features", "class prototypes", "cosine similarity features"), 120, 470, 280, 110, "purple"),
        Node("family", "Family aggregates", ("taxonomy grouping", "family-level averages", "weak biological prior"), 445, 470, 260, 110, "green"),
        Node("mlp", "Per-class MLP probes", ("features: PCA/raw/prior/base", "prototype + sequence features", "outputs calibrated probe logits"), 765, 470, 300, 110, "pink"),
        Node("ssm", "ProtoSSM temporal model", (f"S4D blocks × {facts.protossm_layers}", f"d_model={facts.protossm_d_model}, d_state={facts.protossm_d_state}", f"seed ensemble: {facts.protossm_seeds}"), 250, 655, 320, 120, "amber"),
        Node("blend", "Weighted + rank ensemble", (f"probe: {float_or_text(1, facts.ensemble_weight)}", f"ProtoSSM: {facts.ensemble_weight}", "simple blend + rank average"), 670, 655, 300, 120, "red"),
        Node("head", "Calibration head", ("temperature scaling", "Gaussian temporal smoothing", "context boost / power calibration"), 405, 825, 330, 110, "green"),
        Node("out", "BirdCLEF probabilities", ("class-wise probabilities", "submission column order", "submission.csv"), 830, 825, 260, 110, "blue"),
    )
    edges = (
        Edge("audio", "perch"), Edge("perch", "embed"), Edge("perch", "base"), Edge("base", "prior"),
        Edge("embed", "pca"), Edge("pca", "mlp"), Edge("base", "mlp"), Edge("prior", "mlp"), Edge("family", "mlp"),
        Edge("embed", "ssm"), Edge("base", "ssm"), Edge("prior", "ssm"), Edge("mlp", "blend", "probe logits"),
        Edge("ssm", "blend", "temporal logits"), Edge("blend", "head"), Edge("head", "out"),
    )
    return Diagram(
        name="birdclef_v238_architecture",
        title="BirdCLEF v238 Model Architecture",
        subtitle="Perch foundation features feeding prior, MLP-probe, and ProtoSSM ensemble branches",
        width=1260,
        height=1020,
        nodes=nodes,
        edges=edges,
        notes=("Generated from docs/diagrams/generate_v238_diagrams.py", "Target: kaggle-kernels/v238-file-context-boost/script.py"),
    )


def float_or_text(one: float, weight_text: str) -> str:
    try:
        return f"{one - float(weight_text):.2f}"
    except ValueError:
        return f"1 - {weight_text}"


def node_map(diagram: Diagram) -> dict[str, Node]:
    return {n.id: n for n in diagram.nodes}


def edge_points(src: Node, dst: Node) -> tuple[tuple[int, int], tuple[int, int]]:
    # Simple center-to-center routing with endpoints clipped to rectangle edges.
    sx, sy = src.x + src.w // 2, src.y + src.h // 2
    dx, dy = dst.x + dst.w // 2, dst.y + dst.h // 2
    if abs(dx - sx) > abs(dy - sy):
        start = (src.x + (src.w if dx >= sx else 0), sy)
        end = (dst.x + (0 if dx >= sx else dst.w), dy)
    else:
        start = (sx, src.y + (src.h if dy >= sy else 0))
        end = (dx, dst.y + (0 if dy >= sy else dst.h))
    return start, end


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) <= max_chars:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    font_path = next((p for p in candidates if Path(p).exists()), None)
    if font_path:
        return (
            ImageFont.truetype(font_path, 34),
            ImageFont.truetype(font_path, 20),
            ImageFont.truetype(font_path, 17),
            ImageFont.truetype(font_path, 14),
        )
    return (ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default())


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str) -> None:
    draw.line([start, end], fill=fill, width=3)
    import math

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 10
    p1 = (end[0] - size * math.cos(angle - 0.45), end[1] - size * math.sin(angle - 0.45))
    p2 = (end[0] - size * math.cos(angle + 0.45), end[1] - size * math.sin(angle + 0.45))
    draw.polygon([end, p1, p2], fill=fill)


def render_png(diagram: Diagram, out: Path) -> None:
    title_font, subtitle_font, body_font, small_font = load_fonts()
    image = Image.new("RGB", (diagram.width, diagram.height), PALETTE["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), diagram.title, fill=PALETTE["text"], font=title_font)
    draw.text((42, 72), diagram.subtitle, fill=PALETTE["muted"], font=subtitle_font)

    nodes = node_map(diagram)
    for edge in diagram.edges:
        src, dst = nodes[edge.src], nodes[edge.dst]
        start, end = edge_points(src, dst)
        draw_arrow(draw, start, end, PALETTE["muted"])
        if edge.label:
            lx, ly = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
            draw.rounded_rectangle((lx - 58, ly - 13, lx + 58, ly + 13), radius=8, fill=PALETTE["bg"], outline=PALETTE["stroke"])
            draw.text((lx - 50, ly - 9), edge.label[:18], fill=PALETTE["muted"], font=small_font)

    for node in diagram.nodes:
        fill = PALETTE[node.color]
        shadow = (node.x + 5, node.y + 5, node.x + node.w + 5, node.y + node.h + 5)
        draw.rounded_rectangle(shadow, radius=16, fill="#020617")
        draw.rounded_rectangle((node.x, node.y, node.x + node.w, node.y + node.h), radius=16, fill=PALETTE["panel"], outline=fill, width=3)
        draw.rounded_rectangle((node.x, node.y, node.x + node.w, node.y + 34), radius=16, fill=fill)
        draw.text((node.x + 14, node.y + 8), node.title, fill="#ffffff", font=body_font)
        y = node.y + 46
        for line in node.lines:
            for wrapped in wrap_text(line, max(18, node.w // 11)):
                draw.text((node.x + 16, y), f"• {wrapped}", fill=PALETTE["text"], font=small_font)
                y += 19

    y = diagram.height - 48
    for note in diagram.notes:
        draw.text((42, y), note, fill=PALETTE["muted"], font=small_font)
        y += 18
    image.save(out)


def svg_text_lines(lines: Iterable[str], x: int, y: int, width: int, css_class: str) -> str:
    spans = []
    dy = 0
    for line in lines:
        for wrapped in wrap_text(line, max(18, width // 11)):
            spans.append(f'<text x="{x}" y="{y + dy}" class="{css_class}">• {html.escape(wrapped)}</text>')
            dy += 20
    return "\n".join(spans)


def render_svg_text(diagram: Diagram) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{diagram.width}" height="{diagram.height}" viewBox="0 0 {diagram.width} {diagram.height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M2,2 L10,6 L2,10 z" fill="#94a3b8" /></marker>',
        "<style>",
        "svg{background:#0f172a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}.title{font-size:34px;font-weight:700;fill:#e5e7eb}.subtitle{font-size:20px;fill:#94a3b8}.node-title{font-size:17px;font-weight:700;fill:#fff}.body{font-size:14px;fill:#e5e7eb}.muted{font-size:13px;fill:#94a3b8}.edge{stroke:#94a3b8;stroke-width:3;fill:none;marker-end:url(#arrow)}",
        "</style>",
        "</defs>",
        f'<text x="40" y="58" class="title">{html.escape(diagram.title)}</text>',
        f'<text x="42" y="88" class="subtitle">{html.escape(diagram.subtitle)}</text>',
    ]
    nodes = node_map(diagram)
    for edge in diagram.edges:
        src, dst = nodes[edge.src], nodes[edge.dst]
        (sx, sy), (dx, dy) = edge_points(src, dst)
        parts.append(f'<line x1="{sx}" y1="{sy}" x2="{dx}" y2="{dy}" class="edge"/>')
        if edge.label:
            lx, ly = (sx + dx) // 2, (sy + dy) // 2
            parts.append(f'<rect x="{lx-62}" y="{ly-14}" width="124" height="28" rx="8" fill="#0f172a" stroke="#334155"/>')
            parts.append(f'<text x="{lx}" y="{ly+5}" class="muted" text-anchor="middle">{html.escape(edge.label)}</text>')
    for node in diagram.nodes:
        color = PALETTE[node.color]
        parts.extend([
            f'<rect x="{node.x+5}" y="{node.y+5}" width="{node.w}" height="{node.h}" rx="16" fill="#020617" opacity="0.8"/>',
            f'<rect x="{node.x}" y="{node.y}" width="{node.w}" height="{node.h}" rx="16" fill="#111827" stroke="{color}" stroke-width="3"/>',
            f'<path d="M {node.x+16} {node.y} H {node.x+node.w-16} Q {node.x+node.w} {node.y} {node.x+node.w} {node.y+16} V {node.y+34} H {node.x} V {node.y+16} Q {node.x} {node.y} {node.x+16} {node.y}" fill="{color}"/>',
            f'<text x="{node.x+14}" y="{node.y+23}" class="node-title">{html.escape(node.title)}</text>',
            svg_text_lines(node.lines, node.x + 16, node.y + 60, node.w, "body"),
        ])
    y = diagram.height - 35
    for note in diagram.notes:
        parts.append(f'<text x="42" y="{y}" class="muted">{html.escape(note)}</text>')
        y += 18
    parts.append("</svg>")
    return "\n".join(parts)


def render_svg(diagram: Diagram, out: Path) -> None:
    out.write_text(render_svg_text(diagram), encoding="utf-8")


def render_html(diagram: Diagram, out: Path) -> None:
    svg = render_svg_text(diagram)
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(diagram.title)}</title>
  <style>
    body {{ margin: 0; background: #020617; color: #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    main {{ padding: 24px; }}
    .frame {{ overflow: auto; border: 1px solid #334155; border-radius: 18px; box-shadow: 0 20px 60px rgba(0,0,0,.35); }}
    svg {{ display: block; max-width: none; }}
  </style>
</head>
<body>
  <main>
    <div class="frame">{svg}</div>
  </main>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BirdCLEF v238 workflow/model diagrams.")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT, help="Path to v238 script.py")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--diagram", choices=["workflow", "architecture", "all"], default="all")
    parser.add_argument("--formats", nargs="+", choices=["png", "svg", "html"], default=["png", "svg", "html"])
    args = parser.parse_args()

    facts = read_facts(args.script)
    selected: list[Diagram] = []
    if args.diagram in {"workflow", "all"}:
        selected.append(build_workflow(facts))
    if args.diagram in {"architecture", "all"}:
        selected.append(build_architecture(facts))

    args.outdir.mkdir(parents=True, exist_ok=True)
    for diagram in selected:
        for fmt in args.formats:
            out = args.outdir / f"{diagram.name}.{fmt}"
            if fmt == "png":
                render_png(diagram, out)
            elif fmt == "svg":
                render_svg(diagram, out)
            elif fmt == "html":
                render_html(diagram, out)
            print(out)


if __name__ == "__main__":
    main()
