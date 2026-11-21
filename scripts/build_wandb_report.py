#!/usr/bin/env python3
"""Build the W&B report on the runs uploaded by scripts.upload_report_runs.

    python -m scripts.build_wandb_report                    # create a new report, print its URL
    python -m scripts.build_wandb_report --url <report url> # rewrite that report in place, same URL

Panels select runs by config: role (summary / train) and the section flags s1..s5. Section 7 has no
runs and the two-dataset chart of section 2 is a picture: static figures from reports/ on the main branch. The text
blocks live in docs/wandb-report/*.md and repeat the numbers of README.md; keep both in sync.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import wandb_workspaces.reports.v2 as wr

from src.connectors.wandb_tracker import resolve_entity
from src.core.exceptions import AppError
from src.retrieval.config import RetrievalConfig

log = logging.getLogger("build_wandb_report")

DEFAULT_CONFIG = "configs/retrieval/vg_coco.yaml"
TEXT_DIR = Path(__file__).resolve().parents[1] / "docs" / "wandb-report"
TITLE = "Truy hồi ảnh bằng scene graph trên VG ∩ COCO: GraphSAGE và AdaMF-MAT, báo cáo đồ án môn học"
DESCRIPTION = (
    "CLIP + rerank bằng đồ thị: GraphSAGE (cấp độ 1) và tái hiện AdaMF-MAT (cấp độ 2), ablation, "
    "relation queries có confuser và edge masking, đánh giá giải thích (fidelity, sparsity, validity)."
)
ORDER = [wr.OrderBy(name=wr.Metric("Name"), ascending=True)]
# Figures are static files in the repository, served from its main branch.
FIGURE_URL = "https://raw.githubusercontent.com/thinhnt3/master-hcmus-graphML/main/reports"


def _runset(entity: str, project: str, role: str, section: str, name: str) -> wr.Runset:
    filters = f'Config("role") == \'{role}\' and Config("{section}") == True'
    return wr.Runset(entity=entity, project=project, name=name, order=ORDER, filters=filters)


def _summary_set(entity: str, project: str, section: str, name: str) -> wr.Runset:
    return _runset(entity, project, "summary", section, name)


def _train_set(entity: str, project: str, section: str, name: str) -> wr.Runset:
    return _runset(entity, project, "train", section, name)


def _text(name: str) -> wr.MarkdownBlock:
    """Markdown block from docs/wandb-report/<name>.md."""
    return wr.MarkdownBlock((TEXT_DIR / f"{name}.md").read_text(encoding="utf-8"))


def _figure(path: str, caption: str) -> wr.Image:
    """Image block showing reports/<path>.png."""
    return wr.Image(url=f"{FIGURE_URL}/{path}.png", caption=caption)


def bar(title: str, metric: str, x: int, y: int, w: int = 12, h: int = 8) -> wr.BarPlot:
    """Horizontal bar chart of one summary metric, one bar per run."""
    return wr.BarPlot(
        title=title,
        metrics=[wr.Metric(metric)],
        orientation="h",
        font_size="medium",
        max_runs_to_show=30,
        max_bars_to_show=30,
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def curve(title: str, metric: str, x: int, y: int, w: int = 12, h: int = 8) -> wr.LinePlot:
    """Per-epoch curve of one metric, train runs grouped by configuration (mean ± std over seeds)."""
    return wr.LinePlot(
        title=title,
        x=wr.Metric("epoch"),
        y=[wr.Metric(metric)],
        groupby=wr.Config("label"),
        groupby_aggfunc="mean",
        groupby_rangefunc="stddev",
        layout=wr.Layout(x=x, y=y, w=w, h=h),
    )


def build(entity: str, project: str) -> wr.Report:
    """The report with every section, not yet saved."""
    report = wr.Report(entity=entity, project=project, width="readable", title=TITLE, description=DESCRIPTION)
    report.blocks = [
        wr.TableOfContents(),
        wr.H1("Mở đầu: bài toán, câu hỏi, các run"),
        _text("00-intro"),
        wr.H1("1. Cấp độ 1: CLIP + GraphSAGE"),
        _text("10-graphsage"),
        wr.PanelGrid(
            runsets=[_summary_set(entity, project, "s1", "Cấp độ 1")],
            panels=[
                bar("R@1 test sau rerank (%)", "test/r1", 0, 0),
                bar("Graph-only R@1 test (%)", "graph_only/r1", 12, 0),
            ],
        ),
        wr.PanelGrid(
            runsets=[_train_set(entity, project, "s1", "Cấp độ 1, đường học")],
            panels=[
                curve("Val R@1 theo epoch (trung bình ± std 3 seed)", "val_recall@1", 0, 0),
                curve("Train loss theo epoch", "train_loss", 12, 0),
            ],
        ),
        _text("11-graphsage-notes"),
        wr.H1("2. Ablation đồ thị của GraphSAGE và fidelity"),
        _text("20-graph-ablation"),
        wr.PanelGrid(
            runsets=[_summary_set(entity, project, "s2", "Ablation cấp độ 1")],
            panels=[
                bar("R@1 test sau rerank (%)", "test/r1", 0, 0),
                bar("Graph-only R@1 test (%)", "graph_only/r1", 12, 0),
            ],
        ),
        _text("21-graph-ablation-notes"),
        _text("22-graph-ablation-mkg-w"),
        wr.PanelGrid(
            runsets=[_summary_set(entity, project, "s2_mkg_w", "Ablation trên MKG-W caption subset")],
            panels=[
                bar("MKG-W caption subset: R@1 test sau rerank (%)", "test/r1", 0, 0),
                bar("MKG-W caption subset: graph-only R@1 test (%)", "graph_only/r1", 12, 0),
            ],
        ),
        _figure("ablation/level1-ablation", "Ablation trên hai dataset: R@1 tăng so với CLIP và graph-only R@1."),
        wr.H1("3. Cấp độ 2: AdaMF-MAT so với GraphSAGE"),
        _text("30-adamf-mat"),
        wr.PanelGrid(
            runsets=[_summary_set(entity, project, "s3", "So sánh cấp độ 2")],
            panels=[
                bar("R@1 test sau rerank (%)", "test/r1", 0, 0),
                bar("Graph-only R@1 test (%)", "graph_only/r1", 12, 0),
            ],
        ),
        wr.PanelGrid(
            runsets=[_train_set(entity, project, "s3", "Cấp độ 2, đường học")],
            panels=[
                curve("Val R@1 theo epoch (trung bình ± std 3 seed)", "val_recall@1", 0, 0),
                curve("Train loss theo epoch", "train_loss", 12, 0),
            ],
        ),
        _text("31-adamf-mat-notes"),
        wr.H1("4. Ablation của AdaMF-MAT"),
        wr.PanelGrid(
            runsets=[_summary_set(entity, project, "s4", "Ablation AdaMF-MAT")],
            panels=[
                bar("R@1 test sau rerank (%)", "test/r1", 0, 0, w=24, h=10),
            ],
        ),
        _text("40-adamf-ablation-notes"),
        wr.H1("5. Mô hình có đọc cạnh không: relation queries"),
        _text("50-relation-queries"),
        wr.PanelGrid(
            runsets=[_summary_set(entity, project, "s5", "Relation queries")],
            panels=[
                bar("Confuser acc@1, all (%)", "rel_all", 0, 0),
                bar("Confuser acc@1, all, sau edge masking (%)", "rel_all_masked", 12, 0),
                bar("Confuser acc@1, contrast (%)", "rel_contrast", 0, 8),
                bar("Confuser acc@1, contrast, sau edge masking (%)", "rel_contrast_masked", 12, 8),
            ],
        ),
        _text("51-relation-queries-notes"),
        wr.H1("6. Ablation khác, kết luận, chưa làm"),
        _text("60-other-ablations-conclusions"),
        wr.H1("7. Đánh giá giải thích: fidelity, sparsity, validity"),
        _text("70-explanation-fidelity"),
        _figure(
            "explanation-quality/figures/fidelity",
            "Fidelity: xoá nhóm relation giải thích so với xoá ngẫu nhiên cùng số relation.",
        ),
        _text("71-explanation-sparsity"),
        _figure(
            "explanation-quality/figures/keep-only",
            "Sparsity: giữ vị trí đầu khi chỉ giữ nhóm giải thích so với giữ nhóm ngẫu nhiên.",
        ),
        _text("72-explanation-validity"),
        _figure(
            "explanation-quality/figures/validity-ai",
            "Validity: nhãn AI trên 42 relation của 20 truy vấn, theo hai cách đọc tiêu chí.",
        ),
        _text("73-explanation-gnnexplainer"),
        _figure(
            "explanation-quality/figures/gnn-explainer-control",
            "Đối chứng GNNExplainer so với cách bỏ từng relation và ngẫu nhiên.",
        ),
    ]
    return report


def main() -> int:
    """Parse the command line, then create the report or rewrite an existing one."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--entity", help="W&B entity; default: the logged-in user's default entity")
    parser.add_argument("--url", help="existing report to rewrite in place")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = RetrievalConfig.from_yaml(args.config)
        entity = resolve_entity(args.entity)
    except AppError:
        log.exception("cannot read the config or the W&B entity")
        return 1
    report = build(entity, config.wandb.project)
    if args.url:
        # The web UI adds a /wandb/ path prefix on forge hosts that from_url cannot parse.
        existing = wr.Report.from_url(args.url.replace("/wandb/", "/", 1))
        existing.title, existing.description, existing.blocks = report.title, report.description, report.blocks
        report = existing
    report.save()
    log.info("saved %s", report.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
