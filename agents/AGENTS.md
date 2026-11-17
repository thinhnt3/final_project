# Agent instructions for Project 3

## Mục tiêu

Xây dựng hệ thống text-to-image có CLIP frozen, graph reranking và giải thích bằng đường đi/quan hệ của graph. Reference chính là `references/Project03_Multimodal_XAI (2).pdf`.

## Quy tắc bắt buộc

1. Đọc `plan-task/README.md` và task file liên quan trước khi làm việc.
2. Không viết code vào `src/` cho tới khi các task `P0` và `P1` đã được nghiệm thu theo `plan-task/00-roadmap.md`.
3. Mỗi task phải có input, output, tiêu chí nghiệm thu và trạng thái. Không đánh dấu Done nếu chưa có artifact hoặc log kiểm chứng.
4. Không dùng test query/candidate để chọn alpha, hyperparameter, xây cạnh thống kê hoặc huấn luyện projection.
5. Mọi baseline và phương pháp graph dùng cùng split, cùng candidate pool và cùng seed protocol.
6. Thực nghiệm chính chạy tối thiểu 3 seed; báo cáo mean và standard deviation cho Recall@1/5/10 và MRR.
7. Tách rõ `data/raw`, `data/processed`, `artifacts`, `runs` và `reports`; không commit dữ liệu lớn hoặc secret.
8. Ưu tiên thay đổi nhỏ, có thể xóa/thay thế từng module. Không gộp download, preprocessing, train và evaluation vào một script khó tái sử dụng.

## Cách tiếp nhận task

- Chỉ nhận task có dependency đã hoàn thành.
- Trước khi sửa, kiểm tra file đang tồn tại và giữ nguyên thay đổi của agent khác.
- Sau khi làm, cập nhật trạng thái task, lệnh đã chạy, artifact tạo ra và vấn đề còn lại trong task file.
- Nếu phát hiện sai giả định dữ liệu hoặc leakage, dừng nhánh phụ thuộc và ghi blocker vào task file.

## Thứ tự làm việc

`P0 quyết định phạm vi` → `P1 thiết kế contract/split` → các nhánh `P2` song song → `P3` đồng bộ tích hợp → `P4` chạy thực nghiệm → `P5` báo cáo/demo.

## Chưa được phép làm

- Chưa thêm model mới ngoài phạm vi đã chốt.
- Chưa tối ưu theo test.
- Chưa tuyên bố graph cải thiện nếu chưa có CLIP-only, CLIP+Graph và ablation cạnh.
- Chưa tuyên bố explanation faithful nếu chưa perturb đúng cạnh/đường đi được dùng để tính điểm.

## Verification log (2026-11-10)

| Requirement | Evidence | Verification | Status |
|---|---|---|---|
| Một dataset cho Cấp độ 1 | `configs/project_scope.yaml`, Visual Genome metadata | Official `scene_graphs.json` và `image_data.json` đã tải; subset tạo được 40 images/40 queries/1322 relations | VERIFIED |
| Graph scene data load được | `src/data/visual_genome.py` | `scripts/prepare_visual_genome_subset.py` chạy thành công | VERIFIED |
| CLIP + graph rerank contract | `src/retrieval/pipeline.py`, `src/retrieval/reranker.py` | End-to-end smoke chạy với deterministic fallback | PARTIAL: chưa chạy model weights thật |
| Recall@1/5/10 và MRR | `src/evaluation/metrics.py` | Smoke xuất đủ 4 metric | VERIFIED trên fixture/subset fallback |
| Source of truth nằm trong `src/` | `src/models`, `src/retrieval`, `src/evaluation`, `src/data` | `python -m compileall` và smoke pass | VERIFIED |
| 3 seeds, ablation, xAI fidelity | roadmap P4 | Chưa implement | UNKNOWN |

## Latest run evidence (2026-11-10)

- Real CLIP: `runs/clip_visual_genome.json`; 40 JPGs, 10 validation queries, 10 held-out test queries, FAISS index, alpha selected on validation, seeds 13/37/97.
- Ablation: `runs/ablation_visual_genome.json`; relation removal and predicate shuffle were executed. Results do not yet prove that the current graph scorer improves every metric.
- GraphSAGE: `runs/graphsage_visual_genome.json`; trainable encoder and margin training smoke ran on the fixed train split.
- The primary CLIP+Graph result still uses the traceable lexical scorer so explanations correspond exactly to score contributions. Do not describe it as a fully trained GraphSAGE reranker until the two scoring paths are unified.

## Ablation verification update (2026-11-17, main 49bd52d base)

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Pure deterministic graph control | src/models/ablations.py | degree/source/seed tests | VERIFIED |
| Predicate feature mask, direction control | src/models/graphsage.py | mask, reverse-edge, backward tests | VERIFIED |
| Run identity and resume safety | GraphSageConfig + checkpoint.changed_settings | legacy/default/mismatch + YAML snapshot tests | VERIFIED |
| Train/checkpoint/export | scripts/train_graphsage.py, existing fit | all-mode synthetic integration test | VERIFIED |
| Actual VG-COCO metrics | configs/retrieval/ablations/*.yaml | cache meta.json missing; real-data smoke stopped | UNKNOWN |

Full suite: 250 passed. Targeted Ruff clean; mypy unavailable. No final ablation
results. Explanation requires matching model config and transforms raw graph with
the same variant/seed. Read private/ablation-guide.html for the code guide.

Local experiment update (18/11/2026): cache restoration and actual VG-COCO
execution are now VERIFIED. All 15 runs (full + four variants × three seeds)
trained/exported and all five benchmarks completed. Artifact checks confirm
unit finite 10000×512 vectors and identical input hashes. Full fused R@1
48.12±0.23%; objects_only 47.56±0.16%; drop_predicates 48.06±0.28%;
random_edges 47.79±0.18%; undirected 47.84±0.15%. Do not claim full wins on
all metrics or that these differences are statistically significant. Private
report and comparison.csv include the complete results and caveats.

Ablation interpretation audit (18/11/2026): private/analyze_ablation.py recomputed
per-query test ranks from existing vectors, without retraining or tuning. Results
in private/ablation-execution/analysis.json and query-ranks.npz. Full vs
objects_only: net top-1 gains 53/33/40 queries for seeds 13/37/97. Full vs
masked predicates: 14/25/-27. Equal means do not imply equal predictions.
Post-hoc diagnostic only: k=500 for every method, retaining each original
validation-selected alpha. Full R@1/R@10/MRR = 48.1956/86.3156/60.7960%;
masked = 48.0622/86.2800/60.6742%. The official full R@10 weakness is largely
explained by k=10 selection for seeds 37/97; do not treat this diagnostic as a
replacement benchmark or choose new hyperparameters using these test results.
Existing explain.py ranks relations using single-deletion score drops: a fidelity
test of that same top single deletion is circular; joint deletion or separate
rank/margin outcomes and random controls are needed, with limitations stated.

## Commit-readiness verification — 18/11/2026

| Requirement | Implementation | Verification | Status |
|---|---|---|---|
| Core ablations preserve contracts | models/ablations, graphsage | invariants + all 15 real runs | VERIFIED |
| Resume rejects metadata-only or wrong run | RunDirectory.validate_finished | regression test + 15 actual resume skips | VERIFIED |
| Consumers reject incompatible/stale artifacts | models/run_validation + CLI consumers | wrong seed/config/cache and stale hash tests | VERIFIED |
| Joint fidelity with equal random budgets | evaluation/fidelity + evaluate_fidelity CLI | exact toy tests + 200 queries × 3 checkpoints | VERIFIED |
| Paired uncertainty with image grouping | evaluation/ablation_analysis | clustered toy test + 1,500-image bootstrap | VERIFIED |
| Labels/true topology improve retrieval generally | analysis/ablation.json | exploratory conditional CIs contain zero | UNKNOWN |
| Caption-type/annotation noise causes aggregate result | ten cases only | no representative labeled audit | UNKNOWN |

262 tests passed, Ruff entire src/scripts/tests clean, mypy src/scripts clean
(using isolated tooling installed under /tmp; model/runtime dependencies unchanged).
Docs: docs/graphsage-design-decisions.md. No retrieval tuning after observing test.
No commit made; keep private/ ignored. Historical baseline outputs replaced during
restoration should be reviewed separately from code changes when preparing commits.

Formatting/artifact policy update (18/11/2026): new Python modules/tests formatted
fully; existing modules formatted only in changed ranges. Processing steps use
short English comments and substantial calls/signatures use multiline parameters.
`/runs/` and `/private/` now ignored for new artifacts; existing tracked run files
remain tracked and unchanged by this policy. The requirements checklist now
reflects VG-COCO 10K rather than obsolete 40-image pilot code. No result files
were regenerated, deleted, restored or untracked during formatting.
