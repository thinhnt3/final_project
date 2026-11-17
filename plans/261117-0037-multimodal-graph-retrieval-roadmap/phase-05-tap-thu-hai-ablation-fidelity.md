---
title: "Phase 5: Tập công khai thứ hai, ablation, fidelity"
status: in-progress
---

# Phase 5: Tập công khai thứ hai, ablation, fidelity

## Current verified scope — 18/11/2026

Core ablation on VG-COCO and the frozen joint-deletion diagnostic are implemented
and executed; Phase 5 is still in progress (second dataset and hub ablation remain).
Current commands and evidence: `docs/graphsage-design-decisions.md`.
Earlier entries below are historical logs, not the current completion checklist.

## Overview

Trả lời "phần cải thiện đến từ thành phần nào": ablation trên VG ∩ COCO, lặp lại trên FB15k-237-IMG, và đo fidelity của lời giải thích.

## Mục tiêu

- Chứng minh phần tăng đến từ **quan hệ** chứ không chỉ từ danh sách đối tượng (CLIP vốn nhận ra đối tượng khá tốt).
- Chứng minh cấu trúc đồ thị có nghĩa, không chỉ là "thêm đặc trưng".
- Chứng minh cạnh được dùng để giải thích thực sự ảnh hưởng điểm.

## Input

- Dữ liệu, split, đồ thị VG ∩ COCO (Phase 2); bộ đo, α, K chọn trên val (Phase 3); phương pháp chính (Phase 4).
- FB15k-237-IMG (tải ở phase này).

## Việc phải làm trước khi chạy ablation

- [ ] Tìm nguyên nhân kết quả ngược hiện tại (bỏ hết cạnh MRR 0,495 > đầy đủ 0,213). Nghi ngờ (chưa kiểm): truy vấn cũ sinh từ quan hệ + `graph_score` ghép từ khoá. Kiểm lại trên dữ liệu caption COCO.
- [ ] Sửa `remove_relationships(mode="shuffle")` trong `src/evaluation/ablations.py`: hiện chỉ đổi tên vị từ thành `random_relation`, không phải cạnh ngẫu nhiên. Tách thành hai chế độ rõ tên: `drop_predicates` và `random_edges`.

## Ablation (tất cả cùng backbone, α và K chọn trên val)

| # | Biến thể | Trả lời câu hỏi |
|---|---|---|
| A1 | Đầy đủ (đối tượng + quan hệ) | Mốc |
| A2 | Chỉ đối tượng (bỏ mọi cạnh khái niệm–khái niệm) | Quan hệ có giúp không? **Quan trọng nhất** |
| A3 | Bỏ vị từ (giữ cạnh, mất nhãn quan hệ) | Nhãn vị từ có giúp không? |
| A4 | Cạnh ngẫu nhiên cùng số lượng | Cấu trúc thật hay chỉ là thêm đặc trưng? Mặc định giữ bậc từng đỉnh (hoán đổi cặp cạnh); có thể đổi khi bắt đầu phase |
| A5 | Không GNN (chỉ `graph_score` ghép từ khoá) | GNN có giá trị không? |
| A6 | Quét α (0 → 1) và K (10, 20, 40, 100) | Độ nhạy siêu tham số (vẽ trên val, báo test tại điểm chọn) |
| A7 | Có/không xử lý hub | Quyết định ở 2.6 có đúng không? |

Ablation có ConceptNet (VG sạch so với VG + cạnh ConceptNet) thuộc Phase 7.

## Fidelity

- Xoá các cạnh nằm trên đường giải thích so với xoá cùng số cạnh ngẫu nhiên, đo độ giảm điểm của ảnh đúng.
- Đồ thị do người gán (oracle) so với đồ thị do mô hình SGG dự đoán: tuỳ chọn, chỉ làm nếu còn thời gian (cần mô hình SGG huấn luyện trên VG150).

## Tập thứ hai: FB15k-237-IMG

- [ ] Quyết định dạng truy vấn (mô tả thực thể hay bộ ba khuyết đuôi)
- [ ] Loader + đồ thị, theo cùng cấu trúc module của Phase 2
- [ ] Chạy B1, Ours, A1, A2, A4 trên tập này

## Output

- `runs/phase5/<variant>-seed<n>.json`
- `reports/phase5-ablation-fidelity.md`: bảng ablation hai tập (mean ± std), biểu đồ quét α/K, bảng fidelity.

## Success Criteria

- [ ] Trên cả hai tập: A1 > A2 và A1 > A4, chênh lệch vượt sai số qua seed
- [ ] Xoá cạnh giải thích làm giảm điểm nhiều hơn xoá cạnh ngẫu nhiên
- [ ] Kết quả ngược cũ có nguyên nhân được ghi rõ

## Implementation update — ablation from main 49bd52d

Implemented `src/models/ablations.py`: full, objects_only, drop_predicates,
random_edges (independent endpoint permutations preserving directed degrees;
self-relations/duplicate triples allowed), and undirected (existing encoder variant).
Configuration lives in `configs/retrieval/ablations/*.yaml`; training, export,
checkpoint resume validation and explanation use the same intervention settings.
The obsolete pilot ablation tasks above refer to deleted modules, not this implementation.

Input: existing VG-COCO JSON + frozen embedding cache. Output: existing GraphSAGE
run layout and benchmark JSON, with variant-specific names. Acceptance: invariant
tests and training/checkpoint/export integration pass; final experiment acceptance
still requires all configured seeds on actual data, without tuning on test.

Verification: `python -m pytest -q` → 250 passed; targeted Ruff → clean;
`git diff --check` → clean. Real-data smoke blocked before training because
`data/processed/vg_coco/embeddings/openai--clip-vit-base-patch32/meta.json` is absent.
No final ablation metrics claimed. mypy unavailable in the environment.

Educational walkthrough: `private/ablation-guide.html` (offline source
excerpts with block notes, commands, intervention simulator and limitations).
Browser checks: no JS errors, clickable notes, syntax tokens, simulator and
390px layout without page overflow; print PDF generated as a temporary QA artifact.

Remaining: restore/rebuild embedding cache; train four variants with three seeds;
benchmark using the matching YAML; compare graph-only and fused results.
Fidelity, second dataset, hub-policy ablation and a parameter-matched undirected
control are not implemented by this update. W&B is disabled in the new configs.

### Actual local execution — 18/11/2026

Restored all 10,000 source images (no failed downloads), regenerated CLIP cache
(own-image cosine 0.3042 vs random 0.1515), and trained full plus four variants
for seeds 13/37/97 with W&B disabled. All 15 runs have best/last checkpoints and
finite unit graph vectors (10,000 × 512); all input hashes match across runs.
All five benchmarks completed with alpha/k selected on validation.

Test fused R@1 mean ± std (%): full 48.12 ± 0.23; objects_only 47.56 ± 0.16;
drop_predicates 48.06 ± 0.28; random_edges 47.79 ± 0.18; undirected 47.84 ± 0.15.
Full does not win on every metric: drop_predicates/undirected have higher test
R@10 and MRR in these runs. No statistical significance or fidelity claim.
Randomization audit: changed triplet multisets in 9,491 graphs and 96.3% of
triplet positions; all directed degrees and predicate histograms preserved.

Benchmark JSON: runs/benchmark/graphsage-*.json. Prior full benchmark and training
metadata backed up under private/ablation-execution/. CSV, complete metric table,
validation selections, artifact verification and execution logs are private/local;
see private/ablation-guide.html and private/ablation-execution/comparison.csv.
Remaining: fidelity, second dataset, hub ablation, parameter-matched direction
control and stronger statistical analysis. No commits made.

### Analysis, fidelity and commit-readiness verification — 18/11/2026

Added reusable paired-analysis and fidelity modules plus CLI/config/tests.
No retrieval model or hyperparameters changed. Image-cluster bootstrap (2,000
draws, three fixed checkpoints) gives full−masked-label ΔR@1 CI [-0.35, 0.45]
pp and full−random CI [-0.09, 0.74]. No clear semantic-label/connectivity
superiority established. Full−objects CI [0.05,1.06] is exploratory and conditional
on trained checkpoints, not an uncertainty interval over new training seeds.

Fidelity: 200 presampled test queries, 198/195/196 evaluated after explicit
small-graph skips, all three seeds. Joint selected deletion loses top-1 at
17.81% vs random 4.31% (10% requested budget), 25.45% vs 6.06% (25% budget).
Selection still uses single-deletion score drops; report as selection-dependent
perturbation evidence, not independent semantic faithfulness proof.

Safety fixes: finished resume verifies settings + required artifacts;
benchmark/explanation/fidelity check run provenance and stale benchmark inputs.
Full config is now public; historical private config paths in snapshots are not
needed to replay. Tests: 262 passed; Ruff src/scripts/tests clean; mypy src/scripts
clean. All 15 real runs passed resume validation without retraining.
Artifacts: runs/analysis/ablation.json and runs/fidelity/full-joint-test.json.
Interactive educational report stays ignored in private/. No commit created.

### Readability and artifact policy — 18/11/2026

Formatted new Python files completely and only changed regions of existing
modules; grouped operations and expanded substantial parameter lists. Added short
English processing comments and checked import ordering. Generated runs/private
outputs are ignored for future Git additions; already tracked historical results
remain tracked. Current acceptance status is consolidated in
`plan-task/requirements-checklist.md`; second dataset, hub control and semantic
validity remain open. This update does not rerun or alter model results.
