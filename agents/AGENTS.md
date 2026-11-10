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
