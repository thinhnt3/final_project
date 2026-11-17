---
title: "Phase 4: So sánh baseline vs CLIP + Graph"
status: todo
---

# Phase 4: So sánh baseline vs CLIP + Graph

## Overview

Bảng so sánh chính trên tập test VG ∩ COCO của Phase 2: phương pháp của mình có hơn các cách đã có không, và có giữ được khi đổi backbone không. Kèm phân tích 5 truy vấn thành công và 5 thất bại. Phần "cải thiện đến từ đâu" (ablation) nằm ở Phase 5.

## Mục tiêu

- Chứng minh CLIP + Graph hơn CLIP zero-shot trên cùng tập ứng viên.
- Chứng minh phần tăng không phụ thuộc một backbone (CLIP và OpenCLIP).
- Đặt phương pháp cạnh một reranker mạnh (cross-encoder) để trả lời "sao không dùng BLIP để xếp lại".

## Input

- `data/processed/vg_coco/` + `data/splits/vg_coco.json` (Phase 2).
- Đồ thị + nhúng GraphSAGE (Phase 1, 2.7).
- Bộ đo và α, K đã chọn trên val (Phase 3).

## Baseline

| # | Phương pháp | Vai trò | Ghi chú |
|---|---|---|---|
| B1 | CLIP ViT-B/32 zero-shot | Mốc bắt buộc (= α = 1) | Đã có `src/models/embeddings.py` |
| B2 | OpenCLIP ViT-B/32 (LAION-2B) zero-shot | Backbone thứ hai | Cùng giao diện encode |
| B3 | B1 + BLIP ITM xếp lại top-K | Reranker cross-encoder mạnh | **Tạm bỏ** (user chốt 17/11); làm lại nếu còn thời gian, cần GPU |
| B4 | B1 + ghép từ khoá đồ thị, không GNN | Reranker đồ thị đơn giản | Gần như `graph_score` hiện tại |
| Ours | B1/B2 + điểm GraphSAGE | Phương pháp chính | Chạy trên cả hai backbone |

Quy tắc so sánh: mọi reranker dùng chung top-K ứng viên của cùng backbone; α, K chọn trên val; báo cáo mean ± std qua 3 seed huấn luyện (13, 37, 97).

Lưu ý đối chiếu số liệu: gallery test là 750 ảnh, gần với COCO 1K test, nên chỉ đối chiếu B1 với số CLIP đã công bố trên COCO 1K như kiểm tra độ hợp lý, không như kết quả so sánh trực tiếp.

## Output

- `runs/phase4/<method>-<backbone>-seed<n>.json`: metric + xếp hạng từng truy vấn.
- `reports/phase4-comparison.md`: bảng R@1/5/10, MRR (mean ± std); 10 phân tích truy vấn.

## Todo

- [ ] Thêm encoder OpenCLIP cùng giao diện với CLIP
- [ ] Chạy B1, B2, B4, Ours trên test
- [ ] Chọn 5 truy vấn hạng tăng nhiều nhất, 5 truy vấn giảm/không cải thiện; mỗi truy vấn có ảnh đúng, hạng trước/sau, đường đi đã dùng, nguyên nhân
- [ ] Phân loại nguyên nhân thất bại

## Success Criteria

- [ ] Bảng đủ các dòng trên, cùng tập ứng viên, có sai số
- [ ] B1 trên test nằm trong khoảng hợp lý so với số CLIP công bố trên COCO 1K
- [ ] 10 phân tích truy vấn có đường đi giải thích
