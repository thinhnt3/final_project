---
title: "Phase 4: So sánh baseline vs CLIP + Graph"
status: in-progress
---

# Phase 4: So sánh baseline vs CLIP + Graph

## Overview

Bảng so sánh chính trên tập test VG ∩ COCO của Giai đoạn 2: phương pháp của mình có hơn các cách đã có không, và có giữ được khi đổi backbone không. Kèm phân tích 5 truy vấn thành công và 5 thất bại. Phần "cải thiện đến từ đâu" (ablation) nằm ở Giai đoạn 5.

## Mục tiêu

- Chứng minh CLIP + Graph hơn CLIP zero-shot trên cùng tập ứng viên.
- Chứng minh phần tăng không phụ thuộc một backbone (CLIP và OpenCLIP).
- Đặt phương pháp cạnh một reranker mạnh (cross-encoder) để trả lời "sao không dùng BLIP để xếp lại".

## Input

- `data/processed/vg_coco/` + `data/splits/vg_coco.json` (Giai đoạn 2).
- Đồ thị + nhúng GraphSAGE (Giai đoạn 1, 2.7).
- Bộ đo và α, K đã chọn trên val (Giai đoạn 3).

## Baseline

| # | Phương pháp | Vai trò | Ghi chú |
|---|---|---|---|
| B1 | CLIP ViT-B/32 zero-shot | Mốc bắt buộc (= α = 1) | Xong 17/11: `runs/benchmark/clip.json` |
| B2 | OpenCLIP ViT-B/32 (LAION-2B) zero-shot | Backbone thứ hai | Cùng giao diện encode |
| B3 | B1 + BLIP ITM xếp lại top-K | Reranker cross-encoder mạnh | **Tạm bỏ** (user chốt 17/11); làm lại nếu còn thời gian, cần GPU |
| B4 | B1 + so khớp từ giữa caption và đồ thị, không GNN | Reranker đồ thị đơn giản | Xong 17/11: `src/retrieval/lexical.py`, `runs/benchmark/graph_lexical*.json` |
| Ours | B1/B2 + điểm GraphSAGE | Phương pháp chính | Chạy trên cả hai backbone |

Quy tắc so sánh: mọi reranker dùng chung top-K ứng viên của cùng backbone; α, K chọn trên val; báo cáo mean ± std qua 3 seed huấn luyện (13, 37, 97).

Lưu ý đối chiếu số liệu: gallery test là 1.500 ảnh, khác 1K và 5K của COCO, nên số CLIP đã công bố chỉ dùng để kiểm tra độ hợp lý (xem Giai đoạn 3, bước 3.3), không so trực tiếp.

## Output

- `runs/benchmark/<method>.json`: metric từng split, α và k chọn trên validation.
- `reports/phase4-comparison.md`: bảng R@1/5/10, MRR (mean ± std); 10 phân tích truy vấn.

## Todo

- [ ] Thêm encoder OpenCLIP cùng giao diện với CLIP
- [ ] Chạy B1, B2, B4, Ours trên test
- [ ] Chọn 5 truy vấn hạng tăng nhiều nhất, 5 truy vấn giảm/không cải thiện; mỗi truy vấn có ảnh đúng, hạng trước/sau, đường đi đã dùng, nguyên nhân
- [ ] Phân loại nguyên nhân thất bại

## Success Criteria

- [ ] Bảng đủ các dòng trên, cùng tập ứng viên, có sai số
- [x] B1 trên test nằm trong khoảng hợp lý so với số CLIP đã công bố
- [ ] 10 phân tích truy vấn có đường đi giải thích

## Kết quả B1 và B4 (17/11)

B4 chấm điểm đồ thị bằng tỉ lệ từ nội dung của caption có trong chữ của đồ thị ảnh, có trọng số IDF (IDF tính trên đồ thị của ảnh train). Số trên test, gallery 1.500 ảnh, 7.500 truy vấn; α và k chọn trên validation.

| Phương pháp | R@1 | R@5 | R@10 | MRR | α, k |
|---|---|---|---|---|---|
| B1: CLIP | 43,9 | 72,2 | 82,6 | 56,8 | |
| Chỉ đồ thị, khái niệm + vị từ | 12,6 | 29,4 | 39,9 | 21,6 | |
| Chỉ đồ thị, chỉ khái niệm | 11,0 | 27,2 | 37,9 | 19,9 | |
| B4: CLIP + đồ thị, khái niệm + vị từ | 46,4 | 74,8 | 85,4 | 59,3 | 0,91 · 50 |
| B4: CLIP + đồ thị, chỉ khái niệm | 46,5 | 74,8 | 85,3 | 59,3 | 0,90 · 100 |

Nhận xét:

- Riêng đồ thị đạt R@1 12,6%, so với mức ngẫu nhiên 0,07%: đồ thị đủ thông tin để xếp hạng, nên huấn luyện GraphSAGE là có cơ sở.
- So khớp từ đã nâng CLIP thêm 2,5–2,6 điểm R@1. **Mốc GraphSAGE phải vượt là B4 (46,4–46,5), không chỉ CLIP (43,9).**
- Vị từ giúp khi chỉ dùng đồ thị (12,6 so với 11,0) nhưng không giúp khi ghép với CLIP (46,4 so với 46,5). So khớp từng từ không phân biệt được "man riding bike" với "man" và "bike" đứng riêng; đây là chỗ GraphSAGE có thể hơn.
- Lưới α (user duyệt 17/11): bước 0,01 trên [0, 1], giữ nguyên công thức đề bài, không chuẩn hoá thang điểm. Đo trên validation với điểm B4: lưới bước 0,1 cho R@1 46,04; lưới bước 0,01 cho 46,39 (đỉnh hẹp quanh α = 0,91–0,93); chuẩn hoá z-score cho 45,6–45,7, kém hơn. Độ phân tán điểm trong một truy vấn: CLIP 0,036, B4 0,058, chỉ lệch 1,6 lần. Bảng trên là số sau khi đổi lưới.

Review B4 (agent độc lập, 17/11): không có lỗi phải sửa; tính lại độc lập R@1 "chỉ đồ thị" trên test = 0,1263, khớp; IDF chỉ từ train, α và k chỉ chọn trên validation. Theo góp ý: phần ghép dữ liệu chuyển từ script vào `src/retrieval/lexical.py` và có test cho thứ tự hàng/cột, cờ vị từ và IDF chỉ từ train; thiếu đồ thị hoặc thiếu chữ truy vấn báo `DataError`. Kết quả không đổi sau khi sửa. Giới hạn đã biết: hàm bỏ đuôi số nhiều thô (`glasses` không khớp `glass`, `men` không khớp `man`), nên B4 hơi bị đánh giá thấp. Lint (ruff) và kiểu (mypy) sạch; 189 test.

## Kết quả Ours trên backbone CLIP (17/11)

CLIP + GraphSAGE (3 tầng có hướng, đỉnh ảnh là vector học được) trên test: R@1 48,20 ± 0,05, R@5 75,87 ± 0,18, R@10 86,36 ± 0,13, MRR 60,74 ± 0,10 (3 seed). Hơn CLIP 4,3 điểm R@1 và hơn B4 1,7–1,8 điểm. Chi tiết ở Giai đoạn 1, bước 1.5. Chưa làm: biến thể đỉnh ảnh mang CLIP, backbone OpenCLIP (B2), 10 phân tích truy vấn.

Bổ sung 17/11: biến thể đỉnh ảnh mang CLIP đạt R@1 49,43 ± 0,46 trên test; lưới k mở rộng tới cả gallery không đổi kết quả. Bảng đầy đủ ở Giai đoạn 1, mục "Bổ sung 1.5".
