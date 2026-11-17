---
title: "Phase 5: Tập công khai thứ hai, ablation, fidelity"
status: todo
---

# Phase 5: Tập công khai thứ hai, ablation, fidelity

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
