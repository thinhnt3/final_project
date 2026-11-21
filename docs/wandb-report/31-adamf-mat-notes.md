**Nhận xét.**

| Run | R@1 | R@5 | R@10 | MRR | Graph-only R@1 | α |
|---|---|---|---|---|---|---|
| 01 · CLIP (zero-shot) | 43,89 | 72,19 | 82,64 | 56,76 | | 1 |
| 03 · GraphSAGE (learned image node) | 48,12 | 75,33 | 83,87 | 60,42 | 27,54 | 0,80 |
| 04 · GraphSAGE (CLIP image node) | 49,43 | 77,90 | 87,52 | 62,22 | 42,25 | 0,66 |
| 09 · Linear adapter | 49,76 | 77,86 | 87,33 | 62,44 | 46,25 | 0,63 |
| 10 · AdaMF-MAT (full) | 51,42 | 79,36 | 88,91 | 63,98 | 48,48 | 0,58 |
| 17 · AdaMF + L2 (5 seed) | 51,93 | 79,81 | 89,23 | 64,47 | 48,62 | 0,55 |
| 19 · L2 + structure-only dropout 0.3 (5 seed) | 52,74 | 80,64 | 89,68 | 65,19 | 49,77 | 0,52 |
| 20 · L2 + structure-only loss 0.1 (5 seed) | **53,1** | | | | | |

- AdaMF-MAT hơn GraphSAGE **3,3 điểm** R@1 và được tin gần ngang CLIP (α ≈ 0,55 so với 0,8).
- Nhưng linear adapter, không dùng đồ thị, đã hơn GraphSAGE (CLIP image node) 0,3 điểm: phép gộp hàng xóm của GraphSAGE không thêm gì so với một Linear trên CLIP ảnh. Phần tăng của AdaMF-MAT so với adapter (+1,7) gần như toàn bộ nhờ text modality (tên object).
- Chỉ khi ép structure modality tự khớp caption (run 19, 20) mô hình mới vượt baseline L2 khoảng 1 điểm; đây là phần tăng đầu tiên đến từ cạnh quan hệ. Số run 20 lấy từ báo cáo ablation, file kết quả thô không còn.
- Đường học dừng bằng early stopping theo val R@1 (patience 5), checkpoint tốt nhất được dùng để chấm test.
