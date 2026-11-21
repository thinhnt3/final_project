**Nhận xét.**

| Run | R@1 | R@5 | R@10 | MRR | Graph-only R@1 | α |
|---|---|---|---|---|---|---|
| 01 · CLIP ViT-B/32 (zero-shot) | 43,89 | 72,19 | 82,64 | 56,76 | | 1 |
| 02 · Graph lexical match | 46,41 | | | | 12,63 | |
| 03 · GraphSAGE (learned image node) | 48,12 ± 0,23 | 75,33 | 83,87 | 60,42 | 27,54 | 0,80 |
| 04 · GraphSAGE (CLIP image node) | 49,43 ± 0,46 | 77,90 | 87,52 | 62,22 | 42,25 | 0,66 |

- GraphSAGE (learned image node) nâng R@1 của CLIP thêm **4,2 điểm** và hơn so khớp từ trên đồ thị 1,7 điểm. Vì image node không mang CLIP, phần tăng này đến từ đồ thị.
- Biến thể CLIP image node cao hơn 1,3 điểm nhưng graph-only score của nó đã lẫn CLIP ảnh (42,25 so với 27,54), nên không còn đo riêng được đóng góp của đồ thị.
- α ≈ 0,8: điểm đồ thị chỉ được tin khoảng 20%.
