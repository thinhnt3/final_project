**Nhận xét.**

| Run | all | → edge masking | contrast | → edge masking |
|---|---|---|---|---|
| 01 · CLIP (reference) | 28,8 | | 32,6 | |
| 17 · AdaMF + L2 (baseline) | 27,7 | 27,7 | 33,6 | 33,7 |
| 19 · + structure-only dropout 0.3 | 30,0 | 28,5 | 35,0 | 33,2 |
| 20 · + structure-only loss 0.1 | 30,0 | 28,5 | 34,9 | 33,1 |
| 21 · dropout 0.3 + loss 0.1 | 30,0 | 28,4 | 34,9 | 33,0 |

Paired bootstrap theo cụm ảnh, so với baseline, CI 95%: all +2,3 [1,2; 3,4] cho cả ba cấu hình; contrast +1,3 đến +1,4, CI chứa 0 (so0.3 +1,42 [−0,14; 3,08]). Mức tụt khi edge masking trên contrast: baseline −0,05 [−0,14; 0]; so0.3 +1,83 [1,28; 2,43]; sl0.1 +1,74; cả hai +1,93.

- **Baseline không đọc cạnh:** edge masking không đổi gì, điểm còn thấp hơn CLIP. Mô hình đi shortcut qua CLIP ảnh ở image modality.
- **Ép structure modality tự khớp caption làm mô hình bắt đầu đọc cạnh** (mức tụt có ý nghĩa thống kê), đồng thời R@1 caption tăng khoảng 1 điểm.
- **Phân biệt quan hệ A với B** giữa cùng hai object (contrast) chưa tốt hơn baseline một cách có ý nghĩa thống kê, và chỉ ngang CLIP.

**Indirect queries (WordNet IsA).** Thêm cạnh IsA nối object với khái niệm cha (tối đa 2 tầng, `cat → feline → carnivore`). *Indirect query* gọi subject bằng khái niệm cha (`carnivore sitting in chair`), *direct twin* dùng tên gốc (`cat sitting in chair`); 1.387 cặp, ngẫu nhiên trong nhóm confuser 17,2%. Graph-only, 3 seed:

| Model | Graph | Indirect R@1 | Indirect confuser acc@1 | Direct R@1 | Direct confuser acc@1 |
|---|---|---|---|---|---|
| CLIP (reference) | không dùng | 3,8 | 23,9 | 7,6 | 27,3 |
| AdaMF L2 | base | 5,3 | 24,2 | 9,1 | 27,2 |
| AdaMF L2 | KG (+ IsA) | 5,0 | 25,0 | 8,4 | 26,7 |
| AdaMF L2 | KG, IsA masked | 4,1 | 24,3 | 8,4 | 26,9 |
| AdaMF so0.3 | base | 5,3 | 24,7 | 9,5 | 28,6 |
| AdaMF so0.3 | KG (+ IsA) | 5,3 | 25,2 | 9,1 | 27,7 |
| AdaMF so0.3 | KG, IsA masked | 4,7 | 25,8 | 9,2 | 28,3 |

Mô hình có dùng cạnh IsA (masking làm indirect R@1 tụt, direct không đổi) nhưng IsA **không giúp**: CLIP text của `cat` đã đủ gần `carnivore`.
