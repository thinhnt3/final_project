# Bảng việc

Cập nhật: 21/11/2026.

## Dữ liệu

- Dataset 1 - Visual Genome ∩ COCO - done
- Dataset 2 - MKG-W - đã dựng dữ liệu, mã hoá CLIP và chạy CLIP, GraphSAGE, path score, relation path scorer, AdaMF-MAT ([kết quả](docs/mkg-w-results.md)); chưa commit

## Mức 1

- CLIP vs CLIP + GraphSAGE - done
- Phân tích 5 truy vấn thành công, 5 thất bại - xong, chưa commit
- Ablation (bỏ cạnh, cạnh ngẫu nhiên) - đã chạy đủ 3 seed; code/kết quả đã merge main - Nghị

## Mức 2

- Thuật toán từ 2024 trở về sau: AdaMF-MAT - trên MKG-W đã tái lập đủ 3 seed (MRR 34,58 so với 35,85 của bài báo); ablation của bài báo đang chạy - Thịnh
- Bộ 20 truy vấn, tách nhóm trực tiếp và gián tiếp - MKG-W: báo cáo đã tách hai nhóm; có 30 câu gián tiếp viết tay chỉ để đánh giá (`data/splits/mkg_w_handwritten_queries.yaml`); VG ∩ COCO chưa ai nhận
- Ablation - Nghị

## Mức 3

- Xây MMKG: crawl dữ liệu, sinh text query thoả tiêu chí ([build_MMKG.md](docs/build_MMKG.md)) - Nghị
- Fidelity, validity, sparsity - đã chạy fidelity và sparsity/keep-only ba seed; có form audit validity mở rộng, còn thu nhãn người và đối chiếu với máy - Nghị

## Mức 4

- Mở rộng text-to-video - investigate - chưa ai nhận

## Chung

- Báo cáo - đang làm - Quốc Thịnh
- Demo - chưa ai nhận
