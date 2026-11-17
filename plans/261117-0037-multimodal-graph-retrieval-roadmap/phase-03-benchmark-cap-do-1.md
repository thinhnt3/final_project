---
title: "Phase 3: Benchmark đánh giá, hoàn tất nhiệm vụ 1"
status: in-progress
---

# Phase 3: Benchmark đánh giá, hoàn tất nhiệm vụ 1

## Overview

Bộ đo Recall@1/5/10 và MRR dùng chung cho mọi phương pháp, trên tập test cố định (1.500 ảnh, 7.500 truy vấn). α và siêu tham số chỉ chọn trên validation. Làm ngay sau bước 1.1 của Phase 1, trước GraphSAGE.

## Các bước

### 3.1 Metric

- **Output:** `src/evaluation/metrics.py`: R@K và MRR từ thứ hạng của ảnh đúng.
- **Quy tắc:** mỗi truy vấn có đúng 1 ảnh đúng; điểm bằng nhau thì xếp theo cách bất lợi nhất cho ảnh đúng (không được lợi nhờ thứ tự mảng).
- **Xong khi:** test trên dữ liệu đồ chơi tính tay: hạng 1, hạng K, hạng K + 1, điểm bằng nhau.

### 3.2 Bộ chạy benchmark

- **Input:** một hàm chấm điểm (ma trận điểm truy vấn × ảnh của một split), split, danh sách seed.
- **Output:** `runs/benchmark/<phương pháp>.json`: metric từng seed, mean ± std, config, sha256 của embedding và split.
- **Quy tắc:** gallery của một split chỉ gồm ảnh của split đó; phương pháp không có yếu tố ngẫu nhiên thì ghi 1 lần chạy, không giả nhiều seed.
- **Xong khi:** chạy lại ra đúng cùng con số; test xác nhận ảnh train không lọt vào gallery test.

### 3.3 Baseline CLIP (B1)

- **Output:** R@1/5/10 và MRR của CLIP ViT-B/32 trên validation và test.
- **Kiểm tra hợp lý:** so với số CLIP ViT-B/32 zero-shot đã công bố trên COCO (tra lại nguồn trước khi ghi; gallery ở đây 1.500 ảnh, khác 1K và 5K của COCO nên chỉ so khoảng).
- **Xong khi:** con số nằm trong khoảng hợp lý; nếu lệch xa thì dừng lại tìm lỗi trước khi làm bước sau.

### 3.4 Quét α và k

- **Output:** hàm quét lưới α và k trên validation, trả cấu hình tốt nhất: R@1 cao nhất, hoà thì MRR cao nhất (so sau khi làm tròn 6 chữ số), vẫn hoà thì chọn cấu hình gần CLIP thuần nhất (α lớn nhất, rồi k nhỏ nhất). Điểm đồ thị không thêm thông tin thì kết quả là CLIP thuần.
- **Quy tắc:** hai loại điểm phải nằm trong [-1, 1]; hàm quét từ chối mọi split khác validation.
- **Xong khi:** test: α = 1 cho đúng thứ hạng của CLIP; kết quả test không được đọc trong lúc quét.

## Success Criteria

- [ ] Cùng tập truy vấn, cùng tập ảnh, cùng metric cho mọi phương pháp
- [ ] α và siêu tham số chỉ chọn trên validation
- [ ] Phương pháp có huấn luyện: mean ± std qua ít nhất 3 seed
- [ ] Chạy lại ra đúng cùng con số

## Tiến độ

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| 3.1 | Xong 17/11 | `src/evaluation/metrics.py` + test |
| 3.2 | Xong 17/11 | `src/evaluation/benchmark.py`, `scripts/run_benchmark.py`; chạy 2 lần trùng sha256 |
| 3.3 | Xong 17/11 | `runs/benchmark/clip.json` |
| 3.4 | Code + test xong 17/11; chạy thật khi có điểm đồ thị (B4, GraphSAGE) | `src/retrieval/rerank.py` |

Kết quả 3.3, CLIP ViT-B/32, gallery 1.500 ảnh, 7.500 truy vấn mỗi split:

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| validation | 42,6 | 70,8 | 82,1 | 55,7 |
| test | 43,9 | 72,2 | 82,6 | 56,8 |

Đã tính lại độc lập bằng numpy thuần, khớp cả 4 số của test. Mức ngẫu nhiên R@1 = 0,07%. Số đã công bố của CLIP ViT-B/32 zero-shot trên COCO 5K (text → image): R@1 30,4 / R@5 54,8 / R@10 66,1 (bảng 9 của paper LAION-5B, arXiv 2210.08402). Gallery ở đây nhỏ hơn 3,3 lần nên số cao hơn là hợp lý; không so trực tiếp được.

Quy ước 3.4 (user duyệt 17/11): điểm cuối dùng cosine thô theo đúng công thức đề bài, không chuẩn hoá thang điểm; α quét với bước 0,01 trên [0, 1] (`benchmark.alpha_step`). Số đo dẫn tới lựa chọn này ghi ở Phase 4.

Code-review 1.1 và Phase 3 (agent độc lập, 17/11): tính lại độc lập khớp cả validation và test; metric, cách chọn gallery và truy vấn đúng. Đã sửa 4 lỗi ở phần xếp hạng lại, chưa lỗi nào ảnh hưởng tới số đã ghi:

1. α = 1 không cho đúng thứ hạng CLIP trên điểm float32 (trừ 3 làm các điểm sát nhau thành bằng nhau; 1 truy vấn validation tụt từ hạng 38 xuống 39) → tính bằng float64. Kiểm lại trên validation: 0 truy vấn đổi hạng ở k = 1, 5, 10, 50, 1.500.
2. Chọn cấu hình tốt nhất bị nhiễu số học chi phối, và khi hoà thì nghiêng về điểm đồ thị → luật chọn mới ở 3.4. Kiểm lại: điểm đồ thị toàn 0 cho kết quả α = 1.
3. Không kiểm khoảng giá trị của điểm → báo lỗi khi điểm ngoài [-1, 1].
4. Không có gì ngăn quét α trên test → hàm quét từ chối split khác validation.

Sửa thêm: benchmark báo lỗi khi embedding cũ hơn dữ liệu (so sha256 của file đầu vào); truy vấn trỏ tới ảnh không tồn tại báo `DataError`; `Device` chuyển sang `src/embeddings/`. Bộ test xếp hạng lại bắt 12/12 lỗi gài. 168 test. Còn mở: ruff và mypy chưa cài; hai script chưa có test; `agents/AGENTS.md` (nhật ký kiểm tra ngày 10/11) còn trỏ tới các file đã xoá.
