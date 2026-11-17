---
title: "Phase 1: Pipeline cơ bản: CLIP + GraphSAGE + Rerank"
status: in-progress
---

# Phase 1: Pipeline cơ bản: CLIP + GraphSAGE + Rerank

## Overview

Một pipeline chạy đầu-cuối trên VG ∩ COCO (10.000 ảnh): CLIP đóng băng → top-k cosine → điểm đồ thị từ nhúng GraphSAGE → xếp hạng lại theo α → relation giải thích.

Quyết định thiết kế (G1–G10) và lý do: `docs/graphsage-design-decisions.md`. Bản thử 40 ảnh cũ bị xoá và viết lại (G10). Làm tuần tự, mỗi bước có input/output kiểm được; bộ đo dùng chung nằm ở Phase 3.

## Thứ tự làm (user duyệt 17/11)

1.0 → 1.1 → Phase 3 (bộ đo + baseline CLIP) → B4 (Phase 4) → 1.2 → 1.3 → 1.4 → 1.5 → 1.6

B4 đi trước GraphSAGE: nếu riêng đồ thị xếp hạng gần mức ngẫu nhiên thì xem lại đồ thị trước khi huấn luyện GNN.

## Cấu trúc code

```
configs/retrieval/vg_coco.yaml    # model CLIP, đường dẫn, batch, lưới k và α, seed
src/embeddings/clip_encoder.py    # CLIP đóng băng: mã hoá ảnh và văn bản theo batch
src/embeddings/store.py           # lưu/đọc ma trận embedding + danh sách id + sha256
src/retrieval/config.py           # RetrievalConfig đọc từ YAML, validate khi load
src/retrieval/search.py           # top-k cosine
src/retrieval/rerank.py           # điểm = α·cosine + (1−α)·điểm đồ thị
src/evaluation/metrics.py         # R@K, MRR
src/evaluation/benchmark.py       # chạy một phương pháp trên một split, nhiều seed
src/models/graph_data.py          # image_graphs.json → đồ thị con PyG
src/models/graphsage.py           # GraphSAGE có hướng + tầng chiếu
src/models/train.py               # InfoNCE, dừng sớm theo validation
src/models/explain.py             # bỏ từng relation, đo điểm giảm
scripts/embed_vg_coco.py · run_benchmark.py · train_graphsage.py
tests/embeddings · tests/retrieval · tests/evaluation · tests/models
```

Quy ước như Phase 2: hàm thuần không đọc/ghi file; không hard-code đường dẫn hay ngưỡng; mọi thứ chọn trên validation; log, tên, comment bằng tiếng Anh.

## Các bước

### 1.0 Chuẩn bị

- **Mục tiêu:** repo sạch, sẵn để viết mới.
- **Việc:** gộp `feat/vg-coco-graph-build` vào `main`; xoá `src/models/`, `src/retrieval/`, `src/evaluation/`, `src/data/visual_genome.py`, các script và dữ liệu của tập 40 ảnh, `runs/*.json`; thêm `torch_geometric` vào `requirements.txt`; tạo `configs/retrieval/vg_coco.yaml` + `RetrievalConfig`.
- **Xong khi:** `pytest` qua; không còn file nào import module đã xoá.

### 1.1 Lưu embedding CLIP

- **Mục tiêu:** mọi bước sau đọc embedding từ đĩa, không gọi CLIP lại.
- **Input:** `data/processed/vg_coco/{images.json, queries.json, images/}`, `graph/{concepts.json, predicates.json}`.
- **Output:** `data/processed/vg_coco/embeddings/<tên model>/`: `images.npy` (10.000 × 512), `queries.npy` (50.000 × 512), `concepts.npy` (1.778 × 512), `predicates.npy` (493 × 512), mỗi file kèm danh sách id cùng thứ tự; `meta.json` (tên model, số chiều, sha256 của file đầu vào và đầu ra).
- **Quy tắc:** vector chuẩn hoá độ dài 1; float32; thứ tự hàng = thứ tự id trong file id; ảnh mở bằng Pillow, chuyển RGB.
- **Xong khi:** (a) kích thước đúng, không có NaN, mọi vector dài 1; (b) cosine giữa caption và đúng ảnh của nó cao hơn rõ rệt so với ảnh ngẫu nhiên (ghi cả hai số); (c) chạy lại một phần nhỏ cho cùng vector trong sai số 1e-5; (d) test không cần mạng và không cần tải model (dùng bộ mã hoá giả trong test).

### 1.2 Đồ thị con PyG

- **Mục tiêu:** mỗi ảnh thành một đồ thị con theo G2.
- **Input:** `graph/image_graphs.json`, embedding khái niệm và vị từ.
- **Output:** hàm dựng `Data` cho một ảnh: đỉnh 0 là đỉnh ảnh; đỉnh đối tượng (một đỉnh mỗi khái niệm); đỉnh phụ (một đỉnh mỗi bộ ba). Cạnh có hướng: ảnh → đối tượng; chủ thể → đỉnh phụ → đối tượng. Kèm loại đỉnh và chỉ số tra vector ban đầu.
- **Xong khi:** test trên đồ thị viết tay: số đỉnh, số cạnh, hướng cạnh, ảnh không có bộ ba, ảnh không có khái niệm (chỉ còn đỉnh ảnh); ghép batch giữ đúng đỉnh ảnh của từng đồ thị con.

### 1.3 Mô hình

- **Mục tiêu:** GraphSAGE có hướng, 3 tầng (G4, G5, G9), ra vector đồ thị của ảnh trong không gian CLIP.
- **Thiết kế:** mỗi tầng = `SAGEConv` chiều xuôi + `SAGEConv` chiều ngược, cộng lại; ReLU giữa các tầng, không ReLU sau tầng cuối; lấy vector của đỉnh ảnh → tầng chiếu tuyến tính → chuẩn hoá. Đỉnh ảnh: vector học được dùng chung (chính) hoặc vector CLIP ảnh (biến thể), chọn bằng config. Cờ tắt hướng để làm ablation.
- **Xong khi:** test: đầu ra đúng kích thước và dài 1; đổi chiều một relation làm vector đổi (có hướng) và không đổi khi tắt hướng; hai đồ thị con trong một batch không ảnh hưởng nhau; có thành phần âm ở đầu ra.

### 1.4 Huấn luyện

- **Mục tiêu:** học GraphSAGE bằng InfoNCE trong batch (G7).
- **Input:** đồ thị con của ảnh train, embedding caption train.
- **Output:** `runs/graphsage/<tên>/`: checkpoint tốt nhất theo validation, `history.json` (loss và R@K validation từng vòng), `config.yaml` đã dùng, vector đồ thị của 10.000 ảnh.
- **Quy tắc:** mỗi ảnh một lần trong batch; mỗi vòng chọn ngẫu nhiên 1 trong 5 caption của ảnh; dừng sớm theo R@K trên validation; seed cố định; ảnh validation và test không bao giờ vào batch huấn luyện.
- **Xong khi:** loss giảm; trên một tập nhỏ 64 ảnh mô hình học thuộc được (R@1 train gần 100%); chạy lại cùng seed cho cùng kết quả validation; R@K validation của riêng điểm đồ thị cao hơn mức ngẫu nhiên rõ rệt.

### 1.5 Xếp hạng lại

- **Mục tiêu:** điểm cuối theo đề bài, α và k chọn trên validation.
- **Output:** bảng CLIP so với CLIP + Graph trên test, cả hai biến thể đỉnh ảnh, qua bộ đo của Phase 3.
- **Xong khi:** α = 1 cho đúng kết quả của baseline CLIP; α và k chỉ được chọn trên validation; kết quả test chạy một lần sau khi đã chốt tham số.

### 1.6 Giải thích

- **Mục tiêu:** trả về relation đóng góp vào điểm đồ thị (G8).
- **Cách:** với mỗi ứng viên, bỏ từng relation, tính lại điểm đồ thị, xếp relation theo mức giảm.
- **Xong khi:** mỗi kết quả kèm top relation và mức giảm điểm; test: relation không liên quan truy vấn có mức giảm nhỏ hơn relation khớp truy vấn trên ví dụ dựng tay.

## Success Criteria

- [ ] Một lệnh cho ra bảng CLIP và CLIP + Graph trên cùng tập ứng viên, kèm relation giải thích
- [ ] Điểm đồ thị lấy từ nhúng GraphSAGE, không phải so khớp từ
- [ ] Không tham số nào được chọn trên test
- [ ] `pytest` qua, không cần mạng

## Tiến độ

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| 1.0 | Xong 17/11 | Nhánh feat gộp vào `main` (`a795fd1`); code, script, config và dữ liệu của tập 40 ảnh đã xoá; PyG 2.8.1; `configs/retrieval/vg_coco.yaml` |
| 1.1 | Xong 17/11 | `data/processed/vg_coco/embeddings/openai--clip-vit-base-patch32/` (122 MB, không commit) + `meta.json` |
| 1.2 | Xong 17/11 | `src/models/graph_data.py` + `tests/models/test_graph_data.py` |
| 1.3–1.6 | Chưa làm | |

Kết quả 1.1: ảnh 10.000 × 512, caption 50.000 × 512, khái niệm 1.778 × 512, vị từ 493 × 512; float32, lệch độ dài tối đa 1,8e-7, không NaN. (b) Cosine caption với đúng ảnh 0,304, với ảnh ngẫu nhiên 0,152. (c) Mã hoá lại 4 caption và 3 ảnh: lệch tối đa 3,9e-7. (d) Test dùng dữ liệu dựng tay, không tải model. Chữ khái niệm đưa vào CLIP nguyên dạng ("pants"), không thêm câu mẫu; láng giềng gần nhất hợp lý (`couch` → loveseat, armchair). Chạy trên MPS khoảng 3 phút.

Kết quả 1.2: dựng 10.000 đồ thị con trong 0,2 giây. Đối chiếu với `graph_stats.json` của bước 2.7: 174.743 đỉnh đối tượng, 127.812 đỉnh phụ (mỗi bộ ba một đỉnh), 430.367 cạnh = 174.743 cạnh "chứa" + 2 × 127.812, khớp cả ba. Trung bình 31,3 đỉnh mỗi ảnh. Lỗi bắt được bằng test: PyG tự cộng dồn mọi trường có chữ "index" trong tên khi ghép batch, làm lệch chỉ số tra vector của các đồ thị con từ thứ hai trở đi; trường đổi tên thành `feature_row`.
