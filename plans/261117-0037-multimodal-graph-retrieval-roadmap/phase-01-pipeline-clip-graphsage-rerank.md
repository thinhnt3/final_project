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

- **Mục tiêu:** học GraphSAGE bằng InfoNCE trong batch (G7), chạy an toàn: theo dõi được, dừng giữa chừng không mất việc, tái lập được.
- **Input:** đồ thị con của ảnh train và validation, embedding caption train và validation.
- **Output:** `runs/graphsage/<tên lượt chạy>/`:
  - `config.yaml`: bản chụp toàn bộ config đã dùng, kèm commit git và sha256 của dữ liệu đầu vào
  - `history.json`: mỗi vòng một dòng, ghi ngay sau mỗi vòng
  - `train.log`: mọi dòng log
  - `last.pt`: checkpoint sau mỗi vòng, đủ để chạy tiếp
  - `best.pt`: checkpoint tốt nhất theo validation
  - `graph_vectors.npy` + ids: vector đồ thị của 10.000 ảnh, tính từ `best.pt` khi huấn luyện xong
- **Quy tắc dữ liệu:** mỗi ảnh một lần trong batch; mỗi vòng chọn ngẫu nhiên 1 trong 5 caption của ảnh; ảnh validation chỉ dùng để đánh giá; script huấn luyện không đọc tập test.

**Yêu cầu an toàn**

| Nhóm | Yêu cầu |
|---|---|
| Tham số | Mọi siêu tham số nằm trong mục `training` của `configs/retrieval/vg_coco.yaml`; không có giá trị nào viết trong code |
| Kiểm tra trước khi chạy | Embedding không cũ hơn dữ liệu; bảng khái niệm và vị từ khớp id; dựng được mọi đồ thị con; chạy thử một batch xuôi + ngược; thư mục lượt chạy đã có thì từ chối, trừ khi chạy tiếp hoặc được cho phép ghi đè |
| Theo dõi | Thanh tiến độ tqdm theo batch; sau mỗi vòng một dòng log gồm loss train, loss validation, R@1/5/10 và MRR validation của riêng điểm đồ thị, tốc độ học, thời gian vòng; log ghi cả ra màn hình và `train.log` |
| Checkpoint | `last.pt` sau mỗi vòng (trọng số, optimizer, số vòng, trạng thái sinh số ngẫu nhiên, metric tốt nhất); `best.pt` khi validation tốt hơn; ghi qua file tạm rồi đổi tên để không hỏng khi bị ngắt |
| Chạy tiếp | `--resume` đọc `last.pt` và tiếp tục đúng vòng kế; kết quả phải trùng với lượt chạy không bị ngắt |
| Khi lỗi | Loss thành NaN hoặc vô hạn thì dừng, báo lỗi, giữ nguyên checkpoint tốt trước đó; Ctrl-C thì lưu `last.pt` rồi thoát; cắt gradient theo ngưỡng trong config |
| Dừng sớm | Dừng khi metric validation không cải thiện sau số vòng `patience`; ghi lý do dừng |
| Tái lập | Seed trong config, gieo cho Python, NumPy, PyTorch; chạy lại cùng seed cho cùng `history.json` |

- **Xong khi:** (a) loss giảm; (b) trên tập nhỏ 64 ảnh mô hình học thuộc được (R@1 train gần 100%); (c) chạy lại cùng seed cho cùng kết quả validation; (d) ngắt giữa chừng rồi `--resume` cho cùng kết quả với lượt chạy liền mạch; (e) loss NaN làm dừng và không ghi đè `best.pt` (test); (f) R@K validation của riêng điểm đồ thị cao hơn mức ngẫu nhiên rõ rệt.

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
| 1.3 | Xong 17/11 | `src/models/graphsage.py`, mục `graphsage` trong config, `tests/models/test_graphsage.py` |
| 1.4 | Xong 17/11 (code, test, code-review, 3 seed) | `src/models/{train,train_data,checkpoint}.py`, `src/connectors/wandb_tracker.py`, `scripts/train_graphsage.py`, `runs/graphsage/directed-learned-l3-h512/seed{13,37,97}/` |
| 1.5–1.6 | Chưa làm | |

Kết quả 1.1: ảnh 10.000 × 512, caption 50.000 × 512, khái niệm 1.778 × 512, vị từ 493 × 512; float32, lệch độ dài tối đa 1,8e-7, không NaN. (b) Cosine caption với đúng ảnh 0,304, với ảnh ngẫu nhiên 0,152. (c) Mã hoá lại 4 caption và 3 ảnh: lệch tối đa 3,9e-7. (d) Test dùng dữ liệu dựng tay, không tải model. Chữ khái niệm đưa vào CLIP nguyên dạng ("pants"), không thêm câu mẫu; láng giềng gần nhất hợp lý (`couch` → loveseat, armchair). Chạy trên MPS khoảng 3 phút.

Kết quả 1.2: dựng 10.000 đồ thị con trong 0,2 giây. Đối chiếu với `graph_stats.json` của bước 2.7: 174.743 đỉnh đối tượng, 127.812 đỉnh phụ (mỗi bộ ba một đỉnh), 430.367 cạnh = 174.743 cạnh "chứa" + 2 × 127.812, khớp cả ba. Trung bình 31,3 đỉnh mỗi ảnh. Lỗi bắt được bằng test: PyG tự cộng dồn mọi trường có chữ "index" trong tên khi ghép batch, làm lệch chỉ số tra vector của các đồ thị con từ thứ hai trở đi; trường đổi tên thành `feature_row`.

Kết quả 1.3: `GraphImageEncoder` gồm 3 tầng có hướng (mỗi tầng 2 `SAGEConv`), tầng chiếu tuyến tính, chuẩn hoá; 3,41 triệu tham số học được với chiều ẩn 512. Bảng vector khái niệm, vị từ, ảnh là buffer đóng băng, không lưu trong checkpoint. Chạy thử trên dữ liệu thật: đầu ra (256, 512), dài 1, không NaN; xuôi + ngược cho 1.024 đồ thị mất 0,65 giây trên CPU và 1,15 giây trên MPS, nên huấn luyện trên CPU là đủ. Test bắt 10/10 lỗi gài. Hai điều test làm rõ: (1) 2 tầng đã cho ảnh biết vai của từng đối tượng trong relation; tầng 3 chỉ cần để ghép đúng hai đầu khi có nhiều relation cùng vị từ, tài liệu thiết kế đã sửa theo; (2) ở tầng cuối, lớp gộp chiều xuôi không nhận gradient qua hàng xóm vì đỉnh ảnh chỉ có cạnh đi ra, đây là tính chất kiến trúc.

Kết quả 1.4 (seed 13, cấu hình khởi đầu: 3 tầng có hướng, chiều ẩn 512, batch 256, lr 0,001, nhiệt độ 0,07): dừng sớm ở vòng 27, tốt nhất ở vòng 22; 126 giây trên CPU (khoảng 4,7 giây mỗi vòng). Validation, riêng điểm đồ thị: R@1 27,8 / R@5 57,2 / R@10 71,0 / MRR 41,7. Để so: so khớp từ (B4) riêng đồ thị trên validation đạt R@1 13,2; CLIP đạt 42,6. Loss train giảm đều 5,06 → 1,53; loss validation chạm đáy quanh vòng 20 (1,99) rồi nhích lên, tức bắt đầu quá khớp nhẹ. Chưa ghép với CLIP (bước 1.5) và chưa chấm trên test.

Kiểm tra an toàn đã làm: tham số nằm hết trong mục `training` và `wandb` của YAML; kiểm tra trước khi chạy (embedding không cũ, bảng khớp id, một batch chạy thử); tqdm theo batch; mỗi vòng ghi loss và R@K validation ra màn hình, `train.log`, `history.json` và W&B; `last.pt` mỗi vòng, `best.pt` khi tốt hơn, ghi qua file tạm; `--resume` cho kết quả trùng lượt chạy liền mạch (test); loss NaN dừng trước khi cập nhật trọng số và giữ checkpoint cũ (test); thư mục lượt chạy đã có thì từ chối trừ khi `--resume` hoặc `--overwrite`; cùng seed cho cùng `history.json` (test); mô hình học thuộc được 16 ảnh (test). Test bắt 17/17 lỗi gài. Ctrl-C: vòng đang chạy bị bỏ, các vòng đã xong nằm trong `last.pt`, chạy tiếp bằng `--resume`. W&B: theo yêu cầu user, chế độ online, project `vg-coco-graph-retrieval`.

Kết quả 1.4, ba seed (validation, riêng điểm đồ thị):

| Seed | Vòng tốt nhất / số vòng | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| 13 | 22 / 27 | 27,8 | 57,2 | 71,0 | 41,7 |
| 37 | 30 / 35 | 27,0 | 56,8 | 70,9 | 41,0 |
| 97 | 20 / 25 | 27,5 | 56,8 | 70,3 | 41,5 |

Code-review 1.4 (agent độc lập, 17/11): phần lõi đúng (hàm mất mát, chia batch, dừng sớm, xử lý NaN, không rò rỉ validation/test). Đã sửa:

1. `--overwrite` không xoá file của lượt cũ, nên `--resume` sau đó có thể chạy tiếp nhầm lượt cũ → giờ xoá mọi file lượt chạy đã ghi; `--resume` và `--overwrite` loại trừ nhau.
2. `--resume` không kiểm tra config và ghi đè bản chụp config → giờ so kiến trúc, model CLIP, sha256 dữ liệu và mọi tham số huấn luyện (trừ `max_epochs`, `patience`, `seeds`); khác thì từ chối; bản chụp gốc giữ nguyên.
3. Lỗi giữa chừng không vào `train.log`, lượt bị ngắt hiện "thành công" trên W&B → lỗi được ghi một lần ra cả màn hình và `train.log`; lượt hỏng đánh dấu thất bại trên W&B.
4. `--resume` với nhiều seed: seed đã xong thì bỏ qua, seed bị ngắt thì chạy tiếp, seed chưa chạy thì bắt đầu mới; chạy tiếp dùng lại cùng run W&B.
5. Thêm kiểm tra `best.pt` và `last.pt` khớp nhau khi chạy tiếp; gradient không hữu hạn báo `TrainingError` trước khi cập nhật trọng số; trạng thái sinh số ngẫu nhiên lưu trong `last.pt`; batch cuối chỉ có 1 ảnh được gộp vào batch trước; log và `history.json` ghi tốc độ học thật của optimizer.
6. Quyền riêng tư W&B: mặc định W&B gửi tên máy, email tài khoản, thông tin phần cứng và log màn hình. Giờ đã tắt, chỉ gửi config lượt chạy và metric. **Ba lượt chạy seed 13, 37, 97 đã tải lên trước khi tắt nên có chứa tên máy và email.**
7. Thêm test cho các chỗ review chỉ ra chưa có: ghép caption với đúng ảnh khi đánh giá, loss validation, khôi phục vòng tốt nhất khi chạy tiếp, xoá khi ghi đè. Đã chạy thử cả 6 tình huống vòng đời lượt chạy bằng script thật. 246 test.

Ghi nhận, không sửa: 233 câu caption train trùng nhau giữa các ảnh và 25 ảnh có đồ thị con giống hệt nhau là ảnh sai giả trong batch (ảnh hưởng nhỏ); loss validation phụ thuộc cỡ batch nên không so được giữa các cấu hình cỡ batch khác nhau; Ctrl-C bỏ vòng đang chạy, không lưu giữa vòng; script chưa có test tự động riêng.
