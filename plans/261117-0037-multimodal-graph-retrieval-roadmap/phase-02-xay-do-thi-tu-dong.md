---
title: "Phase 2: Dữ liệu VG ∩ COCO + pipeline xây đồ thị"
status: todo
---

# Phase 2: Dữ liệu VG ∩ COCO + pipeline xây đồ thị

## Overview

Dựng tập dữ liệu thật cho Task 1–4 từ phần giao Visual Genome ∩ COCO: ảnh VG có `coco_id`, truy vấn là caption COCO (độc lập với scene graph), nhãn đồ thị chuẩn hoá theo synset WordNet có sẵn trong VG, chia split bằng scikit-learn. Làm tuần tự, mỗi bước có input/output kiểm được. **Duyệt dữ liệu (bước 2.6) là cổng bắt buộc trước khi xây đồ thị (bước 2.7).**

Cách làm: code từng bước → test → chạy → xem output → mới sang bước sau. Không chạy gộp nhiều bước khi output bước trước chưa được xem.

## Quyết định đã chốt

| Quyết định | Giá trị | Nguồn |
|---|---|---|
| Tập ảnh | VG có `coco_id` (51.498 / 108.077 ảnh, đếm từ `image_data.json`) | Kiểm trên máy |
| Truy vấn | Caption COCO, 5 câu/ảnh, mỗi câu có đúng 1 ảnh đúng | Brainstorm |
| Test | Chỉ lấy từ VG ∩ Karpathy test (phương án B) | User chốt |
| Ảnh Karpathy test | Không bao giờ vào train/val, kể cả ảnh không được chọn vào test | Suy ra từ B |
| Chuẩn hoá đối tượng | Synset đầu tiên của object trong `scene_graphs.json`; nhãn = lemma (`bicycle.n.01` → `bicycle`) | User chốt |
| Chuẩn hoá vị từ | Chữ thường, gộp khoảng trắng; **không** dùng synset (`ON` → `along.r.01` làm mất nghĩa) | Kiểm trên máy |
| Chia split | scikit-learn `train_test_split`, `random_state` cố định, phân tầng theo mật độ đồ thị | User chốt |
| Quy mô | 10.000 ảnh: train 7.000 / val 1.500 / test 1.500 (7.500 truy vấn test). Nâng từ 5.000 theo yêu cầu user 17/11 | Plan gốc 70/15/15 |
| Ảnh dùng | Ảnh VG (toạ độ bbox khớp scene graph), không dùng file ảnh COCO | Brainstorm |

Giới hạn đã biết: synset gộp số ít/nhiều và biến thể cùng nghĩa (`bike`, `bikes` → `bicycle.n.01`) nhưng **không** gộp từ gần nghĩa (`guy.n.01` ≠ `man.n.01`, đã kiểm trên ảnh 1). Gộp theo cây WordNet/ConceptNet thuộc Phase 7; bước 2.6 đo xem có cần sớm hơn không.

## Cấu trúc code

```
configs/data/vg_coco.yaml        # URL, đường dẫn, kích thước split, seed, ngưỡng lọc
src/data/
  config.py                      # DataConfig (dataclass frozen) đọc từ YAML, validate khi load
  schemas.py                     # dataclass: ImageRecord, Caption, SceneObject, Relation, SceneGraph
  sources.py                     # tải file thô + sha256 + MANIFEST
  coco_captions.py               # đọc dataset_coco.json (Karpathy) → {coco_id: split, captions}
  vg_coco_join.py                # lọc VG theo coco_id, ghép caption, ghi funnel
  synsets.py                     # hàm thuần: synset → lemma, chuẩn hoá vị từ
  scene_graphs.py                # đọc + chuẩn hoá scene graph qua synsets.py
  splits.py                      # viết lại bằng sklearn; giữ validate_no_overlap
  graph_build.py                 # từ vựng chỉ từ train + đồ thị từng ảnh (bước 2.7)
  download.py                    # giữ, thêm retry + báo ảnh lỗi
scripts/
  build_vg_coco.py               # --stage sources|join|normalize|split|export|all
  review_vg_coco.py              # sinh báo cáo duyệt dữ liệu (bước 2.6)
tests/data/                      # test từng module trên fixture nhỏ, không cần mạng
tests/fixtures/vg_coco/          # 5–10 bản ghi VG + caption viết tay, phủ các ca biên
```

Nguyên tắc:
- Hàm thuần (chuẩn hoá, ghép, chia) không đọc/ghi file; I/O chỉ ở `sources.py`, hàm load/save và script.
- Không hard-code đường dẫn hay ngưỡng; mọi thứ đi qua `DataConfig`.
- Mỗi stage đọc output stage trước từ đĩa → chạy lại từng stage riêng được.
- Mỗi hàm public có type hint + docstring 1–2 dòng; log/tên/comment bằng tiếng Anh.

Thư mục dữ liệu:

```
data/raw/visual_genome/          # đã có: image_data.json, scene_graphs.json
data/raw/coco/dataset_coco.json  # Karpathy split + caption
data/raw/MANIFEST.json           # size + sha256 của mọi file thô
data/interim/vg_coco/            # join + normalize trên toàn bộ ảnh có coco_id, chưa chia
data/processed/vg_coco/          # 10K ảnh đã chọn: images, queries, scene_graphs, ảnh
data/splits/vg_coco.json
```

`data/interim/` và ảnh trong `data/processed/vg_coco/images/` thêm vào `.gitignore`. Dữ liệu 40 ảnh cũ (`data/processed/visual_genome/`, `prepare_visual_genome_subset.py`) giữ nguyên tới khi Phase 1 chuyển sang `vg_coco`.

Thư viện thêm vào `requirements.txt`: `scikit-learn`, `pyyaml`, `pytest`, `matplotlib` (`.venv` hiện chưa có `sklearn`).

## Các bước

### 2.1 Nguồn dữ liệu thô

- **Mục tiêu:** đủ file thô, tải lại được bằng một lệnh, phát hiện file hỏng.
- **Input:** URL trong config: 2 file VG (đã có) + `caption_datasets.zip` của Karpathy (chứa `dataset_coco.json`: `cocoid`, `split` ∈ {train, val, test, restval}, caption gốc).
- **Output:** `data/raw/coco/dataset_coco.json`; `data/raw/MANIFEST.json` (`path`, `bytes`, `sha256`, `url`, `fetched_at`).
- **Quy tắc:** file đã có và đúng sha256 → bỏ qua; sai sha256 → báo lỗi, không ghi đè im lặng. Lần tải đầu ghi sha256 vào MANIFEST làm mốc.
- **Xong khi:** chạy hai lần, lần hai không tải gì; sửa 1 byte file → lệnh báo lỗi.
- **Rủi ro:** URL Karpathy có thể đã chết → dùng mirror (Kaggle/HF), ghi nguồn thật vào MANIFEST.

### 2.2 Ghép VG ↔ COCO

- **Mục tiêu:** mỗi ảnh VG có `coco_id` mang đủ metadata, split Karpathy và 5 caption.
- **Input:** `image_data.json`, `dataset_coco.json`.
- **Output:**
  - `data/interim/vg_coco/images.json`: `{vg_image_id, coco_id, url, width, height, karpathy_split, captions: [{caption_id, text}] × 5}`.
  - `data/interim/vg_coco/funnel.json`: số ảnh sau mỗi bộ lọc + số bị loại theo từng lý do.
- **Quy tắc:**
  - Bỏ ảnh `coco_id` rỗng; bỏ ảnh có `coco_id` không có trong file Karpathy (đếm riêng).
  - Ảnh > 5 caption: lấy 5 câu `sentid` nhỏ nhất; < 5 caption: loại (đếm riêng).
  - Caption giữ nguyên văn (`raw`), chỉ `strip()`; không sửa chính tả.
  - Hai ảnh VG cùng `coco_id`: loại cả hai, đếm riêng (tránh một ảnh nằm ở hai split).
- **Xong khi:** funnel bắt đầu đúng 51.498; tổng số bị loại = số đầu − số cuối; test fixture phủ đủ các ca trên.

### 2.3 Chuẩn hoá scene graph theo synset

- **Mục tiêu:** cùng một khái niệm → cùng một nhãn → sau này cùng một đỉnh.
- **Input:** `scene_graphs.json` (739 MB, load vào RAM được) + danh sách id từ 2.2.
- **Output:**
  - `data/interim/vg_coco/scene_graphs.json`: `{vg_image_id, objects: {object_id: {name_raw, synset, label}}, relations: [{subject_id, object_id, predicate_raw, predicate}]}`.
  - `data/interim/vg_coco/normalization_report.json`.
- **Quy tắc:**
  - Object: `synset` = phần tử đầu của `synsets`; `label` = lemma của synset, `_` → khoảng trắng.
  - Object không có synset: giữ, `synset = null`, `label` = tên thô chữ thường; **chưa** quyết định loại (để 2.6).
  - Vị từ: chữ thường, gộp khoảng trắng, bỏ dấu câu hai đầu.
  - Quan hệ trỏ tới object không tồn tại: bỏ, đếm. Quan hệ trùng `(subject label, predicate, object label)` trong cùng ảnh: giữ một, đếm.
- **Report gồm:** số tên thô → số synset (mức gộp); % object có synset; top 50 tên thô không có synset; số vị từ trước/sau chuẩn hoá.
- **Xong khi:** test `synsets.py` cho `bicycle.n.01 → bicycle`, `gym_shoe.n.01 → gym shoe`, `" ON " → "on"`, object thiếu synset; report sinh ra.

### 2.4 Chia train/val/test bằng scikit-learn

- **Mục tiêu:** split cố định, tái lập được, test sạch theo phương án B.
- **Input:** `images.json` + `scene_graphs.json` của interim.
- **Output:** `data/splits/vg_coco.json` = `{seed, config_sha256, sizes, strata_edges, train, val, test}`.
- **Quy tắc:**
  - Ảnh hợp lệ: ≥ `min_relations` quan hệ sau chuẩn hoá (mặc định 1, trong config) và không thuộc `exclude_ids`.
  - Kho test = ảnh hợp lệ có `karpathy_split == "test"`; kho train/val = ảnh hợp lệ còn lại (train + val + restval).
  - Tầng = tứ phân vị số quan hệ/ảnh, mốc tính trên toàn bộ ảnh hợp lệ.
  - Test: `train_test_split(test_pool, train_size=1500, stratify=strata, random_state=seed)`.
  - Train/val: lấy 8.500 ảnh từ kho train/val cùng cách, rồi tách 7.000/1.500 bằng một lần `train_test_split` nữa, cũng phân tầng.
  - Kho test < 1.500 → báo lỗi kèm số thật, không tự giảm.
  - Split dùng một seed cố định; nhiều seed ở Phase 3 là seed huấn luyện, không phải seed chia.
- **Xong khi:** test kiểm (a) không trùng giữa 3 tập, (b) không ảnh Karpathy test nào trong train/val, (c) cùng seed + config → file giống hệt sha256, (d) tỉ lệ các tầng giữa 3 tập lệch ≤ 2 điểm %.

### 2.5 Xuất tập con + tải ảnh

- **Mục tiêu:** 10.000 ảnh đã chọn có đủ nhãn và file ảnh trên đĩa.
- **Input:** split 2.4 + interim.
- **Output:** `data/processed/vg_coco/{images.json, queries.json, scene_graphs.json, images/*.jpg, download_report.json}`. `queries.json`: `{query_id, text, relevant_ids: [vg_image_id], split}`, `query_id = "coco-<caption_id>"`.
- **Quy tắc:** tải có retry; ảnh lỗi ghi vào `download_report.json`. Có ảnh lỗi → thêm id vào `exclude_ids` trong config, chạy lại từ 2.4 (vẫn tái lập được), không vá tay.
- **Xong khi:** đủ 10.000 ảnh mở được bằng Pillow; số truy vấn = 5 × số ảnh; mọi `relevant_ids` có file ảnh.

### 2.6 Duyệt dữ liệu (cổng, user duyệt)

- **Mục tiêu:** hiểu dữ liệu và chốt các ngưỡng còn mở trước khi xây đồ thị.
- **Bắt buộc (user yêu cầu 17/11):** khi đã xuống tập đã chọn (nay 10.000 ảnh), review thủ công trên chính tập này (xem ảnh + caption + bộ ba), không chỉ đọc số thống kê.
- **Input:** output 2.2–2.5.
- **Output:** `reports/data-review-vg-coco.md` + hình trong `reports/figures/data-review/`; trang mẫu 20 ảnh (ảnh, 5 caption, 10 bộ ba đầu) để xem bằng mắt.
- **Nội dung:**
  - Funnel từ 51.498 xuống 10.000.
  - Phân bố số object, số quan hệ mỗi ảnh; so sánh train/val/test.
  - Độ dài caption; caption bất thường (quá ngắn, trùng nhau).
  - Độ phủ synset; top 30 khái niệm và vị từ (ứng viên hub: `wall`, `floor`, `sky`…); đuôi dài (số khái niệm xuất hiện < 5 ảnh).
  - **Độ khớp caption–đồ thị:** % danh từ trong caption có trong nhãn đồ thị của chính ảnh đó. Số này cho biết đồ thị có cơ hội giúp truy hồi hay không.
  - Từ gần nghĩa chưa gộp (kiểu `guy`/`man`), chỉ liệt kê.
- **Quyết định chốt sau duyệt:** loại hay giữ object thiếu synset; ngưỡng tần suất khái niệm; cách xử lý hub; có cần gộp theo cây WordNet sớm không.
- **Xong khi:** user ghi duyệt vào cuối report. Chưa duyệt thì không làm 2.7.

### 2.7 Xây đồ thị tĩnh (chỉ sau 2.6)

Quyết định 2.6 (user chốt theo đề xuất 17/11):

| # | Quyết định |
|---|---|
| D1 | Id đỉnh khái niệm = synset (không có synset → `name:<tên thô chuẩn hoá>`); chữ hiển thị/nhúng = tên thô phổ biến nhất của khái niệm, đếm trên train |
| D2 | Bỏ object có tên thô thuộc danh sách rác/màu/vị trí trong config, **xét tên thô bất kể có synset** (VG gán `he` → helium, `it` → information technology, `an` → associate in nursing); ~3.932 object (1,1%) |
| D3 | Bỏ khái niệm và vị từ có < 5 ảnh train |
| D4 | Giữ hub; tính IDF trên train, lưu kèm khái niệm; dùng hay không chốt ở Phase 1/3 |
| D5 | Chưa gộp từ gần nghĩa (Phase 7) |
| D6 | Chưa thêm thuộc tính VG; đo B4 xong mới quyết định |
| D7 | Caption là truy vấn, **không trộn vào đồ thị ảnh** (trộn thì ảnh test mang sẵn đáp án). Phân tích caption thành đồ thị truy vấn bằng SceneGraphParser là lựa chọn phía truy vấn ở Phase 4; 2.7 không phụ thuộc (user đồng ý 17/11) |

- **Mục tiêu:** từ vựng khái niệm/vị từ cố định + đồ thị từng ảnh trên từ vựng đó, dùng chung cho GraphSAGE (Phase 1/3) và B4 (Phase 4).
- **Input:** `data/processed/vg_coco/{scene_graphs.json, images.json}`, config mục `graph`.
- **Output:** `data/processed/vg_coco/graph/`
  - `concepts.json`: `concept_id`, `key` (synset hoặc `name:<tên>`), `text` (D1), `synset`, `train_images`, `idf`
  - `predicates.json`: `predicate_id`, `text`, `train_images`
  - `image_graphs.json`: mỗi ảnh: danh sách id khái niệm (mỗi khái niệm 1 đỉnh/ảnh) + bộ ba `[chủ thể, vị từ, đối tượng]` theo id; bỏ bộ ba có đầu mút hoặc vị từ ngoài từ vựng
  - `graph_stats.json`: số đỉnh/cạnh, phân bố bậc, tỉ lệ object/quan hệ bị bỏ theo lý do (D2, D3, ngoài từ vựng) và theo split, ảnh còn 0 quan hệ
- **Một định dạng, hai cách dùng:** đồ thị từng ảnh dùng trực tiếp cho code GraphSAGE hiện có; đồ thị chung (ảnh–chứa–khái niệm, khái niệm–vị từ–khái niệm) suy ra lúc load từ cùng file, không lưu riêng.
- **Chống rò rỉ:** từ vựng, chữ hiển thị, IDF, ngưỡng D3 chỉ tính trên ảnh train. Ảnh val/test vẫn giữ đồ thị của chúng nhưng khái niệm ngoài từ vựng train bị bỏ; tỉ lệ bỏ báo theo split.
- **Code:** `src/data/graph_build.py` (hàm thuần), stage `graph` trong `scripts/build_vg_coco.py`, mục `graph` trong `configs/data/vg_coco.yaml` (`min_train_images: 5`, `junk_names: [...]`), test trong `tests/data/test_graph_build.py`. Không sửa code GraphSAGE ở bước này.
- **Xong khi:** (a) chạy 2 lần cùng sha256; (b) test xác nhận từ vựng không đổi khi xoá/đổi ảnh val/test; (c) độ khớp caption ↔ đồ thị đo lại bằng chữ D1 không giảm quá 2 điểm % so với 40,8%; (d) xem tay 3 ảnh (2342774, 2406973, 2390208) trước/sau; (e) `pytest tests/data` qua.

## Success Criteria

- [ ] Mỗi stage chạy riêng được: `python -m scripts.build_vg_coco --stage <tên>`; `--stage all` chạy từ file thô ra `data/processed/vg_coco/`
- [ ] `pytest tests/data` qua hết, không cần mạng
- [ ] Funnel, normalization report, split, download report đều ghi ra đĩa
- [ ] Không ảnh Karpathy test nào trong train/val; split tái lập được theo sha256
- [ ] Report duyệt dữ liệu có xác nhận của user trước khi bắt đầu 2.7

## Tiến độ

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| 2.1 | Xong 17/11 | `data/raw/MANIFEST.json` (3 file, sha256); chạy lần 2 chỉ verify, không tải; URL VG + Karpathy còn sống |
| 2.2 | Xong 17/11, chờ user xem funnel | `data/interim/vg_coco/funnel.json`, `images.json` (50.918 ảnh) |
| 2.3 | Xong 17/11 (sau khi user chốt backfill + chuẩn hoá vị từ) | `data/interim/vg_coco/scene_graphs.json`, `normalization_report.json` |
| 2.4 | Xong 17/11 | `data/splits/vg_coco.json` (sha256 `6f153e48…`, chạy 2 lần trùng); 69 test |
| 2.5 | Xong 17/11 (nâng lên 10K cùng ngày) | `data/processed/vg_coco/` 10.000 ảnh, 50.000 truy vấn (test 7.500); split sha256 `c2095127…`; 86 test |
| 2.6 | Xong 17/11: user đã xem `sample.html`, chốt D1–D7 theo đề xuất | `reports/data-review/index.html` (HTML + sơ đồ archify), `reports/data-review/{stats.json, sample.html, figures/, diagrams/}` |
| 2.7 | Xong 17/11 (code + test + code-review); chờ user quyết tiêu chí (c) | `data/processed/vg_coco/graph/`, `data/splits/vg_coco_graph.yaml` (thông số đồ thị), `reports/data-review/diagrams/graph-build-structure.html`; 108 test |

Kết quả 2.7 (sau code-review): 1.778 khái niệm (1.529 có synset), 493 vị từ từ 7.000 ảnh train; 174.743 đỉnh khái niệm, 127.812 bộ ba trên 10.000 ảnh. Giữ lại object 93,3 / 92,7 / 92,5% và quan hệ 84,8 / 83,8 / 82,5% (train / val / test); trung vị 17–18 khái niệm và 12 bộ ba mỗi ảnh; ảnh không còn bộ ba: 46 / 7 / 10. Mỗi khái niệm có mặt ở trung vị 15 ảnh train (p90 154, lớn nhất 2.218). Hub: `man` 31,7% ảnh train, vị từ `on` 85,9%. Thông số ghi ở `data/splits/vg_coco_graph.yaml` (stage tự sinh, kèm sha256 đầu vào và đầu ra).

Tiêu chí xong: (a) 2 lần chạy trùng sha256 — đạt; (b) test chống rò rỉ — đạt; (d) 3 ảnh xem tay — đạt (`trouser` → "pants", `land` → "ground"); (e) pytest — đạt, 108 test. **(c) không đạt:** độ khớp caption ↔ đồ thị 37,4% (ngẫu nhiên 3,6%), giảm 3,4 điểm so với 40,8%, ngưỡng đặt ra là 2. Tách nguyên nhân: tên rác 0,1; khái niệm hiếm (D3) 1,2; bỏ nhãn lemma 0,5; mỗi khái niệm còn một chữ hiển thị (D1) 1,6. Phần D1 gồm từ đồng nghĩa trong cùng synset (`sofa` → "couch") và tính từ lẫn trong tên thô (`white shirt`), tức thông tin thuộc tính mà D6 để sau. Thử lưu thêm aliases cho mỗi khái niệm: độ khớp lên 44,3% nhưng mốc ngẫu nhiên tăng gấp đôi (8,3%) vì tính từ lan sang mọi ảnh cùng khái niệm → bỏ. Tỉ số đúng ảnh / ngẫu nhiên giữ nguyên (9,8 → 10,3 lần). Chờ user quyết: chấp nhận, hay hạ ngưỡng D3.

Code-review 2.7 (agent độc lập, 17/11): đếm lại độc lập từ dữ liệu gốc khớp hoàn toàn (khái niệm, vị từ, IDF, thứ tự id), không rò rỉ. Đã sửa: (1) test yếu — 5 lỗi gài vào code đều lọt; viết lại test, nay bắt 11/11 lỗi gài; (2) lọc rác theo tên thô để sót khái niệm với chữ hiển thị thừa ("yellow center", "backs", "right side") → khái niệm có tên phổ biến nhất là tên rác bị bỏ nguyên cả khái niệm (49 khái niệm, 167 object); (3) kiểm đầu vào: ảnh trùng, ảnh thiếu split, ảnh thiếu đồ thị → `DataError`; (4) hàm ánh xạ ảnh trả về bộ đếm thay vì sửa tham số; (5) `write_yaml` tách sang `yaml_io.py`; (6) file thông số ghi sha256 của cả 3 file đầu vào. Còn mở: `back` (lưng, lưng ghế) là vật thật nhưng nằm trong danh sách rác; ruff/mypy chưa cài trong `.venv` nên chưa chạy lint/typecheck. Ghi nhận, không sửa ở bước này: bước chuẩn hoá 2.3 gộp quan hệ theo nhãn lemma nên một ít cạnh giữa hai synset khác nhau cùng lemma đã mất trước 2.7.

Kết quả 2.3 sau chỉnh: object có synset 79,2% → **94,1%** (VG 1.419.745, backfill 265.983 từ 7.211 tên, không có 106.591); 36.293 nhãn; vị từ 19.545 dạng, top 50 phủ 85,4%; 751.097 quan hệ. Còn thiếu synset: từ màu (`white`, `black`), từ rác (`a`, `this`, `the`), danh từ ghép (`street light`, `cell phone`), từ mơ hồ bị từ chối backfill (`top`, `orange`) → xử lý ở 2.6.

Kết quả 2.4: 49.381 ảnh hợp lệ (bỏ 1.537 ảnh không còn quan hệ); mốc tầng [9, 14, 20] quan hệ. Tỉ lệ tầng train/val/test lệch tối đa 1,64 điểm %. Kiểm độc lập trên file: không trùng, không ảnh Karpathy test trong train/val, coco_id khác nhau (kiểm trên bản 5K; bản 10K chạy cùng code, cùng seed).

Sửa tiêu chí (d) của 2.4: phân tầng chỉ bảo đảm mỗi split khớp **kho nó được lấy ra**; test ↔ train lệch bao nhiêu là thuộc tính dữ liệu (kho Karpathy test vs phần còn lại), đo và ghi ở trên, không phải test đơn vị.

Kết quả 2.3 lần 1 (50.918 ảnh, 21 s, RAM đỉnh 4,2 GB): 1,79 triệu object, 79,2% có synset; 60.607 tên thô → 39.311 nhãn (6.551 synset); 760.855 quan hệ giữ lại, 338.067 quan hệ trùng nhãn bị gộp; 1.537 ảnh không còn quan hệ; median 14 quan hệ/ảnh (tứ phân vị 9/14/20). Vị từ: 21.457 dạng, top 50 phủ 83,1%.

Phát hiện: (1) object thiếu synset phần lớn là từ phổ biến (`man` 7.817, `window` 4.822…) — VG gán synset không đầy đủ; 78,5% object thiếu synset có cùng tên thô đã được gán synset ở chỗ khác. (2) Vị từ dính mạo từ/trợ động từ: `on a`, `has a`, `of a`, `wearing a`, `are on`.

Funnel 2.2: 108.077 VG → 51.498 có `coco_id` → 50.918 sau khi loại 290 `coco_id` dùng chung (cùng ảnh Flickr, VG lưu hai độ phân giải) → 50.918 có trong Karpathy → 50.918 đủ 5 caption. Theo split Karpathy: train 33.669, restval 13.009, val 2.106, **test 2.134** (≥ 1.500, đủ cho phương án B).

Lệnh: `python -m scripts.build_vg_coco --stage sources|join`; test: `pytest tests/data` (16 test, không cần mạng).

## Câu hỏi mở

- (Đã trả lời) Kho VG ∩ Karpathy test có 2.134 ảnh trước lọc `min_relations`; số hợp lệ cuối cùng biết sau 2.3.
