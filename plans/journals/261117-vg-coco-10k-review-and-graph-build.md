# 17/11/2026: nâng tập VG ∩ COCO lên 10K, duyệt dữ liệu, xây đồ thị

Nhật ký công việc của Phase 2, bước 2.5–2.7. Đây là lịch sử làm việc. Quyết định hiện hành nằm ở `plans/261117-0037-multimodal-graph-retrieval-roadmap/phase-02-xay-do-thi-tu-dong.md`.

## Đã làm

- **Nâng tập con từ 5.000 lên 10.000 ảnh** (train 7.000 / val 1.500 / test 1.500, 50.000 caption truy vấn). Test vẫn chỉ lấy từ Karpathy test. Tải đủ 10.000 ảnh, 0 lỗi.
- **Duyệt dữ liệu (2.6):** báo cáo chuyển từ markdown sang `reports/data-review/index.html`, kèm ba sơ đồ archify (pipeline, cách tính độ khớp, cấu trúc bước 2.7). User đã xem 24 ảnh mẫu và chốt D1–D7.
- **Xây đồ thị (2.7):** `src/data/graph_build.py`, stage `--stage graph`. Từ vựng 1.778 khái niệm và 493 vị từ, tính chỉ từ ảnh train; 174.743 đỉnh khái niệm và 127.812 bộ ba trên 10.000 ảnh. Thông số ghi ở `data/splits/vg_coco_graph.yaml`.
- Code và dữ liệu 2.7 nằm trên nhánh `feat/vg-coco-graph-build` (commit `114b797`, `db08b75`), đã push, chưa gộp vào `main`.

## Quyết định

| # | Quyết định |
|---|---|
| D1 | Id đỉnh = synset; chữ hiển thị = tên thô phổ biến nhất trong train (`trouser` → "pants") |
| D2 | Bỏ object có tên rác, chỉ màu, chỉ vị trí; danh sách giữ nguyên dù `back`, `side`, `top` cũng là vật thật |
| D3 | Bỏ khái niệm và vị từ dưới 5 ảnh train |
| D4 | Giữ hub, lưu IDF; dùng hay không quyết ở Phase 1/3 |
| D5 | Chưa gộp từ gần nghĩa (Phase 7) |
| D6 | Chưa thêm thuộc tính VG; đo B4 rồi quyết |
| D7 | Caption là truy vấn, không trộn vào đồ thị ảnh; SceneGraphParser là lựa chọn phía truy vấn ở Phase 4 |

User chấp nhận độ khớp caption ↔ đồ thị 37,4% sau khi xây đồ thị, dù tiêu chí đặt ra là không giảm quá 2 điểm so với 40,8%.

## Điều học được

- **"Cận trên" là chữ dùng sai.** Độ khớp từ vựng 40,8% không phải recall và không chặn trên recall. Nó chỉ kiểm điều kiện cần: đồ thị đúng ảnh khớp caption gấp khoảng 10 lần đồ thị ngẫu nhiên. Con số gắn với recall phải lấy từ baseline B4.
- **Đồ thị nắm "có vật gì", không nắm "đang làm gì, màu gì".** 60% từ nội dung của caption không khớp chủ yếu là hành động (`sitting`, `standing`), màu và kích thước (`white`, `large`), từ chỉ cảnh (`street`, `room`). Truy vấn chỉ khác nhau ở hành động hoặc màu sẽ không được đồ thị giúp.
- **Aliases làm hỏng phép đo.** Lưu thêm tên gọi khác cho mỗi khái niệm đẩy độ khớp lên 44,3%, nhưng mốc ngẫu nhiên tăng gấp đôi, vì `white` trong `white shirt` lan sang mọi ảnh có áo. Đã bỏ. Luôn nhìn cả mốc ngẫu nhiên, không chỉ con số chính.
- **Review độc lập bắt được thứ tự kiểm không bắt được.** Agent gài 5 lỗi vào code và cả 5 lọt qua bộ test đầu tiên; một test không thể fail. Sau khi viết lại, bộ test bắt 11/11 lỗi gài. Lỗi gài thứ 12 không bị bắt hoá ra chỉ ra một đoạn code thừa.
- **Lọc theo tên thô nhưng gộp theo synset thì để sót.** `center` bị bỏ nhưng khái niệm `center.n.01` còn sống với chữ hiển thị "yellow center". Sửa: khái niệm có tên phổ biến nhất là tên rác thì bỏ cả khái niệm (49 khái niệm, 167 object).
- **Lỗi SSL khi tải ảnh là do môi trường:** Python của uv không tìm thấy bộ chứng chỉ CA. Chạy với `SSL_CERT_FILE` trỏ tới `certifi/cacert.pem`.

## Còn mở

- Nhánh `feat/vg-coco-graph-build` chưa gộp vào `main`. Toàn bộ code pipeline (`src/core`, `src/data/config.py`, `graph_build.py`…) và test chỉ có trên nhánh feat; trên `main`, `scripts/build_vg_coco.py` và `scripts/review_vg_coco.py` (bị `.gitignore` bỏ qua) không import được cho tới khi gộp.
- Chưa chạy lint và typecheck: `.venv` chưa cài ruff, mypy.
- Bước chuẩn hoá 2.3 gộp quan hệ theo nhãn lemma, nên một ít cạnh giữa hai synset khác nhau cùng lemma đã mất trước bước 2.7.
- Việc tiếp theo: Phase 1, nối embedding GraphSAGE vào điểm đồ thị trên dữ liệu mới; B4 ở Phase 4 để có R@K của riêng đồ thị.
