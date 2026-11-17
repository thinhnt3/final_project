# Duyệt dữ liệu VG ∩ COCO (Phase 2, bước 2.6)

Báo cáo chính là trang HTML (có sơ đồ động): mở `reports/data-review/index.html` bằng trình duyệt.

- Sơ đồ pipeline: `reports/data-review/diagrams/vg-coco-pipeline.html` (nguồn: `vg-coco-pipeline.dataflow.json`)
- Sơ đồ cách tính độ khớp caption ↔ đồ thị: `reports/data-review/diagrams/caption-graph-coverage.html`
- 24 ảnh mẫu để review thủ công: `reports/data-review/sample.html`
- Số liệu máy: `reports/data-review/stats.json`; tái tạo bằng `python -m scripts.review_vg_coco`
- Sơ đồ dựng bằng `/ak:archify` từ file JSON cùng thư mục: sửa JSON rồi chạy `deliver` để tạo lại

Bản hiện tại: 10.000 ảnh (train 7.000 / val 1.500 / test 1.500), split sha256 `c2095127…`.

## Xác nhận của user

- [ ] Đã xem `sample.html`, ghi nhận thêm (nếu có): …
- [x] Chốt D1–D6: theo đề xuất (17/11)
- [ ] Đồng ý bắt đầu 2.7
