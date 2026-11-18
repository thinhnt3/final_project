---
title: "Phase 5: Tập công khai thứ hai, ablation, fidelity"
status: in-progress
---

# Phase 5: Tập công khai thứ hai, ablation, fidelity

## Trạng thái hiện tại — 18/11/2026

**Đã chạy thí nghiệm loại bỏ thành phần trên VG ∩ COCO và đo độ trung thực bằng
can thiệp lên mô hình cố định. Chưa hoàn tất toàn bộ giai đoạn 5.** Còn tập công
khai thứ hai, thí nghiệm xử lý đỉnh bậc cao và các giới hạn kiểm chứng bên dưới.

Hướng dẫn chạy: `docs/graphsage-design-decisions.md`, mục “Ablation và fidelity”.
Checklist tổng: `plan-task/requirements-checklist.md`.

## Hai việc trong checklist cũ đã làm đến đâu?

Các dòng cũ nhắc đến bản thử 40 ảnh và `src/evaluation/ablations.py` đã bị xoá.
Chúng không còn mô tả đúng bộ mã hiện tại. Trạng thái chính xác là:

- [x] **Kiểm tra lại trên mô tả COCO:** đã dùng 10.000 ảnh, mô tả độc lập từ COCO và GraphSAGE được huấn luyện thật; không còn lấy truy vấn bằng cách ghép chữ từ quan hệ của đồ thị.
- [x] **Thay cách giả lập cạnh ngẫu nhiên:** `src/models/ablations.py` đã tách `drop_predicates` và `random_edges`. Bản thứ nhất đặt đặc trưng vị từ bằng 0; bản thứ hai hoán vị riêng các đầu chủ thể và tân ngữ, giữ bậc vào/ra, số quan hệ và phân bố vị từ. Không dùng nhãn giả `random_relation` để gọi là cạnh ngẫu nhiên.
- [ ] **Tái hiện và xác nhận nguyên nhân chính xác của số cũ** “bỏ hết cạnh MRR 0,495 > đầy đủ 0,213”: chưa làm. Đây là kết quả của bộ dữ liệu và bộ chấm điểm bản thử cũ; chưa đủ bằng chứng quy lỗi riêng cho cách tạo truy vấn hay ghép từ khoá. Việc triển khai lại không đồng nghĩa đã chứng minh nguyên nhân lịch sử này.

Hai việc triển khai/kiểm tra lại đã xong; mục điều tra nguyên nhân lịch sử vẫn mở
và không phải điều kiện chặn chạy bộ mã mới. Không sử dụng số của bản thử cũ để
kết luận về GraphSAGE hiện tại.

## Mục tiêu và đầu vào

Kiểm tra phần cải thiện đến từ danh sách đối tượng, nhãn vị từ hay cấu trúc quan
hệ; đánh giá lời giải thích có phản ánh ảnh hưởng đến dự đoán hay không. Kết quả
cần được báo cáo trung thực, kể cả khi bản đầy đủ không hơn mọi biến thể.

Đầu vào: dữ liệu/đồ thị VG ∩ COCO đã chia tập; bộ nhớ đệm vector CLIP đóng băng;
cấu hình và điểm lưu mô hình của ba hạt giống huấn luyện 13, 37, 97. Tập thứ hai
dự kiến là FB15k-237-IMG, chưa triển khai. Mọi lựa chọn α/k và điểm lưu tốt nhất
đều dựa trên tập xác thực, không chọn lại bằng tập kiểm thử.

## Thí nghiệm loại bỏ thành phần

| Mục | Biến thể | Trạng thái |
|---|---|---|
| A1 | Đầy đủ: đối tượng và quan hệ | Đã huấn luyện, xuất vector và đánh giá đủ ba hạt giống |
| A2 | Chỉ đối tượng: bỏ các bộ ba và đỉnh quan hệ, giữ cạnh ảnh → đối tượng | Đã chạy đủ ba hạt giống |
| A3 | Bỏ đặc trưng vị từ: giữ đỉnh quan hệ, cạnh và hướng | Đã chạy đủ ba hạt giống |
| A4 | Đầu mút ngẫu nhiên, giữ bậc vào/ra | Đã chạy đủ ba hạt giống; cho phép bộ ba trùng và quan hệ tự nối |
| A5 | Không dùng mạng đồ thị: so khớp từ có trọng số | Đã có phương pháp đối chứng trên main |
| A6 | Quét α và k trên tập xác thực | Đã dùng để chọn cấu hình; chưa có biểu đồ phân tích độ nhạy đầy đủ |
| A7 | Có/không xử lý đỉnh bậc cao | Chưa làm |
| Bổ sung | Bỏ hướng bằng lớp không hướng sẵn có | Đã chạy đủ ba hạt giống; số tham số ít hơn bản đầy đủ |

Cấu hình: `configs/retrieval/ablations/*.yaml`. Không sửa đè tệp chú giải gốc.
Thí nghiệm thêm cạnh ConceptNet thuộc giai đoạn 7.

## Kết quả đã có trên VG ∩ COCO

Tập kiểm thử gồm 1.500 ảnh và 7.500 truy vấn. Đơn vị %, trung bình ± độ lệch
chuẩn qua ba lượt huấn luyện; α/k chọn riêng trên tập xác thực.

| Biến thể | R@1 sau kết hợp với CLIP |
|---|---:|
| Đầy đủ | 48,12 ± 0,23 |
| Chỉ đối tượng | 47,56 ± 0,16 |
| Bỏ vị từ | 48,06 ± 0,28 |
| Đầu mút ngẫu nhiên | 47,79 ± 0,18 |
| Bỏ hướng | 47,84 ± 0,15 |

Bản đầy đủ không thắng trên mọi chỉ số. Hai lượt đầy đủ chọn k=10, làm R@10 giữ
mức CLIP. Kiểm tra bổ sung với cùng k=500, giữ α cũ, giúp giải thích hiện tượng;
đây là phân tích sau thực nghiệm, không thay kết quả chính thức hoặc chọn lại k.

Lấy mẫu lại theo cụm ảnh 2.000 lần, giữ ba mô hình cố định, cho khoảng 95% của
chênh lệch R@1: đầy đủ − chỉ đối tượng [0,05; 1,06] điểm phần trăm; đầy đủ − bỏ
vị từ [−0,35; 0,45]; đầy đủ − ngẫu nhiên [−0,09; 0,74]. Các khoảng mang tính
khám phá, chưa hiệu chỉnh đa so sánh và không đo bất định qua các lần huấn luyện
mới. Chưa chứng minh được nhãn vị từ/cạnh thật là nguyên nhân chính của cải thiện.

Kiểm tra phép ngẫu nhiên: 9.491/10.000 đồ thị đổi tập bộ ba, 96,3% vị trí bộ ba
thay đổi; bậc vào/ra và phân bố vị từ được giữ nguyên.

## Độ trung thực của lời giải thích bằng can thiệp

- [x] Chọn trước 200 truy vấn kiểm thử, dùng đủ ba mô hình, không huấn luyện lại.
- [x] Xoá đồng thời nhóm quan hệ giải thích, so với 20 nhóm ngẫu nhiên cùng số lượng.
- [x] Giữ CLIP, α/k, tập ứng viên và điểm của mọi ảnh đối thủ; chỉ đổi đồ thị của ảnh được dự đoán đứng đầu ban đầu.
- [x] Đo giảm điểm đồ thị có dấu, tăng thứ hạng và mất vị trí đầu; giữ cả dự đoán đúng lẫn sai.
- [x] Ghi tỷ lệ xoá thực tế và lý do bỏ qua đồ thị có ít hơn ba quan hệ.
- [ ] Nghiệm thu validity: có form audit và exporter không giới hạn 20 query; còn thu nhãn người và đối chiếu máy, không coi lượt AI thử nghiệm là human benchmark.
- [x] Đã chạy relation sparsity và keep-only hai budget, ba seed; không tuyên bố explanation tối thiểu.
- [ ] So sánh đồ thị do người gán với đồ thị do mô hình sinh: tuỳ chọn, chưa làm.

Có 198/195/196 truy vấn hợp lệ theo hạt giống 13/37/97. Ở mức yêu cầu xoá 10%,
tỷ lệ mất vị trí đầu trung bình qua ba mô hình là 17,81% so với ngẫu nhiên 4,31%;
ở mức 25% là 25,45% so với 6,06%. Vì xoá tối thiểu hai quan hệ và chừa một quan
hệ, tỷ lệ thực tế có thể khác mức yêu cầu ở đồ thị nhỏ.

**Giới hạn:** nhóm giải thích được chọn từ ảnh hưởng của phép xoá từng quan hệ.
Phép xoá nhóm và đo hạng vẫn phụ thuộc cách chọn đó; đây là bằng chứng can thiệp,
không phải chứng minh độc lập rằng lời giải thích đúng về ngữ nghĩa. Ảnh mục tiêu
là ảnh mô hình dự đoán, không bắt buộc là ảnh đúng trong dữ liệu.

Sparsity trung bình
79,76%/69,83% ở budget yêu cầu 10%/25%; keep-only giữ top-1 94,23%/98,81%
so với random 81,52%/85,82%. Validity do trợ lý xem ảnh và chấm sơ bộ: 17/42
relation được chấm đúng ảnh và liên quan query, 23 không đạt, 2 chưa rõ.
Code chỉ tổng hợp nhãn JSON; chưa có pipeline vision judge tự động hoặc người
chấm độc lập. Rubric hai chiều là đề xuất triển khai, không phải công thức đề
ấn định. Mẫu 20 query cân bằng của một seed không đại diện toàn benchmark.
Protocol audit v2 mới ở bước chuẩn bị, chưa có judge run, nhãn người hoặc
kết quả agreement. CLI tổng hợp nhãn hiện chỉ hỗ trợ schema v1.
Form tái sử dụng: `tools/validity-review/`; lệnh `python -m scripts.prepare_validity_review --all`
xuất mọi query hợp lệ của seed/budget yêu cầu. Mẫu AI 20 query/42 relation ở trên
là lượt thử nghiệm đã thực hiện, không phải giới hạn form hoặc kết luận cho cả 200 query.
Lệnh: `python -m scripts.evaluate_explanation_quality`;
artifact: `runs/explanation-quality/`; report local: `private/reports/xai-quality/index.html`.

## Tập công khai thứ hai: FB15k-237-IMG

- [ ] Chốt dạng truy vấn: mô tả thực thể hay bộ ba khuyết đuôi.
- [ ] Viết bộ đọc và xây đồ thị theo ranh giới mô-đun hiện có.
- [ ] Chạy CLIP, phương pháp chính, A1/A2/A4 trên tập này.
- [ ] Tổng hợp trung bình/độ lệch chuẩn qua các hạt giống tương ứng.

## Đầu ra và cách kiểm tra

- `runs/benchmark/graphsage-*.json`: đánh giá từng biến thể.
- `runs/graphsage/<tên-lượt>/seed<n>/`: cấu hình, lịch sử và kết quả từng lượt; điểm lưu/vector nằm local.
- `runs/analysis/ablation.json`: thắng/thua theo cặp, khoảng bất định và mười trường hợp minh hoạ.
- `runs/fidelity/full-joint-test.json`: dấu vết can thiệp và tổng hợp theo hạt giống.
- `private/ablation-guide.html`: tài liệu tương tác local, không commit.

Đã kiểm tra đủ 15 lượt: điểm lưu và vector hữu hạn, chuẩn hoá 10.000×512; mã băm
đầu vào nhất quán. Kiểm tra tiếp tục chạy bỏ qua đúng 15 lượt hoàn chỉnh. Chạy
lại phép đo độ trung thực cho kết quả giống hoàn toàn. Bộ mã cuối có 263 kiểm thử
qua, Ruff và mypy sạch; các bản ghi 250/262 kiểm thử trước đó là mốc lịch sử.

`runs/` và `private/` bỏ qua đầu ra mới trong Git. Tệp lịch sử đã được theo dõi
vẫn được theo dõi. Đã commit mã và JSON/YAML kết quả nhỏ theo cấu trúc có sẵn;
không commit điểm lưu, bộ nhớ đệm, dấu vết thô hoặc tài liệu riêng tư.

## Điều kiện nghiệm thu và việc còn lại

- [x] Các biến thể cốt lõi trên tập thứ nhất có kết quả đủ ba lượt huấn luyện.
- [x] Đã báo cáo so sánh can thiệp với nhóm ngẫu nhiên, có giới hạn diễn giải.
- [ ] Lặp lại các thí nghiệm cần thiết trên tập công khai thứ hai.
- [ ] Thí nghiệm xử lý đỉnh bậc cao và biểu đồ độ nhạy đầy đủ.
- [ ] Đối chứng bỏ hướng có số tham số tương đương bản đầy đủ.
- [ ] Phân tích thủ công hoàn chỉnh các trường hợp thành công/thất bại và đánh giá ngữ nghĩa.
- [ ] Tái hiện nguyên nhân kết quả ngược của bản thử cũ nếu tiếp tục yêu cầu này.

Không coi “mọi biến thể phải kém bản đầy đủ” là điều kiện để sửa số liệu hoặc đổi
cấu hình sau khi thấy tập kiểm thử. Giai đoạn này vẫn đang thực hiện.
