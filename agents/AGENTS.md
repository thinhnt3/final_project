# Agent instructions for Project 3

## Mục tiêu

Xây dựng hệ thống text-to-image có CLIP frozen, graph reranking và giải thích bằng đường đi/quan hệ của graph. Reference chính là `references/Project03_Multimodal_XAI (2).pdf`.

## Quy tắc bắt buộc

1. Đọc `plan-task/README.md` và task file liên quan trước khi làm việc.
2. Không viết code vào `src/` cho tới khi các task `P0` và `P1` đã được nghiệm thu theo `plan-task/00-roadmap.md`.
3. Mỗi task phải có input, output, tiêu chí nghiệm thu và trạng thái. Không đánh dấu Done nếu chưa có artifact hoặc log kiểm chứng.
4. Không dùng test query/candidate để chọn alpha, hyperparameter, xây cạnh thống kê hoặc huấn luyện projection.
5. Mọi baseline và phương pháp graph dùng cùng split, cùng candidate pool và cùng seed protocol.
6. Thực nghiệm chính chạy tối thiểu 3 seed; báo cáo mean và standard deviation cho Recall@1/5/10 và MRR.
7. Tách rõ `data/raw`, `data/processed`, `artifacts`, `runs` và `reports`; không commit dữ liệu lớn hoặc secret.
8. Ưu tiên thay đổi nhỏ, có thể xóa/thay thế từng module. Không gộp download, preprocessing, train và evaluation vào một script khó tái sử dụng.

## Cách tiếp nhận task

- Chỉ nhận task có dependency đã hoàn thành.
- Trước khi sửa, kiểm tra file đang tồn tại và giữ nguyên thay đổi của agent khác.
- Sau khi làm, cập nhật trạng thái task, lệnh đã chạy, artifact tạo ra và vấn đề còn lại trong task file.
- Nếu phát hiện sai giả định dữ liệu hoặc leakage, dừng nhánh phụ thuộc và ghi blocker vào task file.

## Thứ tự làm việc

`P0 quyết định phạm vi` → `P1 thiết kế contract/split` → các nhánh `P2` song song → `P3` đồng bộ tích hợp → `P4` chạy thực nghiệm → `P5` báo cáo/demo.

## Chưa được phép làm

- Chưa thêm model mới ngoài phạm vi đã chốt.
- Chưa tối ưu theo test.
- Chưa tuyên bố graph cải thiện nếu chưa có CLIP-only, CLIP+Graph và ablation cạnh.
- Chưa tuyên bố explanation faithful nếu chưa perturb đúng cạnh/đường đi được dùng để tính điểm.

## Nhật ký lịch sử: bản thử ngày 10/11/2026

Bản thử khi đó có 40 ảnh, 40 truy vấn và 1.322 quan hệ. CLIP, chỉ số truy hồi,
FAISS, bộ so khớp từ và GraphSAGE thử nghiệm đã chạy. Bộ xếp hạng chính còn dùng
điểm từ khoá; chưa tích hợp GraphSAGE huấn luyện vào kết quả chính. Các tệp/mô-đun
của bản thử sau đó đã được xoá và thay thế. Không dùng nhật ký này để mô tả hệ
thống hiện tại hoặc mặc nhiên quy nguyên nhân cho số liệu cũ.

## Nhật ký triển khai ngày 17/11/2026

Từ nền `49bd52d`, đã thêm `src/models/ablations.py`, các phép biến đổi đồ thị,
đặt đặc trưng vị từ bằng 0 và bỏ hướng. Các mặc định ngầm của cấu hình cũ được
giữ là đồ thị đầy đủ/hạt giống hoán vị 0. Mốc đầu có 250 kiểm thử qua, Ruff sạch;
chưa có mypy và chưa có bộ nhớ đệm CLIP để chạy dữ liệu thật. Đây là giới hạn của
mốc lịch sử đó, không phải trạng thái hiện tại.

## Bằng chứng chạy thật ngày 18/11/2026

Đã khôi phục 10.000 ảnh, tạo lại bộ nhớ đệm và chạy đủ 15 lượt: bản đầy đủ cộng
bốn biến thể, mỗi loại ba hạt giống 13/37/97. Vector 10.000×512 hữu hạn, chuẩn hoá,
có điểm lưu và mã băm đầu vào nhất quán. R@1 sau kết hợp (%): đầy đủ
48,12±0,23; chỉ đối tượng 47,56±0,16; bỏ vị từ 48,06±0,28; ngẫu nhiên
47,79±0,18; bỏ hướng 47,84±0,15. Không khẳng định bản đầy đủ thắng mọi chỉ số.

Phân tích theo cặp: lợi thế ròng top-1 của đầy đủ so với chỉ đối tượng là
53/33/40 truy vấn, so với bỏ vị từ là 14/25/−27. Trung bình gần nhau không có
nghĩa dự đoán giống nhau. Kiểm tra k=500 sau thực nghiệm, giữ α đã chọn, cho
R@1/R@10/MRR của đầy đủ là 48,1956/86,3156/60,7960%; bỏ vị từ là
48,0622/86,2800/60,6742%. Không dùng kết quả này để thay đánh giá chính thức
hay chọn lại tham số trên tập kiểm thử.

Độ trung thực: 200 truy vấn lấy mẫu trước, có 198/195/196 ca hợp lệ theo mô hình.
Xoá nhóm giải thích làm mất vị trí đầu 17,81% so với ngẫu nhiên 4,31% ở mức yêu
cầu 10%; tương ứng 25,45% so với 6,06% ở mức 25%. Phải ghi rõ ảnh mục tiêu là
prediction ban đầu, số quan hệ xoá thực tế và các ca bị bỏ qua. Cách chọn nhóm
dựa trên ảnh hưởng của phép xoá từng quan hệ; chưa phải bằng chứng độc lập về
ngữ nghĩa. Xoá nhóm/đo hạng không tự loại bỏ được sự phụ thuộc vào cách chọn.

## Kiểm chứng hiện hành

| Yêu cầu | Bằng chứng | Kết luận |
|---|---|---|
| Phép biến đổi giữ đúng đặc tính đã định nghĩa | Kiểm thử bậc, dữ liệu gốc, hạt giống và 15 lượt thật | Đã xác nhận |
| Không tin riêng `result.json` khi tiếp tục chạy | `RunDirectory.validate_finished`, kiểm thử hồi quy và 15 lượt kiểm tra thực tế | Đã xác nhận |
| Chặn lượt chạy/đầu vào cũ hoặc không tương thích | `src/models/run_validation.py`, kiểm thử sai cấu hình, mã băm và tập dữ liệu | Đã xác nhận |
| Xoá nhóm có đối chứng ngẫu nhiên cùng số lượng | `src/evaluation/fidelity.py`, kiểm thử tính tay và dữ liệu thật | Đã xác nhận trong phạm vi phép đo |
| Khoảng bất định giữ cụm mô tả của cùng ảnh | `src/evaluation/ablation_analysis.py`, kiểm thử và 1.500 cụm ảnh | Đã xác nhận với ba mô hình cố định |
| Nhãn vị từ/cạnh thật giúp tổng quát qua lần huấn luyện mới | Khoảng khám phá có chứa 0 | Chưa xác định |
| Loại mô tả/nhiễu chú giải là nguyên nhân tổng thể | Mới có mười trường hợp minh hoạ | Chưa xác định |
| Nguyên nhân chính xác của MRR ngược ở bản thử cũ | Chưa tái hiện bản thử đó | Chưa xác định |

Bộ mã cuối qua 263 kiểm thử, Ruff và mypy sạch. Hai bài thực hành lỗi có chủ
đích ở clone riêng trong `private/bug-lab/` có 264 kiểm thử qua sau sửa, không
được tính thành kiểm thử của nhánh chính. Không coi lỗi mô phỏng là sự cố thật.

## Tài liệu, định dạng và Git

- Tài liệu diễn giải dùng tiếng Việt; giữ nguyên tên hàm, đường dẫn, tham số và tên thư viện. Chú thích trong mã, tên biến và nhật ký chương trình vẫn theo quy ước tiếng Anh đã chốt.
- Tệp Python mới định dạng toàn bộ, tệp cũ chỉ sửa vùng thay đổi. Chia nhóm thao tác, chú thích ngắn và tách dòng lời gọi nhiều tham số; phần nhập mô-đun theo Ruff.
- `runs/` và `private/` bỏ qua đầu ra mới; tệp lịch sử đã theo dõi vẫn được theo dõi. Không tự xoá hoặc ngừng theo dõi kết quả của người dùng.
- Đã chia lịch sử thành tám commit; commit mô phỏng lỗi resume được đánh dấu rõ và sửa ngay sau đó. Nhánh chính cuối cùng đã sửa, không có kiểm thử dự kiến thất bại còn bật.
- Chỉ cập nhật tài liệu đã có trên main. Tài liệu mới local và mọi nội dung `private/` không được commit.
- Hướng dẫn hiện hành: `docs/graphsage-design-decisions.md`; trạng thái nghiệm thu: `plan-task/requirements-checklist.md`; report tương tác local: `private/ablation-guide.html`.

Giai đoạn 5 vẫn còn tập thứ hai, thí nghiệm xử lý đỉnh bậc cao và đánh giá ngữ
nghĩa. Không chỉnh tham số truy hồi sau khi quan sát tập kiểm thử để ép bản đầy
đủ thắng các biến thể.
