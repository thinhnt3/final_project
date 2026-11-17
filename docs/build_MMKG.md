# Cách xây MMKG

Tài liệu cho người nhận việc "xây MMKG" ở cấp độ 3 (xem [task.md](../task.md)). Nó chỉ ghi yêu cầu của đề, vài kinh nghiệm thu thập, cách tạo bộ truy vấn theo hướng dẫn của thầy, và dạng file pipeline cần. Chọn miền, nguồn, quy mô và cách dựng đồ thị là quyền của người làm.

## 1. Đề bài yêu cầu gì

- Tự xây một MMKG nhỏ trong miền hẹp, ưu tiên tiếng Việt.
- Quy mô gợi ý: 50–150 thực thể, 300–1.000 triple, 200–500 ảnh.
- Chạy phương pháp tốt nhất của cấp độ 1/2 trên đó.
- Nộp kèm: data card, cách liên kết ảnh với thực thể, phân tích khả năng chuyển giao từ tập công khai sang tập tự xây.
- Cạnh suy từ thống kê phải xây từ tập train.

## 2. Kinh nghiệm thu thập

Rút ra từ một lần thu thập ảnh và tri thức về địa điểm ở Việt Nam trên Wikidata, Wikimedia Commons và Wikipedia.

- **Không dùng SPARQL của Wikidata.** Nó bị giới hạn 1 request/phút; Action API ổn định hơn.
- **Lưu lại mọi phản hồi API**, để chạy lại không gọi mạng và không bị nguồn thay đổi giữa chừng.
- **Chỉ toạ độ thì chưa đủ để biết ở Việt Nam.** Khung toạ độ còn chứa Hải Nam và Quảng Tây; cần thêm thuộc tính quốc gia hoặc bài vi.wikipedia.
- **Ảnh trong bài Wikipedia hay là ảnh của nơi khác.** Chỉ lấy file không xuất hiện ở bài của thực thể khác; file dùng chung gần như luôn là ảnh nhỏ trong hộp điều hướng.
- **Lọc theo tên file** để bỏ bản đồ, logo, sơ đồ, cờ, SVG; bỏ ảnh quá nhỏ.
- **Caption:** ưu tiên structured caption của Commons, sau đó mới tới mô tả file; bỏ caption quá ngắn hoặc chỉ lặp lại tên file.
- **Chia tập theo thực thể**, không theo ảnh, để mọi ảnh của một thực thể nằm cùng một tập. Thực thể có quan hệ bộ phận với nhau (ví dụ các tháp của cùng một khu di tích) nên vào cùng một tập.
- **Giữ license, đường dẫn license và tác giả cho từng ảnh.** Ảnh CC BY-SA khi phát hành lại hoặc demo công khai phải ghi công và giữ điều kiện share-alike.

## 3. Cách tạo bộ truy vấn đánh giá

Phần này ghi lại hướng dẫn của thầy cho câu hỏi "có dataset rồi thì tạo bộ truy vấn để đánh giá như thế nào". Ý chính: không tự nghĩ truy vấn bằng tay, mà sinh truy vấn từ chính các triplet có sẵn trong đồ thị, và lấy mẫu ngẫu nhiên có kiểm soát.

### 3.1 Lấy truy vấn từ triplet có sẵn

Mỗi triplet, ví dụ (Chùa Một Cột, nằm_ở, Hà Nội), biến được thành một truy vấn đã có sẵn đáp án bằng cách che một đầu:

- Ví dụ: "Di tích nào nằm ở Hà Nội?" → đáp án là Chùa Một Cột.
- Đáp án lấy từ đồ thị nên không phải gán nhãn tay.

### 3.2 Bốn tính chất cần đảm bảo

Bốc bừa triplet thì bộ truy vấn dễ lệch, ví dụ toàn rơi vào địa điểm nổi tiếng có nhiều cạnh.

| Tính chất | Nghĩa là |
|---|---|
| Ngẫu nhiên (stochasticity) | Bốc theo seed, không tự tay chọn truy vấn "đẹp" |
| Đa dạng (diversity) | Trải trên nhiều loại quan hệ, nhiều loại thực thể, nhiều vùng của đồ thị |
| Đại diện (representative) | Phân phối của bộ truy vấn giống phân phối của cả đồ thị |
| Độ khó (difficulty) | Có cả truy vấn dễ lẫn khó |

Thầy nói độ khó phải ngồi soi dữ liệu mới đánh giá được nên tốn thời gian; nhóm có thể ưu tiên ba tính chất đầu.

### 3.3 Hai cách lấy mẫu thầy gợi ý

Cả hai chung một ý: chọn một thước đo cấu trúc, rồi lấy mẫu trải đều theo thước đo đó. Thầy nêu chúng như ví dụ; có thể dùng thước đo khác (bậc của đỉnh, loại quan hệ).

**Cách A: chia nhóm theo độ đông đúc của vùng lân cận**

1. Với mỗi triplet, đếm số triplet nằm trong phạm vi k bước quanh nó (ví dụ k = 2).
2. Chia thành các nhóm, ví dụ thưa (0–10), vừa (10–50), dày (trên 50).
3. Trong mỗi nhóm, bốc ngẫu nhiên một số lượng trong khoảng định trước (ví dụ 50–100 truy vấn mỗi nhóm).

Kết quả là bộ truy vấn có cả địa điểm ít thông tin lẫn địa điểm nhiều thông tin, vùng dày không áp đảo.

**Cách B: lấy mẫu theo một chỉ số thống kê của đồ thị**

1. Tính cho mỗi triplet một chỉ số. Thầy ghi "intermediateness"; nhiều khả năng ý là betweenness, tức triplet đó nằm trên bao nhiêu đường đi giữa các đỉnh khác.
2. Xem các giá trị phân bố từ đâu đến đâu.
3. Lấy mẫu đều trên khoảng giá trị đó, để có cả triplet trung tâm lẫn triplet ở rìa.

### 3.4 Áp vào bộ tự xây

Cách A dễ làm nhất:

1. Liệt kê toàn bộ triplet (không tính cạnh ảnh → thực thể, vì đó không phải tri thức về thực thể).
2. Đếm hàng xóm 2 bước của mỗi triplet.
3. Chia 3–4 nhóm.
4. Bốc ngẫu nhiên với seed cố định, số lượng gần bằng nhau ở mỗi nhóm.
5. Biến mỗi triplet thành một truy vấn bằng mẫu câu tiếng Việt cho từng loại quan hệ (che đầu hoặc che đuôi).
6. Trong báo cáo ghi rõ nhóm chia thế nào, seed bao nhiêu, mỗi nhóm bao nhiêu truy vấn. Vậy là đủ chứng minh ba tính chất đầu.

### 3.5 Điểm nhóm cần tự chốt thêm

Thầy không nói tới các điểm dưới đây; chúng phát sinh khi ghép cách trên vào bài toán truy hồi ảnh của nhóm.

- **Ảnh đúng là gì.** Đáp án của truy vấn là một thực thể, còn pipeline trả về ảnh. Hợp lý nhất: ảnh đúng là mọi ảnh của thực thể trả lời.
- **Một truy vấn có nhiều đáp án.** "Di tích nào nằm ở Hà Nội?" đúng với mọi di tích ở Hà Nội. Hoặc tính tất cả là đúng và dùng chỉ số cho nhiều đáp án, hoặc chỉ lấy triplet có đúng một đáp án.
- **Triplet dùng làm truy vấn có còn nằm trong đồ thị lúc chấm không.** Nếu còn, đồ thị chứa sẵn đáp án và phép đo thành tra cứu. Cần quyết định có che triplet đó khỏi đồ thị khi đánh giá hay không.
- **Chỉ lấy triplet của thực thể ở tập test**, để truy vấn không dùng lại thứ mô hình đã thấy khi train.
- **Hai nhóm trực tiếp và gián tiếp.** Cấp độ 2 yêu cầu tối thiểu 20 truy vấn, báo cáo tách hai nhóm. Truy vấn sinh từ một triplet là nhóm gián tiếp (phải đi qua quan hệ); nhóm trực tiếp có thể lấy từ caption của ảnh.
- **Ghi rõ các mẫu câu trong data card**, để người đọc biết truy vấn được sinh thế nào.

## 4. Pipeline cần file gì

Để chạy được embed → train → benchmark như trên tập Visual Genome ∩ COCO, bộ tự xây cần xuất ra:

| File | Nội dung |
|---|---|
| `images.json` | Mỗi ảnh: id là số nguyên, tập (`train` / `val` / `test`) |
| `queries.json` | Mỗi truy vấn: id, văn bản, tập, danh sách ảnh đúng |
| `graph/concepts.json`, `graph/predicates.json` | Từ vựng đỉnh và quan hệ, id là số nguyên, kèm văn bản |
| `graph/image_graphs.json` | Mỗi ảnh: danh sách đỉnh và các bộ ba (chủ thể, quan hệ, đích) |
| File split | Thực thể nào thuộc tập nào, cố định trong repo |

Code đánh giá hiện giả định mỗi truy vấn có đúng một ảnh đúng. Nếu một truy vấn có nhiều ảnh đúng thì phần chỉ số phải sửa theo.
