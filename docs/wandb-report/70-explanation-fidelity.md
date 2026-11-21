Đề yêu cầu đánh giá giải thích định lượng: **fidelity** (bỏ cạnh được dùng làm giải thích thì điểm hoặc thứ hạng phải đổi theo hướng hợp lý), **validity** và **sparsity**, với bằng chứng cụ thể cho tối thiểu 10 truy vấn.

Giải thích của một cặp truy vấn và ảnh là các relation làm điểm đồ thị giảm nhiều nhất khi bỏ riêng từng cái. Lượt ngày 21/11: 200 truy vấn test, ba mô hình GraphSAGE đầy đủ (seed 13, 37, 97) huấn luyện lại từ `configs/retrieval/ablations/full.yaml`; đánh giá được 197/195/195 truy vấn, phần còn lại có ít hơn ba relation. Ảnh được giải thích là ảnh đứng đầu sau khi trộn CLIP với điểm đồ thị, dù đúng hay sai.

**Fidelity.** Xoá đồng thời cả nhóm relation giải thích khỏi đồ thị của ảnh, giữ nguyên CLIP và điểm của mọi ảnh khác, rồi so với 20 nhóm ngẫu nhiên cùng số relation.

| Mức xoá | Mất vị trí đầu | Ngẫu nhiên | Điểm đồ thị giảm | Ngẫu nhiên |
|---|---|---|---|---|
| 10% | 18,23% | 4,66% | 0,0236 | 0,0022 |
| 25% | 25,05% | 6,64% | 0,0327 | 0,0034 |

Xoá nhóm giải thích làm ảnh mất vị trí đầu nhiều gấp khoảng 4 lần xoá ngẫu nhiên, và điểm đồ thị giảm nhiều gấp khoảng 10 lần. Theo từng truy vấn (587 cặp truy vấn và mô hình, mức 25%), nhóm giải thích làm giảm điểm nhiều hơn trung bình ngẫu nhiên ở 99,5% số cặp. Số này lệch nhẹ so với lượt 18/11 ở mục 2 (17,81% và 25,45%) vì là mô hình huấn luyện lại.
