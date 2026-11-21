**Đối chứng bằng GNNExplainer.** Cách chọn ở trên xếp relation theo mức giảm điểm khi xoá, rồi lại đo bằng phép xoá. Để kiểm tra tín hiệu fidelity không chỉ do cách chọn tự chấm chính nó, relation được chọn lại bằng GNNExplainer (Ying et al., 2019): một mask học trên cạnh, không dùng phép xoá. Truy vấn, mức xoá và nhóm ngẫu nhiên giữ nguyên.

| Mức xoá | Cách chọn | Mất vị trí đầu | Điểm đồ thị giảm | Giữ vị trí đầu khi chỉ giữ nhóm chọn |
|---|---|---|---|---|
| 10% | Bỏ từng relation | 18,23% | 0,0236 | 91,83% |
| 10% | GNNExplainer | 8,52% | 0,0090 | 87,91% |
| 10% | Ngẫu nhiên | 4,66% | 0,0022 | 81,06% |
| 25% | Bỏ từng relation | 25,05% | 0,0327 | 96,59% |
| 25% | GNNExplainer | 12,78% | 0,0153 | 92,33% |
| 25% | Ngẫu nhiên | 6,64% | 0,0034 | 85,52% |

Nhóm của GNNExplainer vượt ngẫu nhiên ở cả ba phép đo nhưng yếu hơn cách bỏ từng relation, vốn được chọn theo đúng đại lượng đang đo. Hai cách chọn trùng nhau 31,4% số relation ở mức 10% và 43,8% ở mức 25%. Mask được khớp trên logit lúc huấn luyện (cosine chia temperature 0,07); với cosine thô, mask là nhiễu và phép xoá cho kết quả ngang ngẫu nhiên.

Đồ thị con của 10 truy vấn, bảng đủ 42 nhãn validity và số theo từng seed: `reports/explanation-quality/index.html` trong repository. Giao thức: `docs/graphsage-design-decisions.md`.
