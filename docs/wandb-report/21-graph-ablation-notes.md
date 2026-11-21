**Nhận xét.**

| Run | R@1 | Graph-only R@1 |
|---|---|---|
| 03 · GraphSAGE đầy đủ | 48,12 | 27,54 |
| 05 · objects only | 47,55 | 26,15 |
| 06 · drop predicates | 48,06 | 27,35 |
| 07 · random edges | 47,79 | 27,38 |
| 08 · undirected | 47,84 | 28,16 |

- Bỏ hẳn relation làm R@1 giảm 0,56 điểm (khoảng bootstrap 95% [0,05; 1,06]): relation có đóng góp nhưng nhỏ.
- Bỏ nhãn predicate (−0,05, [−0,35; 0,45]), xáo cạnh (−0,33, [−0,09; 0,74]) hay bỏ hướng đều **không** khác bản đầy đủ có ý nghĩa thống kê. Mô hình dùng *sự có mặt* của relation nhiều hơn là *nội dung* của nó.
- **Fidelity** (200 query, xoá nhóm relation mà lời giải thích chọn, so với nhóm ngẫu nhiên cùng số lượng): xoá nhóm được chọn làm ảnh mất hạng 1 ở 17,81% query so với 4,31% khi xoá ngẫu nhiên (mức 10%), và 25,45% so với 6,06% (mức 25%). Các relation được lời giải thích chọn ảnh hưởng tới kết quả nhiều hơn hẳn nhóm ngẫu nhiên; đây là kiểm tra bằng can thiệp, chưa chứng minh lời giải thích đúng về ngữ nghĩa.
- Chi tiết giao thức: `docs/graphsage-design-decisions.md`, mục Ablation và fidelity.
