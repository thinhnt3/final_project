Mỗi ablation train lại GraphSAGE (learned image node) trên đồ thị đã sửa, 3 seed, rồi chấm như bản đầy đủ:

- **objects only:** bỏ mọi relation, chỉ giữ object node và cạnh image → object.
- **drop predicates:** giữ relation node và topology, nhưng đặt đặc trưng predicate bằng 0.
- **random edges:** hoán vị subject và object của các relation trong từng ảnh (giữ bậc, số relation, phân bố predicate).
- **undirected:** một `SAGEConv` trên cạnh hai chiều, không phân biệt hướng.
