3 seed nếu không ghi khác; số liệu từ `plans/reports/ablation-261120-0650-relation-suite.md` (file kết quả thô không còn).

| Thí nghiệm | Kết quả |
|---|---|
| Tỉ lệ structure-only dropout 10 / 30 / 50 / 70% | 10% đã đủ để đọc cạnh; từ 50% R@1 caption giảm để đổi lấy confuser acc@1 |
| Trọng số structure-only loss 0,1 / 0,3 / 1 | 0,1 tốt nhất (R@1 53,2 ở 3 seed) |
| Chọn checkpoint theo R@1 + relation | Không cứu được baseline; edge sensitivity trên val ≈ 0 ở mọi epoch |
| Predicate-swap hard negatives | Trọng số 1 làm train mất ổn định (R@1 48,5); 0,1 không tác dụng |
| Same-pair batches (run 22) | Không đọc cạnh; CLIP ảnh vẫn đủ tách các ảnh |
| Train 25 / 50 / 100% (7k ảnh) | Baseline không đọc cạnh ở mọi mức; so0.3 đọc cạnh ở mọi mức, all 28,2 → 29,5 → 30,2 |
| Toàn bộ VG ∩ COCO (45.817 ảnh train) | Baseline R@1 55,8 (+3,7) và **tự đọc cạnh** (30,6 → 29,4 khi masking); so0.3 56,0, mất lợi thế |
| Scene-graph noise mô phỏng | Nhiễu nặng: baseline 48,1, so0.3 45,5, sl0.1 45,6; mô hình đọc cạnh chịu nhiễu kém hơn |
| Caption nhắc tới triplet của ảnh | Khớp chính xác 17,9%; + từ đồng nghĩa 19,3%; + khái niệm cha 22,0% |

**Kết luận.**

1. Cấp độ 1: GraphSAGE nâng CLIP 4,2 điểm R@1; relation có đóng góp nhỏ (bỏ relation −0,56), còn nội dung và hướng của relation chưa có tác dụng rõ.
2. Cấp độ 2: AdaMF-MAT vượt GraphSAGE 3,3 điểm, nhưng gần như toàn bộ nhờ image modality và tên object; adaptive fusion và MAT không đóng góp.
3. Với 7k ảnh train, AdaMF bỏ qua cạnh vì CLIP ảnh đã đủ khớp caption. Ép structure modality học riêng làm mô hình đọc cạnh và đạt R@1 cao nhất 53,1.
4. Khi có nhiều dữ liệu hơn (≈ 6,5 lần), baseline tự đọc cạnh và lợi ích của can thiệp biến mất.
5. Chưa cấu hình nào phân biệt quan hệ A với B tốt hơn baseline hoặc CLIP có ý nghĩa thống kê; đọc cạnh phải trả giá bằng độ bền khi đồ thị nhiễu.

**Chưa làm.** Predicate-swap với trọng số giữa 0,1 và 1; bộ dữ liệu đầy đủ với 5 seed kèm so/sl và nhiễu; nhiễu từ scene-graph generator thật (cần GPU); dataset thứ hai (MKG-W, nhánh riêng).
