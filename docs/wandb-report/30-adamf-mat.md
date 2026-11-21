**AdaMF-MAT trong bài toán này.** Thực thể là một ảnh, quan hệ duy nhất là "is described by", tail là caption. Ba modality, mỗi cái qua một Linear về 256 chiều:

- **structure:** GraphSAGE 3 layer có hướng ở trên, image node learned (không mang CLIP ảnh).
- **image:** vector CLIP ViT-B/32 của ảnh (frozen).
- **text:** trung bình vector CLIP text của tên các object trong ảnh, không có predicate.

**Adaptive fusion:** `u_m = aᵀ·tanh(e_m)`, `w = softmax(u)`, `e = Σ w_m·e_m`, tính riêng cho từng ảnh; `e` được chiếu về không gian CLIP và train bằng InfoNCE. **MAT:** một generator sinh vector image và text giả từ vector structure cộng nhiễu; encoder phải chấm cặp thật cao hơn cặp giả.

**Linear adapter** là AdaMF bỏ cả structure lẫn text: một Linear trên vector CLIP ảnh, không dùng đồ thị. Nó là mốc để tách phần tăng nhờ "học lại vector ảnh" khỏi phần tăng nhờ đồ thị.
