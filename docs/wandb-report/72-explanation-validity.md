**Validity: nhãn do AI chấm, chưa có người chấm.** Validity hỏi relation trong giải thích có đúng với ảnh và có liên quan tới truy vấn không. Một trợ lý AI xem ảnh và chấm một lượt 20 truy vấn với 42 relation (seed 13, mức 10%, 10 ảnh trả về đúng và 10 sai) ngày 21/11.

- Đúng với ảnh: 38/42, 4 chưa rõ, không có cái nào sai.
- Đọc đúng câu chữ của rubric (relation chạm tới một vật thể, hành động hoặc ràng buộc được hỏi): 39/42 liên quan, 35/42 đạt cả hai tiêu chí (83,3%).
- Đọc chặt (relation phải nói đúng quan hệ hoặc hành động được hỏi, không chỉ là có mặt vật thể): 15/42 liên quan, 14/42 đạt cả hai (33,3%). Ví dụ `plane — has → nose` đúng ảnh và chạm tới máy bay, nhưng không nói gì về "đang bay trên trời".

Giới hạn: một người chấm là AI, một lượt, không có đối chiếu với người; mẫu cân bằng đúng/sai nên không đại diện cho cả tập test. Khoảng cách giữa hai cách đọc cho thấy cần chốt tiêu chí trước khi nhờ người chấm.
