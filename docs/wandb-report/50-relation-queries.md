Caption COCO hiếm khi nhắc đúng quan hệ (17,9% caption train chứa một triplet của ảnh), nên R@1 theo caption không cho biết mô hình có dùng cạnh không. Vì vậy dựng thêm bộ **relation queries**:

- **Relation query:** câu sinh từ một triplet chỉ có ở đúng một ảnh test, ví dụ `man riding horse`.
- **Confuser:** ảnh test khác có **đủ cả hai object** (`man` và `horse`) nhưng **không có quan hệ** được hỏi, ví dụ người đứng cạnh ngựa.
- **Confuser acc@1:** chỉ xếp hạng ảnh đúng cùng các confuser của nó, đếm tỉ lệ ảnh đúng đứng đầu. Ngẫu nhiên: 21,4%.
- Vì sao cần confuser: trên cả 1.500 ảnh, mô hình chỉ cần tìm ảnh có người và ngựa; confuser đều có người và ngựa, nên cách duy nhất để chọn đúng là đọc quan hệ.
- **all** (1.408 query) và **contrast** (734 query, chỉ giữ confuser có một quan hệ *khác* giữa đúng hai object đó, ví dụ `man next to horse`; ngẫu nhiên 31,4%).
- **Edge masking:** xoá cạnh được hỏi khỏi đồ thị ảnh đúng rồi chấm lại. Tụt nghĩa là mô hình có đọc cạnh đó.

Các biểu đồ dưới: confuser acc@1 trên graph-only score, 5 seed. CLIP không có đồ thị nên không có cột edge masking.
