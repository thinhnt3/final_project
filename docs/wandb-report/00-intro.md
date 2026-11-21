**Bài toán.** Truy hồi ảnh bằng văn bản (text→image retrieval): cho một caption COCO, tìm đúng ảnh trong 1.500 ảnh test. Dữ liệu là phần giao Visual Genome ∩ COCO: ảnh và scene graph (object, relation) lấy từ Visual Genome, query là caption COCO của cùng ảnh. Chia 7.000 / 1.500 / 1.500 ảnh (train / val / test), mỗi ảnh 5 caption.

**Pipeline chung.** CLIP ViT-B/32 (frozen) lấy top-k ảnh, rồi rerank bằng điểm của mô hình đồ thị:

`score = α · cosine_CLIP + (1 − α) · score_model`

α và k chọn trên validation, test chỉ chấm sau khi đã chốt. Độ đo chính: **R@1** (tỉ lệ caption có ảnh đúng ở hạng 1), thêm R@5, R@10, MRR. **Graph-only** là điểm của riêng mô hình, không cộng CLIP.

**Câu hỏi của đồ án.**
1. Cấp độ 1: GraphSAGE trên scene graph có cải thiện CLIP không, và cải thiện nhờ cạnh quan hệ hay không (ablation, fidelity)?
2. Cấp độ 2: tái hiện AdaMF-MAT (adaptive multimodal fusion + modality adversarial training); nó hơn GraphSAGE ở đâu, và mô hình có thật sự đọc cạnh không?

**Các run trong report.** Mỗi cấu hình có một *summary run* (`NN · tên`, số liệu test trung bình qua seed, lấy từ `runs/benchmark/*.json`) và các *train run* (`NN · tên · seed S`, đường học từng epoch lấy từ `history.json`). Seed 13 / 37 / 97; các cấu hình ghi 5 seed thêm 211 / 307. Mọi run tải lên từ file kết quả local, không train lại. Code: repo `master-hcmus-graphML`, nhánh `main`.
