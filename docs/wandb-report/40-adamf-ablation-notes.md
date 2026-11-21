**Nhận xét.** R@1 sau rerank, 3 seed (17 và 19 là 5 seed):

| Nhóm | Run | R@1 |
|---|---|---|
| Fusion và MAT | 10 · AdaMF-MAT (full) / 11 · w/o MAT | 51,42 / 51,63 |
| | 12 · mean fusion + MAT / 13 · mean fusion w/o MAT | 52,16 / 51,69 |
| Bỏ từng modality (so với 11) | 14 · w/o structure / 15 · w/o text / 16 · w/o image | 51,92 / 50,54 / 47,75 |
| | 09 · image only (linear adapter) | 49,76 |
| Chuẩn hoá và ép structure | 17 · L2 / 18 · mean + L2 | 51,93 / 52,03 |
| | 19 · + structure-only dropout 0.3 / 20 · + structure-only loss 0.1 | 52,74 / 53,1 |
| | 22 · + same-pair hard negatives | 51,83 |

- **Adaptive fusion và MAT đều không giúp:** bật hay tắt, R@1 nằm trong 51,4–52,2.
- **Image là modality quyết định:** bỏ đi tụt 3,9 điểm; thêm text vào image được +2,2 (49,76 → 51,92); bỏ structure không đổi.
- **Vì sao structure bị bỏ qua:** ở bản gốc không chuẩn hoá, trọng số structure chỉ ≈ 0,0002 nhưng vector dài 84–2.011 lần, phần đóng góp thật ≈ 7%. Chuẩn hoá L2 từng modality trước khi trộn làm phần đóng góp bằng đúng trọng số, nhưng một mình nó không đủ để mô hình dùng structure.
- **Structure-only dropout 0.3:** ở 30% số ảnh trong mỗi batch, image và text bị che (trọng số = 0) nên fused vector chỉ còn structure. **Structure-only loss 0.1:** cộng thêm `0,1 · InfoNCE(f(e_s))` cho mọi ảnh. Cả hai buộc GraphSAGE tự khớp caption.
