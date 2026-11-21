**Cùng ablation trên dataset thứ hai (MKG-W caption subset).** Cùng năm biến thể, cùng 3 seed và cùng giao thức, chạy ngày 21/11 trên tập con của MKG-W dựng theo khuôn VG ∩ COCO (441 mục test, 882 truy vấn). Panel ngay dưới đoạn này là 6 summary run của tập con (23 đến 28, số liệu từ `runs/mkg_w_captions/benchmark/*.json`); hình cuối đặt hai dataset cạnh nhau.

R@1 sau khi xếp hạng lại, %; trong ngoặc là mức tăng so với CLIP:

| Biến thể | VG ∩ COCO (CLIP 43,88) | MKG-W caption subset (CLIP 35,60) |
|---|---|---|
| Đầy đủ | 48,12 ± 0,23 (+4,24) | 43,35 ± 0,54 (+7,75) |
| objects only | 47,56 ± 0,16 (+3,67) | 40,67 ± 1,25 (+5,07) |
| random edges | 47,79 ± 0,18 (+3,91) | 42,67 ± 0,47 (+7,07) |
| drop predicates | 48,06 ± 0,28 (+4,18) | 41,42 ± 0,66 (+5,82) |
| undirected | 47,84 ± 0,15 (+3,96) | 42,25 ± 0,56 (+6,65) |

Bản đầy đủ trừ biến thể, điểm % R@1, kèm khoảng bootstrap 95% theo cặp:

| Biến thể | VG ∩ COCO | MKG-W caption subset |
|---|---|---|
| objects only | 0,56 [0,05; 1,06] | 2,68 [1,06; 4,35] |
| random edges | 0,33 [−0,09; 0,74] | 0,68 [−0,45; 1,81] |
| drop predicates | 0,05 [−0,35; 0,45] | 1,93 [0,38; 3,36] |
| undirected | chưa có khoảng | 1,10 [−0,11; 2,34] |

- Trên tập con, cạnh đóng góp rõ hơn: bỏ hết relation làm mất 2,68 trong 7,75 điểm tăng, và graph-only R@1 tụt từ 28,19% xuống 21,01%.
- Bỏ nhãn predicate cũng làm mất 1,93 điểm, trong khi ở VG ∩ COCO gần như không đổi: ở đây mô hình dùng cả nội dung của relation.
- Xáo cạnh chưa tách được khỏi bản đầy đủ ở cả hai dataset. Phép xáo giữ nguyên tập object, số relation và nhãn predicate, chỉ đổi cặp subject và object.
- Tập test của tập con nhỏ, nên độ lệch chuẩn và khoảng tin cậy rộng hơn VG ∩ COCO.
