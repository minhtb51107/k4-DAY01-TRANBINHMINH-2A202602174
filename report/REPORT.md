# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.x / PyTorch 2.x / Ultralytics 8.4.145[cite: 2]

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`[cite: 2, 3, 4]

**Thay đổi so với notebook nguồn:** Không

> **Lưu ý:** Báo cáo tuân thủ nguyên tắc không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân.

## 1. Phân loại ảnh – prediction cấp ảnh

*Nguồn evidence: `classification_predictions.json`, sample `traffic`.*

* **Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):**

  * `class_id`: 468[cite: 2]
  * `class_name`: `"cab"`[cite: 2]
  * `rank`: 1[cite: 2]
  * `score`: 0.510915[cite: 2]
  * `taxonomy_name`: `"ImageNet-1K"`[cite: 2]

* **Record này mô tả toàn ảnh như thế nào?**

  Nó cung cấp một nhãn duy nhất (`"cab"` - xe taxi) đại diện cho toàn bộ bối cảnh của bức ảnh với độ tin cậy cao nhất, dù thực tế trong ảnh có chứa rất nhiều đối tượng tham gia giao thông khác[cite: 2, 3].

* **Ai định nghĩa class list mà checkpoint có thể dự đoán?**

  Được định nghĩa bởi tập dữ liệu taxonomy `"ImageNet-1K"` dùng để huấn luyện mô hình phân loại này[cite: 2].

* **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?**

  Để đảm bảo tính nhất quán của dữ liệu. Cùng một tên lớp (`"cab"`) nhưng ở các bộ taxonomy khác nhau có thể mang ý nghĩa hoặc cách phân loại khác nhau, giữ ID và tên taxonomy giúp hệ thống map chính xác định nghĩa gốc.

* **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**

  Guideline cần quy định rõ nguyên tắc chọn nhãn: chỉ chọn chủ thể nổi bật nhất/chiếm diện tích lớn nhất, hoặc cho phép gán nhiều nhãn (multi-label) để mô tả đầy đủ bức ảnh.

* **Vì sao model score không phải ground truth?**

  Model score chỉ là xác suất hoặc độ tự tin do thuật toán máy tính dự đoán (ví dụ: 0.510915)[cite: 2]. Ground truth là nhãn chính xác tuyệt đối, mang tính tiêu chuẩn do con người (annotator) gán thủ công.

## 2. Phát hiện vật thể – lớp và box cho từng object

*Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.*

* **Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):**

  * `class_name`: `"bowl"`[cite: 3]
  * `score`: 0.719062[cite: 3]
  * `bbox_xyxy`: [32.65, 342.12, 100.16, 384.93][cite: 3]
  * `bbox_width`: 67.51[cite: 3]
  * `bbox_height`: 42.81[cite: 3]

* **Diễn giải vị trí box bằng lời:**

  Hộp giới hạn (bounding box) bao quanh cái bát bắt đầu từ tọa độ góc trên bên trái ở pixel (32.65, 342.12), kết thúc ở góc dưới bên phải tại pixel (100.16, 384.93), tạo thành một khung chữ nhật có chiều rộng xấp xỉ 67.5 pixel và chiều cao khoảng 42.8 pixel[cite: 3].

* **So sánh số prediction ở hai threshold:**

  Với ngưỡng (threshold) thấp là 0.35[cite: 3], mô hình trả về số lượng prediction lớn, bắt được nhiều vật thể nhưng dễ bị lẫn rác (False Positive). Nếu nâng threshold lên cao (VD: 0.6), số lượng prediction sẽ giảm, độ chính xác của từng box cao hơn nhưng nguy cơ bỏ sót vật thể (False Negative) cũng tăng lên.

* **Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?**

  Ngưỡng thấp làm tăng tối đa độ bao phủ (không bỏ lọt vật) nhưng sẽ khiến khối lượng công việc của reviewer tăng vọt vì phải rà soát và loại bỏ rất nhiều bounding box sai lệch.

* **Đề xuất một quy tắc box chặt:**

  Các cạnh của bounding box phải tiếp xúc vừa khít với các điểm viền ngoài cùng của vật thể (phần nhìn thấy được). Không được vẽ lẹm vào vật thể và khoảng trống (padding) giữa mép box với vật thể không được vượt quá 2-3 pixel.

* **Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?**

  Guideline cần quy định rõ: annotator chỉ được vẽ box quanh phần "nhìn thấy được" hay phải ước lượng vẽ trùm lên phần "bị che khuất". Trong trường hợp vật thể bị che khuất quá mức (ví dụ trên 80%) và mất đi các đặc trưng nhận diện cốt lõi, cần escalation để quyết định giữ hay bỏ box đó.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

*Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.*

* **Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):**

  * `instance_id`: `"kitchen-002"`[cite: 4]
  * `class_name`: `"bowl"`[cite: 4]
  * `score`: 0.735743[cite: 4]
  * `polygon_point_count`: 67[cite: 4]
  * `polygon_xy`: [[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0]...][cite: 4]

* **Polygon bổ sung chi tiết gì so với box?**

  Thay vì chỉ là khung chữ nhật thô cứng chứa cả phần nền như box, Polygon dùng chuỗi tọa độ (67 điểm ở ví dụ trên) để vẽ thành một đa giác khép kín ôm khít hình dáng thực tế của cái bát, loại bỏ hoàn toàn các phần background thừa thãi[cite: 4].

* **`instance_id` dùng để làm gì và không phải loại ID nào?**

  `instance_id` (ví dụ: `"kitchen-002"`) được dùng để định danh tính duy nhất của một cá thể riêng biệt trong ảnh (để hệ thống phân biệt "cái bát A" và "cái bát B")[cite: 4]. Nó hoàn toàn khác với Class ID (dùng để gán nhãn thể loại, ví dụ ID 45 đại diện chung cho mọi cái bát)[cite: 4].

* **Đề xuất một quy tắc biên mask:**

  Đường viền của mask phải bám sát chính xác từng pixel ở mép biên của vật thể, không được tràn sang phần nền (background) hoặc lấn vào vùng của các vật thể lân cận.

* **Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?**

  Cần quy định khi xử lý vùng rìa mờ (motion blur/out of focus) thì lấy mép trong hay mép ngoài. Nếu hai vật thể cùng loại dính sát vào nhau, guideline cần chỉ rõ là tách thành hai mask riêng biệt hay gộp thành một. Những ca giao thoa phức tạp cần escalation lên team lead để chốt phương án.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                    | Đơn vị/định dạng ground truth                                      | Lỗi hoặc điểm mơ hồ quan sát được                                                                              | Annotator làm gì?                                                                           | Reviewer xem gì?                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phân loại ảnh**         | Nhãn danh mục tổng thể cho toàn bức ảnh (1 label/ảnh).             | Nhãn không phản ánh đúng ngữ cảnh chính, model nhầm lẫn bối cảnh phòng bếp thành `"gong"` (chiêng)[cite: 2].   | Nhìn bao quát toàn ảnh và chọn 1 nhãn miêu tả đúng nhất đối tượng trọng tâm theo guideline. | Đánh giá xem nhãn được chọn có thực sự là đối tượng nổi bật nhất/phù hợp với guideline dự án hay không.                                      |
| **Phát hiện vật thể**     | Bounding box `[x_min, y_min, x_max, y_max]` kết hợp với tên nhãn.  | Box quá lỏng (dư nhiều nền), box quá chặt (cắt lẹm mất chi tiết), hoặc bắt sai loại vật thể.                   | Vẽ khung hình chữ nhật sát với mép ngoài cùng của phần vật thể nhìn thấy được và gán nhãn.  | Soi kỹ các mép box xem có sát vật thể chưa, kiểm tra độ phủ (có bỏ sót không) và độ chính xác của nhãn gán.                                  |
| **Instance segmentation** | Polygon (danh sách tạo bởi điểm tọa độ `[x, y]`) hoặc Mask + Nhãn. | Đường viền đa giác vẽ lấn sang vùng nền, bị răng cưa cắt lẹm vào vật thể, phân tách sai các vật thể dính liền. | Chấm các điểm tạo thành đa giác ôm thật sát hình dáng thực tế của từng cá thể (instance).   | Thu phóng (zoom in) ở mức pixel để soi đường viền ranh giới mask, ranh giới giữa các vật thể đè lên nhau và tính duy nhất của từng instance. |

## 5. An toàn dữ liệu

* **Một quy tắc bảo vệ dữ liệu:**

  Nghiêm cấm việc tải xuống, sao chép cục bộ, chụp màn hình, quay video hay chia sẻ hình ảnh/dữ liệu của dự án ra các nền tảng bên ngoài môi trường làm việc đã được chỉ định.

* **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:**

  Quản lý dự án (Project Manager / Team Lead) hoặc bộ phận An toàn thông tin qua kênh báo cáo nội bộ ngay lập tức để có hướng xử lý kịp thời.

## 6. Danh sách bằng chứng

* [x] `classification_predictions.json`
* [x] `detection_predictions.json`
* [x] `segmentation_predictions.json`
* [x] `IMAGE_ATTRIBUTION.md`
* [x] `visuals/classification_top5.png`
* [x] `visuals/detection_predictions.png`
* [x] `visuals/segmentation_prediction.png`
* [x] Ô validation cuối notebook báo **PASS**.
* [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
