# Đồ án Deep Learning

Repository này chứa mã nguồn và hướng dẫn cài đặt cho đồ án Deep Learning. Do dung lượng các file trọng số mô hình (weights) vượt quá giới hạn cho phép của GitHub (>100MB), toàn bộ các file `.pth` đã được lưu trữ riêng trên Google Drive.

---

## ⏬ Hướng dẫn tải Weights (Trọng số mô hình)

Để chạy được chương trình, bạn vui lòng tải các file mô hình dưới đây và sao chép chúng vào thư mục `models/` trong dự án:

* **Mô hình Hybrid Vision Transformer & ResNet18:**
    * [Tải file hybrid_vit_resnet18_best.pth](Dán link Google Drive file best của ViT vào đây)
    * [Tải file hybrid_vit_resnet18_last.pth](Dán link Google Drive file last của ViT vào đây)
* **Mô hình EfficientNet:**
    * [Tải file efficientNet_best.pth](Dán link Google Drive file best của EfficientNet vào đây)
    * [Tải file efficientNet_last.pth](Dán link Google Drive file last của EfficientNet vào đây)


Cài đặt các thư viện cần thiết bằng lệnh:

pip install -r requirements.txt
*Lưu ý: Sau khi tải về, cấu trúc thư mục dạng chuẩn sẽ như sau:*
```text
DoAn_DeepLearning/
├── models/
│   ├── hybrid_vit_resnet18_best.pth
│   ├── hybrid_vit_resnet18_last.pth
│   ├── efficientNet_best.pth
│   └── efficientNet_last.pth
└── ... (các file code .py hoặc .ipynb khác)

