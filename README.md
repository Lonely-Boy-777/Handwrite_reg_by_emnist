Markdown
# Dự án Nhận dạng Ký tự Viết tay bằng Mạng CNN

Đây là chương trình nhận dạng chữ viết tay sử dụng mô hình mạng nơ-ron tích chập (CNN) để huấn luyện. Ứng dụng có giao diện đồ họa thân thiện, cho phép người dùng tải ảnh, chọn ký tự và thực hiện dự đoán.

## 📂 Cấu trúc Dự án

Dự án gồm có 2 phiên bản độc lập, nằm trong hai thư mục **`byclass`** và **`balanced`**. Cả hai phiên bản đều sử dụng cùng một kiến trúc mô hình nhưng được huấn luyện trên hai bộ dữ liệu khác nhau của TensorFlow EMNIST.

* **`/byclass`**:
    * Mô hình được huấn luyện trên bộ dữ liệu `emnist/byclass` (62 lớp).
    * Có khả năng phân biệt chữ viết hoa và viết thường (A vs a).

* **`/balanced`**:
    * Mô hình được huấn luyện trên bộ dữ liệu `emnist/balanced` (47 lớp).
    * Đây là phiên bản được tối ưu, đã hợp nhất các ký tự hoa/thường có hình dạng giống nhau (ví dụ: 'C' và 'c') để tăng độ chính xác.

## 🚀 Hướng dẫn Cài đặt và Sử dụng
Cài đặt các thư viện cần thiết:
Bash
pip install -r requirements.txt
5. Chạy ứng dụng:
Sau khi cài đặt xong, chạy file main.py để khởi động giao diện.
Bash
python main.py
🧠 Huấn luyện Mô hình
Mô hình được huấn luyện bằng Google Colab tại các đường dẫn sau:
•	Model ByClass: Link tới Google Colab
•	Model Balanced: Link tới Google Colab
🛠️ Công nghệ sử dụng
•	Python
•	TensorFlow / Keras
•	OpenCV
•	Tkinter
•	Scikit-learn
•	Numpy
•	Matplotlib / Seaborn

