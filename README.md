

````markdown
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

Để chạy được dự án, vui lòng thực hiện theo các bước sau.

**1. Tải dự án về máy (Clone):**
```bash
git clone [URL-KHO-CHUA-GIT-CUA-BAN]
cd [TEN-THU-MUC-DU-AN]
````

**2. Lựa chọn phiên bản và di chuyển vào thư mục:**
Hãy quyết định bạn muốn chạy phiên bản nào. Ví dụ, để chạy phiên bản `balanced`:

```bash
cd balanced
```

*(Lưu ý: Các lệnh tiếp theo phải được thực hiện bên trong thư mục phiên bản bạn đã chọn)*

**3. Tạo và kích hoạt môi trường ảo:**

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (trên Windows)
venv\Scripts\activate

# Kích hoạt môi trường ảo (trên macOS/Linux)
source venv/bin/activate
```

**4. Cài đặt các thư viện cần thiết:**

```bash
pip install -r requirements.txt
```

*(Lưu ý: Hãy chắc chắn bạn đã có file `requirements.txt` trong mỗi thư mục dự án)*

**5. Chạy ứng dụng:**
Sau khi cài đặt xong, chạy file `main.py` để khởi động giao diện.

```bash
python main.py
```

## 🧠 Huấn luyện Mô hình

Mô hình được huấn luyện bằng Google Colab tại các đường dẫn sau:

  * **Model ByClass:** [Link tới Google Colab](https://colab.research.google.com/drive/1DBSqM-B2FwLzB9SaF8KVC_yFftAEQxef?usp=sharing)
  * **Model Balanced:** [Link tới Google Colab](https://colab.research.google.com/drive/11qXK_W5CEG7WC7YC9O78_w76ZqJl2Wi8?usp=sharing)

## 🛠️ Công nghệ sử dụng

  * Python
  * TensorFlow / Keras
  * OpenCV
  * Tkinter
  * Scikit-learn
  * Numpy
  * Matplotlib / Seaborn

<!-- end list -->

```
```
