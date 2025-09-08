Đây là ứng dụng có giao diện đồ họa (GUI) được xây dựng bằng Python, với
mục đích chính là nhận dạng các ký tự viết tay từ hình ảnh do người
dùng cung cấp. Chương trình sử dụng mô hình Mạng Nơ-ron Tích chập (CNN)
để đưa ra dự đoán với độ chính xác cao.


--- TÍNH NĂNG CHÍNH ---

  * Tải và hiển thị hình ảnh: Cho phép tải lên các file ảnh (.jpg, .png)
    và hiển thị trên giao diện chính.
  * Xoay ảnh: Cung cấp tính năng xoay ảnh sang trái hoặc phải để điều
    chỉnh hướng của văn bản.
  * Cắt ký tự trực quan: Người dùng có thể dùng chuột để vẽ một hình
    chữ nhật xung quanh ký tự cần nhận dạng.
  * Xem trước ảnh tiền xử lí: Hiển thị hình ảnh ký tự sau khi đã được
    tiền xử lý (chuyển sang trắng đen, chuẩn hóa kích thước, căn giữa)
    trước khi đưa vào mô hình.
  * Dự đoán ký tự: Sử dụng mô hình CNN đã được huấn luyện để dự đoán
    ký tự với một nút bấm.
  * Hiển thị kết quả và độ tin cậy: Trả về kết quả dự đoán kèm theo
    tỷ lệ phần trăm độ tin cậy của mô hình. Màu sắc kết quả thay đổi
    dựa trên độ tin cậy.

--- CẤU TRÚC DỰ ÁN ---


Kho chứa này bao gồm hai phiên bản độc lập của ứng dụng, nằm trong
hai thư mục riêng biệt:

  * /byclass:
    Phiên bản sử dụng model được huấn luyện trên bộ dữ liệu EMNIST ByClass
    (62 lớp), có khả năng phân biệt chữ viết hoa và viết thường (ví dụ: A vs a).

  * /balanced:
    Phiên bản sử dụng model được huấn luyện trên bộ dữ liệu EMNIST Balanced
    (47 lớp). Phiên bản này được tối ưu hơn, đã hợp nhất các ký tự có
    hình dạng giống nhau (ví dụ: 'C' và 'c') để tăng độ chính xác.


--- KHỞI ĐỘNG ỨNG DỤNG ---

  Sau khi tải xuống hoàn tất, đảm bảo bạn đang ở trong thư mục của phiên
  bản đã chọn (/byclass hoặc /balanced), hãy chạy 'main.py'
  Cửa sổ ứng dụng sẽ xuất hiện và bạn có thể bắt đầu sử dụng.

  
--- THÔNG TIN HUẤN LUYỆN ---

  Mô hình được huấn luyện bằng Google Colab. Chi tiết xem tại:
  * Model ByClass: https://colab.research.google.com/drive/1DBSqM-B2FwLzB9SaF8KVC_yFftAEQxef
  * Model Balanced: https://colab.research.google.com/drive/11qXK_W5CEG7WC7YC9O78_w76ZqJl2Wi8


--- CÔNG NGHỆ SỬ DỤNG ---

  * Python
  * TensorFlow / Keras
  * OpenCV
  * Tkinter
  * Scikit-learn
  * Numpy
  * Matplotlib / Seaborn


