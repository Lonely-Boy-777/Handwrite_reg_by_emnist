# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from config import *
from byclass.advanced_preprocessor import cleanup_and_standardize_char


class ProOCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nhận dạng chữ viết tay sử dụng mạng nơ ron với Byclass")
        self.geometry("1100x800")
        self.configure(bg="#F0F0F0")

        # --- BIẾN LƯU TRỮ TRẠNG THÁI ---
        self.original_pil_image = None
        self.display_cv_image = None
        self.ready_for_prediction_img = None
        self.crop_rect_id = None
        self.crop_start_x, self.crop_start_y = 0, 0
        self.scale_factor, self.x_offset, self.y_offset = 1.0, 0, 0

        # --- TẢI MODEL ---
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải model từ '{MODEL_PATH}': {e}")
            self.destroy();
            return

        self._setup_styles()
        self._load_icons()
        self._create_widgets()

    def _setup_styles(self):
        BG_COLOR = "#F0F0F0"
        FRAME_BG_COLOR = "#FFFFFF"
        TEXT_COLOR = "#1E1E1E"
        ACCENT_COLOR = "#0078D7"
        ACCENT_ACTIVE_COLOR = "#005a9e"

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 10))
        style.configure('TFrame', background=BG_COLOR)
        style.configure('Toolbar.TFrame', background='#E1E1E1')
        style.configure('TButton', font=('Helvetica', 10), padding=5)
        style.configure('Accent.TButton', font=('Helvetica', 11, 'bold'), foreground='white', background=ACCENT_COLOR,
                        padding=8)
        style.map('Accent.TButton', background=[('active', ACCENT_ACTIVE_COLOR)])
        style.configure('TLabelFrame', background=FRAME_BG_COLOR, borderwidth=1, relief="solid")
        style.configure('TLabelFrame.Label', foreground=TEXT_COLOR, background=FRAME_BG_COLOR,
                        font=('Helvetica', 12, 'bold'))
        style.configure('TLabel', background=FRAME_BG_COLOR)
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'), background=BG_COLOR)
        style.configure('Success.TLabel', foreground='#2E7D32', background=FRAME_BG_COLOR,
                        font=('Helvetica', 32, 'bold'))
        style.configure('Warning.TLabel', foreground='#FF8F00', background=FRAME_BG_COLOR,
                        font=('Helvetica', 32, 'bold'))
        style.configure('Error.TLabel', foreground='#D32F2F', background=FRAME_BG_COLOR, font=('Helvetica', 32, 'bold'))
        style.configure('Confidence.TLabel', foreground='#555555', background=FRAME_BG_COLOR, font=('Helvetica', 12))

    def _load_icons(self):
        try:
            self.upload_icon = ImageTk.PhotoImage(
                Image.open("assets/icons/upload.png").resize((20, 20), Image.Resampling.LANCZOS))
            self.rotate_left_icon = ImageTk.PhotoImage(
                Image.open("assets/icons/rotate-left.png").resize((20, 20), Image.Resampling.LANCZOS))
            self.rotate_right_icon = ImageTk.PhotoImage(
                Image.open("assets/icons/rotate-right.png").resize((20, 20), Image.Resampling.LANCZOS))
            self.help_icon = ImageTk.PhotoImage(
                Image.open("assets/icons/help.png").resize((20, 20), Image.Resampling.LANCZOS))
            self.recognize_icon = ImageTk.PhotoImage(
                Image.open("assets/icons/recognize.png").resize((20, 20), Image.Resampling.LANCZOS))
        except Exception as e:
            print(f"Cảnh báo: Không thể tải icon. Lỗi: {e}")
            self.upload_icon = self.rotate_left_icon = self.rotate_right_icon = self.help_icon = self.recognize_icon = None

    def _create_widgets(self):
        toolbar = ttk.Frame(self, style='Toolbar.TFrame', padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text=" Tải ảnh", image=self.upload_icon, compound=tk.LEFT, command=self.load_image).pack(
            side=tk.LEFT, padx=5, pady=5)
        ttk.Button(toolbar, text=" Xoay Trái", image=self.rotate_left_icon, compound=tk.LEFT,
                   command=lambda: self.transform_image(cv2.ROTATE_90_COUNTERCLOCKWISE)).pack(side=tk.LEFT, padx=5,
                                                                                              pady=5)
        ttk.Button(toolbar, text=" Xoay Phải", image=self.rotate_right_icon, compound=tk.LEFT,
                   command=lambda: self.transform_image(cv2.ROTATE_90_CLOCKWISE)).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(toolbar, text=" Hướng dẫn", image=self.help_icon, compound=tk.LEFT,
                   command=self._show_instructions).pack(side=tk.RIGHT, padx=5, pady=5)

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        self.main_canvas = tk.Canvas(left_frame, bg="#EAEAEA", highlightthickness=1, highlightbackground="#CCCCCC")
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_text = self.main_canvas.create_text(400, 350, text="Tải ảnh lên để bắt đầu...",
                                                        font=('Helvetica', 16, 'italic'), fill='#555555')

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="ns")

        # --- CẢI TIẾN: Tách thành 2 khung riêng biệt ---
        # Khung 1: Xem trước và Nhận dạng
        predict_frame = ttk.LabelFrame(right_frame, text="Thông tin")
        predict_frame.pack(fill=tk.X, pady=(0, 20), ipady=5)

        self.processed_canvas = tk.Canvas(predict_frame, bg="black", width=150, height=150, highlightthickness=1,
                                          highlightbackground="#CCCCCC")
        self.processed_canvas.pack(pady=5, padx=10)
        self.predict_button = ttk.Button(predict_frame, text=" Nhận dạng", image=self.recognize_icon, compound=tk.LEFT,
                                         command=self.predict, state=tk.DISABLED, style='Accent.TButton')
        self.predict_button.pack(pady=15, padx=10, fill=tk.X, ipady=8)

        # Khung 2: Kết quả
        result_frame = ttk.LabelFrame(right_frame, text="Kết quả")
        result_frame.pack(fill=tk.X)
        self.result_char_label = ttk.Label(result_frame, text="", style='Result.TLabel', anchor="center")
        self.result_char_label.pack(pady=(15, 5), padx=10)
        self.confidence_label = ttk.Label(result_frame, text="", style='Confidence.TLabel', anchor="center")
        self.confidence_label.pack(pady=(0, 15), padx=10)

        self.main_canvas.bind("<ButtonPress-1>", self.start_crop)
        self.main_canvas.bind("<B1-Motion>", self.do_crop)
        self.main_canvas.bind("<ButtonRelease-1>", self.end_crop_and_prepare)

    def _show_instructions(self):
        instructions = "HƯỚNG DẪN SỬ DỤNG\n\n1. Nhấn 'Tải ảnh'.\n2. Dùng 'Xoay' nếu cần.\n3. Dùng chuột cắt 1 ký tự, ảnh preview sẽ hiện ra.\n4. Nhấn nút 'Nhận dạng' để xem kết quả. \n*Vui lòng tải lên ảnh không bị mờ để có kết quả chính xác nhất! "
        messagebox.showinfo("Hướng dẫn", instructions)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path: return
        self.display_cv_image = cv2.imread(file_path)
        self.original_pil_image = Image.open(file_path).convert("RGB")
        if self.display_cv_image is None:
            messagebox.showerror("Lỗi", "Không thể đọc file ảnh.")
            return
        self.main_canvas.delete(self.canvas_text)
        self._display_image_on_canvas(self.display_cv_image)
        self.processed_canvas.delete("all")
        self.result_char_label.config(text="")
        self.confidence_label.config(text="")
        self.predict_button.config(state=tk.DISABLED)

    def transform_image(self, code):
        if self.display_cv_image is not None:
            self.display_cv_image = cv2.flip(self.display_cv_image, code) if code in [0, 1] else cv2.rotate(
                self.display_cv_image, code)
            self._display_image_on_canvas(self.display_cv_image)
            # Reset trạng thái sau khi xoay
            self.ready_for_prediction_img = None
            self.processed_canvas.delete("all")
            self.predict_button.config(state=tk.DISABLED)
            self.result_char_label.config(text="")
            self.confidence_label.config(text="")

    def end_crop_and_prepare(self, event):
        if not self.crop_rect_id or self.display_cv_image is None: return
        self.main_canvas.delete(self.crop_rect_id)
        self.crop_rect_id = None

        x1 = int((min(self.crop_start_x, event.x) - self.x_offset) / self.scale_factor)
        y1 = int((min(self.crop_start_y, event.y) - self.y_offset) / self.scale_factor)
        x2 = int((max(self.crop_start_x, event.x) - self.x_offset) / self.scale_factor)
        y2 = int((max(self.crop_start_y, event.y) - self.y_offset) / self.scale_factor)
        if x2 <= x1 or y2 <= y1: return

        manually_cropped = self.display_cv_image[y1:y2, x1:x2]

        final_char_img = cleanup_and_standardize_char(manually_cropped)
        if final_char_img is None: return

        self.display_processed_char(final_char_img)
        self.ready_for_prediction_img = final_char_img
        self.predict_button.config(state=tk.NORMAL)
        self.result_char_label.config(text="")
        self.confidence_label.config(text="")

    def predict(self):
        if self.ready_for_prediction_img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng cắt một ký tự trước khi nhận dạng.")
            return

        img_to_predict = self.ready_for_prediction_img.astype('float32') / 255.0
        img_to_predict = np.expand_dims(img_to_predict, axis=-1)
        img_to_predict = np.expand_dims(img_to_predict, axis=0)

        try:
            prediction = self.model.predict(img_to_predict)
            char_index = np.argmax(prediction)
            character = LABEL_MAPPING.get(char_index, "?")
            confidence = np.max(prediction) * 100

            if confidence >= 80:
                style_name = 'Success.TLabel'
            elif 50 <= confidence < 80:
                style_name = 'Warning.TLabel'
            else:
                style_name = 'Error.TLabel'

            self.result_char_label.config(text=f"{character}", style=style_name)
            self.confidence_label.config(text=f"(Độ tin cậy: {confidence:.2f}%)")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi dự đoán: {e}")

    # --- CÁC HÀM TIỆN ÍCH KHÁC ---
    def _display_image_on_canvas(self, cv_image):
        canvas_w = self.main_canvas.winfo_width() if self.main_canvas.winfo_width() > 1 else 750
        canvas_h = self.main_canvas.winfo_height() if self.main_canvas.winfo_height() > 1 else 700
        img_h, img_w, _ = cv_image.shape
        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        display_w, display_h = int(img_w * self.scale_factor), int(img_h * self.scale_factor)
        resized_cv_img = cv2.resize(cv_image, (display_w, display_h))
        img_rgb = cv2.cvtColor(resized_cv_img, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

        self.main_canvas.delete("all")
        self.main_canvas.create_image(canvas_w / 2, canvas_h / 2, anchor=tk.CENTER, image=self.tk_image)
        self.x_offset = (canvas_w - display_w) / 2
        self.y_offset = (canvas_h - display_h) / 2

    def display_processed_char(self, char_img):
        self.processed_canvas.delete("all")
        img_pil = Image.fromarray(char_img)
        img_pil = img_pil.resize((150, 150), Image.Resampling.NEAREST)
        tk_img = ImageTk.PhotoImage(image=img_pil)
        self.processed_canvas.create_image(75, 75, anchor=tk.CENTER, image=tk_img)
        self.processed_canvas.image = tk_img

    def start_crop(self, event):
        self.crop_start_x, self.crop_start_y = event.x, event.y
        if self.crop_rect_id: self.main_canvas.delete(self.crop_rect_id)
        self.crop_rect_id = self.main_canvas.create_rectangle(self.crop_start_x, self.crop_start_y, self.crop_start_x,
                                                              self.crop_start_y, outline='cyan', width=2, dash=(4, 2))

    def do_crop(self, event):
        if self.crop_rect_id: self.main_canvas.coords(self.crop_rect_id, self.crop_start_x, self.crop_start_y, event.x,
                                                      event.y)