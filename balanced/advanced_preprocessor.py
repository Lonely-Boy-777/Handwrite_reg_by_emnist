# advanced_preprocessor.py
import cv2
import numpy as np
from config import *

def center_on_mass(image_28x28):
#Nhận một ảnh 28x28 và căn giữa ký tự dựa trên trọng tâm.
    moments = cv2.moments(image_28x28)
    if moments["m00"] == 0:
        return image_28x28

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Dịch chuyển cần thiết để đưa trọng tâm về (14, 14)
    shiftx = IMG_SIZE // 2 - cx
    shifty = IMG_SIZE // 2 - cy

    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    centered_image = cv2.warpAffine(image_28x28, M, (IMG_SIZE, IMG_SIZE))
    return centered_image

def deskew(image):
#Hàm chỉnh thẳng một ảnh ký tự bị nghiêng.
    moments = cv2.moments(image)
    if abs(moments['mu02']) < 1e-2:
        return image.copy()

    skew = moments['mu11'] / moments['mu02']
    h, w = image.shape
    M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
    deskewed_image = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return deskewed_image


def cleanup_and_standardize_char(char_image_bgr):
# Nhận một ảnh ký tự ĐÃ ĐƯỢC CẮT (dạng BGR) và thực hiện pipeline làm sạch.
    if char_image_bgr is None or char_image_bgr.shape[0] == 0 or char_image_bgr.shape[1] == 0:
        return None
    # 1. Chuyển xám và nhị phân hóa
    gray = cv2.cvtColor(char_image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # --- BƯỚC 1 CẢI TIẾN: Chỉnh thẳng ký tự
    deskewed_char = deskew(binary)
    # 2. Thêm một đường viền nhỏ xung quanh để tránh bị cắt mất rìa khi resize
    border_size = 4
    padded_char = cv2.copyMakeBorder(deskewed_char, border_size, border_size, border_size, border_size,
                                     cv2.BORDER_CONSTANT, value=0)
    # 3. Resize về 28x28
    resized_char = cv2.resize(padded_char, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # --- BƯỚC 2 CẢI TIẾN: Căn giữa theo khối lượng
    # Thực hiện sau khi đã có ảnh 28x28 để đảm bảo căn giữa chính xác
    final_char = center_on_mass(resized_char)
    return final_char