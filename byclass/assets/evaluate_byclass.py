import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# PHẦN 1: CẤU HÌNH CHO MODEL EMNIST BYCLASS

MODEL_PATH = '../saved_model_char_byclass/emnist_byclass_cnn_model.h5'
NUM_CLASSES = 62
LABEL_MAPPING = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z',
    36:'a', 37:'b', 38:'c', 39:'d', 40:'e', 41:'f', 42:'g', 43:'h', 44:'i', 45:'j',
    46:'k', 47:'l', 48:'m', 49:'n', 50:'o', 51:'p', 52:'q', 53:'r', 54:'s', 55:'t',
    56:'u', 57:'v', 58:'w', 59:'x', 60:'y', 61:'z'
}

# PHẦN 2: QUÁ TRÌNH KIỂM ĐỊNH

print(">>> BẮT ĐẦU QUÁ TRÌNH KIỂM ĐỊNH MODEL BYCLASS...")

# 1. Tải lại model đã huấn luyện
print(f"Đang tải model từ: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Lỗi: Không thể tải model. Chi tiết lỗi: {e}")
    exit()

# 2. Tải và chuẩn bị tập dữ liệu TEST của ByClass
print("Đang tải và chuẩn bị tập dữ liệu test của EMNIST/ByClass...")
(ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split='test',
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])
    return image, label

ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
print("Dữ liệu test đã sẵn sàng.")

# 3. Lấy tất cả các dự đoán và nhãn thật từ tập test
print("\n>>> Đang thực hiện dự đoán trên toàn bộ tập test...")
y_pred_probs = model.predict(ds_test)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = []
for images, labels in ds_test:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# Lấy danh sách tên các lớp (ký tự) để hiển thị báo cáo
class_names = [LABEL_MAPPING[i] for i in range(NUM_CLASSES)]

# 4. In báo cáo phân loại
print("\n" + "="*50)
print("BÁO CÁO PHÂN LOẠI (PRECISION, RECALL, F1-SCORE) CHO MODEL BYCLASS")
print("="*50)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# 5. Tạo và hiển thị ma trận nhầm lẫn
print("\n" + "="*50)
print("ĐANG TẠO MA TRẬN NHẦM LẪN CHO MODEL BYCLASS...")
print("="*50)
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(20, 17))
sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Ma trận nhầm lẫn cho bộ dữ liệu EMNIST ByClass Test')
plt.xlabel('Nhãn được dự đoán (Predicted Label)')
plt.ylabel('Nhãn thật (True Label)')
plt.tight_layout()
plt.savefig('confusion_matrix_byclass.png') # Lưu lại ảnh với tên riêng
print("Đã lưu ma trận nhầm lẫn vào file 'confusion_matrix_byclass.png'")
plt.show()