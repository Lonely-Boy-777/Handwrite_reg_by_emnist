# train_balanced.py
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# PHẦN 1: CẤU HÌNH DỰ ÁN

print(">>> BẮT ĐẦU THIẾT LẬP CẤU HÌNH CHO EMNIST BALANCED...")

# Đường dẫn để lưu model sau khi huấn luyện
MODEL_SAVE_DIR = 'saved_model_char_balanced'
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'emnist_balanced_cnn_model.h5')

# Cấu hình ảnh (Không đổi)
IMG_SIZE = 28
CHANNELS = 1

# Cấu hình model và dữ liệu
# Bộ EMNIST Balanced có 47 lớp
NUM_CLASSES = 47
# Cấu hình huấn luyện
BATCH_SIZE = 256
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Ánh xạ nhãn cho 47 lớp của bộ EMNIST Balanced

LABEL_MAPPING = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z',
    36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'
}
print(">>> CẤU HÌNH HOÀN TẤT.")

# PHẦN 2: ĐỊNH NGHĨA KIẾN TRÚC MODEL

def create_char_cnn_model():
#Xây dựng mô hình CNN để phân loại ký tự.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# PHẦN 3: TẢI DỮ LIỆU VÀ HUẤN LUYỆN

print("\n>>> ĐANG TẢI DỮ LIỆU EMNIST/BALANCED (LẦN ĐẦU CÓ THỂ MẤT MỘT LÚC)...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
print(">>> TẢI DỮ LIỆU HOÀN TẤT.")

# Hàm xử lý dữ liệu
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])
    label = tf.one_hot(label, NUM_CLASSES) # Tự động dùng NUM_CLASSES = 47
    return image, label

# Xây dựng pipeline dữ liệu (Không đổi)
print("\n>>> ĐANG XÂY DỰNG PIPELINE DỮ LIỆU...")
ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
print(">>> PIPELINE DỮ LIỆU ĐÃ SẴN SÀNG.")

# Xây dựng, biên dịch và huấn luyện model (Không đổi)
model = create_char_cnn_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=EARLY_STOPPING_PATIENCE,
                                                  restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH,
                                                      monitor='val_accuracy',
                                                      save_best_only=True)

print("\n>>> BẮT ĐẦU HUẤN LUYỆN VỚI BỘ BALANCED...")
history = model.fit(ds_train,
                    epochs=EPOCHS,
                    validation_data=ds_test,
                    callbacks=[early_stopping, model_checkpoint])

# Đánh giá và lưu model cuối cùng
print("\n>>> ĐÁNH GIÁ TRÊN TẬP TEST...")
loss, accuracy = model.evaluate(ds_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
print(f"\n>>> ĐÃ LƯU MÔ HÌNH TỐT NHẤT TẠI: {MODEL_PATH}")