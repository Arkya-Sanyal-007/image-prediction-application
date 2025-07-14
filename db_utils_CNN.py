import sqlite3
import random
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DB_PATH = "photo_database.sqlite"
cnn_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(128, 128, 3))

def load_dataset_from_db():
    X, y = [], []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT image_label, image_data FROM photo_dataset")

    for image_label, image_data in cur.fetchall():
        try:
            img = Image.open(BytesIO(image_data)).convert('RGB').resize((128, 128))
            img_array = np.array(img).astype("float32")
            img_array = preprocess_input(img_array)
            features = cnn_model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
            X.append(features)
            y.append(image_label)
        except Exception as e:
            print(f"Error processing image from DB: {e}")
    conn.close()
    return np.array(X), np.array(y)

def get_random_images_by_label(image_label, count=5):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT image_data FROM photo_dataset WHERE image_label = ?", (image_label,))
    rows = cur.fetchall()
    conn.close()
    selected = random.sample(rows, min(count, len(rows)))
    return [row[0] for row in selected]