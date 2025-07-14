# Image Class Prediction & Viewer (Flet + TensorFlow)

This is a desktop/web application built with **Flet**, **TensorFlow**, and **MobileNetV2** that allows you to:

- Upload an image
- Predict its class using a pre-trained classifier
- Display similar images from a local/database dataset
- Retrain the model directly from the interface

## 🚀 Features

- Upload and preview images
- Predict class using a trained model (`model.pkl`)
- Feature extraction using MobileNetV2
- View similar images from database (`photo_database.sqlite`)
- Retrain model with a button click
- PostgreSQL or SQLite-compatible backend

## 📂 Code Structure

- `app_CNN.py` – Main application (UI)
- `db_utils_CNN.py` – Feature extraction and DB operations
- `model_utils.py` – Training & loading ML model
- `retrain.py` – Re-trains the classifier (triggered from app)

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

### requirements.txt
```
flet
tensorflow
pillow
numpy
scikit-learn
psycopg2-binary
```

## 🧪 How to Run

```bash
python app_CNN.py
```

> Make sure you have `model.pkl` and `photo_database.sqlite` in the root directory.

## 📄 Documentation

Full documentation is available in [`docs/Machine Learning Applications Project.docx`](docs/Machine%20Learning%20Applications%20Project.docx)