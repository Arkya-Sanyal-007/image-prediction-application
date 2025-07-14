import flet as ft
import numpy as np
from PIL import Image
import base64
import subprocess
from io import BytesIO
from model_utils import load_model
from db_utils_CNN import get_random_images_by_label
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

clf = load_model()
class_labels = clf.classes_
cnn_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(128, 128, 3))

def preprocess_image(path):
    img = Image.open(path).convert('RGB').resize((128, 128)) 
    img_array = np.array(img).astype("float32")
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    return features

def main(page: ft.Page):
    page.title = "Image Class Prediction & Viewer"
    page.scroll = ft.ScrollMode.AUTO

    uploaded_img = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)
    status_text = ft.Text()
    num_slider = ft.Slider(min=1, max=10, divisions=9, label="{value}", value=5)
    image_grid = ft.Column(scroll=ft.ScrollMode.AUTO, auto_scroll=False)

    def display_images_in_rows(images):
        image_grid.controls.clear()
        row = ft.Row(wrap=True, spacing=10, run_spacing=10)
        for i, img_data in enumerate(images):
            encoded = base64.b64encode(img_data).decode("utf-8")
            img_widget = ft.Image(
                src_base64=encoded,
                width=200,
                height=200,
                fit=ft.ImageFit.CONTAIN
            )
            row.controls.append(img_widget)
            if (i + 1) % 3 == 0:
                image_grid.controls.append(row)
                row = ft.Row(wrap=True, spacing=10, run_spacing=10)
        if row.controls:
            image_grid.controls.append(row)
        page.update()

    def on_upload(e: ft.FilePickerResultEvent):
        if e.files:
            path = e.files[0].path
            uploaded_img.src = path
            uploaded_img.update()

            image_grid.controls.clear()
            status_text.value = "Predicting..."
            page.update()

            input_array = preprocess_image(path)
            probs = clf.predict_proba(input_array)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            predicted_label = class_labels[best_idx]

            if confidence < 0.30:
                status_text.value = f"Prediction confidence too low ({confidence:.2%}). No image shown."
                page.update()
                return

            status_text.value = f"Predicted Label: {predicted_label} (Confidence: {confidence:.2%})"
            images = get_random_images_by_label(predicted_label, count=int(num_slider.value))
            display_images_in_rows(images)
            page.update()

    def retrain_model(e):
        status_text.value = "Retraining model..."
        page.update()
        result = subprocess.run(["python", "retrain.py"], capture_output=True, text=True)
        if result.returncode == 0:
            status_text.value = "Model retrained successfully. Reloading..."
            global clf
            clf = load_model()
            status_text.value = "Model reloaded and ready to use."
        else:
            status_text.value = f"Retraining failed:\n{result.stderr}"
        page.update()

    file_picker = ft.FilePicker(on_result=on_upload)
    page.overlay.append(file_picker)

    page.add(
        ft.Text("Upload an image:"),
        ft.ElevatedButton("Select Image", on_click=lambda _: file_picker.pick_files(allow_multiple=False)),
        uploaded_img,
        status_text,
        ft.Text("Number of images to display:"),
        num_slider,
        ft.Text("Images from predicted label:"),
        image_grid,
        ft.ElevatedButton("Retrain Model", on_click=retrain_model, bgcolor=ft.colors.BLUE, color=ft.colors.WHITE),
    )

ft.app(target=main)