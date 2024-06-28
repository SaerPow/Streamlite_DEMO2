import streamlit as st
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import time
# import torch
# import cv2

# Загрузка предобученной модели YOLOv8
model = YOLO('yolov8n.pt')  # Используйте 'yolov8n.pt' или другую версию модели

# Использование GPU, если доступно
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# Функция для предсказания и отображения bounding box
def detect_objects(image):
    # Преобразование изображения в формат, подходящий для модели
    img = np.array(image)

    # Изменение размера изображения для ускорения обработки
    resized_img = cv2.resize(img, (640, 640))

    start_time = time.time()
    results = model(resized_img)
    end_time = time.time()

    processing_time = end_time - start_time

    # Получение данных по bounding box и меткам
    bbox_data = results[0].boxes
    for bbox in bbox_data:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        label = bbox.cls
        conf = bbox.conf.item()  # Преобразование тензора в скалярное значение

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        cv2.putText(
            img,
            f'{model.names[int(label)]} {conf:.2f}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    return img, processing_time


# Интерфейс Streamlit
st.title("YOLOv8 Object Detection")
st.write("Загрузите изображение и модель YOLOv8 обнаружит объекты на нем.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение.', use_column_width=True)

    st.write("")
    st.write("Обрабатываем изображение...")

    result_image, processing_time = detect_objects(image)

    st.image(result_image, caption='Изображение с обнаруженными объектами.', use_column_width=True)
    st.write(f"Время обработки: {processing_time:.2f} секунд")




