import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import torch
import itertools

# Load YOLOv8 models
model_np = YOLO("D:/PycharmProjects/Streamlit/np_last (last).pt")  # Change the path as per your model location
model_char = YOLO("D:/PycharmProjects/Streamlit/let_last.pt")  # Path to the new YOLOv8 character model
# Check if GPU is available and move model to GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_np.to(device)
model_char.to(device)

letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#A B C E H K M O P T X Y _0 _1 _2 _3 _4 _5 _6 _7 _8 _9

def adjust_contrast_brightness(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate mean and standard deviation of pixel values
    mean, std = np.mean(gray), np.std(gray)

    # Define target mean and standard deviation for normalization
    target_mean, target_std = 200, 70

    # Calculate alpha and beta
    alpha = target_std / std
    beta = target_mean - mean * alpha

    # Adjust brightness and contrast
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image


# Function to detect objects and draw bounding boxes
def detect_objects(image):
    # Convert PIL image to numpy array
    img = np.array(image)

    # Perform object detection using YOLOv8 for license plates
    start_time = time.time()
    results = model_np(img)
    end_time = time.time()
    processing_time = end_time - start_time

    # Convert image to RGB (if it's grayscale)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Draw bounding boxes on the image
    bbox_image = img.copy()  # Create a copy to draw bounding boxes

    # Initialize variables for storing cropped images
    cropped_images = []

    # View results
    if not results:
        return bbox_image, processing_time, cropped_images

    for r in results:
        if not hasattr(r, 'obb') or not hasattr(r.obb, 'xyxyxyxy') or len(r.obb.xyxyxyxy) == 0:
            continue

        # Extract OBB information
        obb = r.obb

        # Extract coordinates of the rotated rectangle (OBB)
        box = obb.xyxyxyxy[0].cpu().numpy().astype(np.float32).reshape(4, 2)

        # Sort the points to get them in the order: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype=np.float32)
        s = box.sum(axis=1)
        rect[0] = box[np.argmin(s)]
        rect[2] = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        rect[1] = box[np.argmin(diff)]
        rect[3] = box[np.argmax(diff)]

        # Destination rectangle points (256x56 pixels)
        dst = np.array([[0, 0], [256, 0], [256, 56], [0, 56]], dtype=np.float32)

        # Compute the perspective transform matrix and apply it to the cropped image
        M = cv2.getPerspectiveTransform(rect, dst)

        # Convert image to numpy array for warpPerspective
        img_np = np.array(image)
        cropped_image = cv2.warpPerspective(img_np, M, (256, 56))

        # Adjust brightness and contrast of cropped image
        cropped_image = adjust_contrast_brightness(cropped_image)

        # Draw the OBB on the original image
        pts = rect.astype(int)
        cv2.polylines(bbox_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # Green color, thickness 2

        # Extract confidence score (probability)
        prob = obb.conf[0].item()  # Assuming 'conf' is a tensor; use [0] to get the scalar value

        # Annotate the original image with the probability
        cv2.putText(bbox_image, f'{prob:.2f}', (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

        # Append cropped image to list
        cropped_images.append(cropped_image)

    return bbox_image, processing_time, cropped_images


# Function to detect characters using YOLOv8 character model
def detect_characters(image):
    # Convert PIL image to numpy array
    start_time = time.time()
    img = np.array(image)
    results = model_char(img)
    char_results = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].item())
            char_results.append((bbox, cls))

    # Sort characters by x-coordinate to display them in order
    char_results = sorted(char_results, key=lambda x: x[0][0])
    end_time = time.time()
    processing_time2 = end_time - start_time
    st.write(f"Время обнаружения символов: {processing_time2:.2f} seconds")
    return char_results


# Streamlit interface
st.title("YOLOv8 OBB Обнаружение автомобильного номера")
st.write("Загрузите изображение для поиска автомобильного номера")

uploaded_file = st.file_uploader("Загрузите изображение:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение для поиска номера', use_column_width=True)

    # Perform object detection for license plates
    st.write("Обнаружение номера...")
    result_image, processing_time, cropped_images = detect_objects(image)

    # Check if any objects were detected
    if not cropped_images:
        st.warning("Не удалось найти номер!")
    else:
        # Display processed image with bounding boxes
        st.image(result_image, caption='Изображение с найденным номером.', use_column_width=True)
        st.write(f"Время обнаружения: {processing_time:.2f} seconds")

        # Display cropped images below the main display
        st.write("Вырезанные номера:")

        for i, cropped_image in enumerate(cropped_images):
            st.image(cropped_image, caption=f'Найденный номер {i + 1}', use_column_width=True)

            # Detect characters in the cropped image
            char_results = detect_characters(cropped_image)

            # Extract and display detected characters
            pred_texts = ''.join(letters[cls] for _, cls in char_results)
            st.write(f"<span style='font-size: 46px;'>{pred_texts}</span>", unsafe_allow_html=True)


