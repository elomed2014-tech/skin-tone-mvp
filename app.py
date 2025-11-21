import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Skin Tone Matcher", page_icon="💄")

# --- БАЗА ДАННЫХ ---
COSMETIC_DATABASE = {
    "1. Ivory Light": (210, 180, 170),
    "2. Natural Beige": (170, 130, 110),
    "3. Warm Honey": (154, 103, 96),
    "4. Deep Bronze": (100, 70, 60)
}

LEFT_CHEEK_INDICES = [330, 347, 280]
RIGHT_CHEEK_INDICES = [101, 118, 50]

# --- ФУНКЦИИ (Те же, что и раньше) ---
def calculate_distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def find_closest_match(user_rgb):
    min_dist = float('inf')
    best_name = "Unknown"
    best_rgb = (0,0,0)
    for name, prod_rgb in COSMETIC_DATABASE.items():
        dist = calculate_distance(user_rgb, prod_rgb)
        if dist < min_dist:
            min_dist = dist
            best_name = name
            best_rgb = prod_rgb
    return best_name, best_rgb

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title(" Подбор тонального крема")
st.write("Сделайте фото при хорошем освещении.")

# Виджет камеры от Streamlit
img_file_buffer = st.camera_input("Сделайте селфи")

if img_file_buffer is not None:
    # Читаем загруженное фото
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # MediaPipe Init
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    # Обработка
    results = face_mesh.process(img_array)
    
    h, w, c = img_array.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Координаты щек
            def get_coords(indices):
                coords = []
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    coords.append([int(lm.x * w), int(lm.y * h)])
                return np.array(coords, np.int32)

            left_poly = get_coords(LEFT_CHEEK_INDICES)
            right_poly = get_coords(RIGHT_CHEEK_INDICES)

            # Маска и цвет
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [left_poly, right_poly], 255)
            mean_color = cv2.mean(img_array, mask=mask)
            
            u_r, u_g, u_b = int(mean_color[0]), int(mean_color[1]), int(mean_color[2])
            
            # Матчинг
            match_name, match_rgb = find_closest_match((u_r, u_g, u_b))

            # --- ВЫВОД РЕЗУЛЬТАТОВ ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Ваша кожа")
                # Рисуем цветной квадрат средствами Streamlit
                st.color_picker("Определенный цвет", f"#{u_r:02x}{u_g:02x}{u_b:02x}", disabled=True)
            
            with col2:
                st.header("Рекомендация")
                # Рисуем цвет продукта
                st.color_picker("Цвет продукта", f"#{match_rgb[0]:02x}{match_rgb[1]:02x}{match_rgb[2]:02x}", disabled=True)
            
            st.success(f"Вам подходит: **{match_name}**")

            # Рисуем зоны на фото для наглядности
            cv2.polylines(img_array, [left_poly, right_poly], True, (0, 255, 0), 2)
            st.image(img_array, caption="Зоны анализа", use_column_width=True)
            
    else:
        st.error("Лицо не найдено. Попробуйте снова.")