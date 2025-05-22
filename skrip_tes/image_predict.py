from ultralytics import YOLO
import os
import cv2

model = YOLO("runs/test daw/count.pt")# path ke file model

image_folder = "test image/1. Gambar janjang buah sawit di TPH (Minimal 40 Gambar)" #path ke folder gambar untuk dites prediksi
output_folder = "output_image"

existing_folders = os.listdir(output_folder)
folder_numbers = [int(folder.split('_')[-1]) for folder in existing_folders if folder.startswith("model_")]
new_folder_number = max(folder_numbers, default=0) + 1
model_output_folder = os.path.join(output_folder, f"model_{new_folder_number}")
os.makedirs(model_output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)
        print(f"Processing {filename}...")

        results = model(
            source=image_path,
            conf=0.4,
            save=True,
            project=model_output_folder,
            name="result",
            exist_ok=True
        )

        predicted_image = os.path.join(model_output_folder, "result", filename)
        img = cv2.imread(predicted_image)

        jumlah = 0
        for box in results[0].boxes:
            jumlah += 1
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)

        text = f"Total Deteksi: {jumlah} kelapa sawit"
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 6.0
        font_thicness = 10
        text_size = cv2.getTextSize(text, font, font_scale, font_thicness)[0]
        text_width, text_height = text_size

        background_color = (0, 0, 0)
        text_x = 10
        text_y = 200
        cv2.rectangle(img, (text_x, text_y - text_height), (text_x + text_width, text_y + 10), background_color, -1)

        text_color = (0, 255, 0)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thicness)

        cv2.imwrite(predicted_image, img)