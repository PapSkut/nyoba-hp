import os
import cv2
import numpy as np

result_folder = "output image/model_23/result" # path ke folder hasil prediksi
output_folder = "collage_output" #folder output untuk menyimpan hasil kolase
os.makedirs(output_folder, exist_ok=True)

resize_width = 800
resize_height = 600
grid_size = 5

images = []
for filename in os.listdir(result_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(result_folder, filename)
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (resize_width, resize_height))
        images.append(img_resized)

num_images = len(images)
num_columns = grid_size
num_rows = (num_images + grid_size - 1) // grid_size

grid_images = []
for i in range(num_rows):
    start_index = i * num_columns
    end_index = min((i + 1) * num_columns, num_images)
    row_images = images[start_index:end_index]
    
    if len(row_images) < num_columns:
        row_images += [np.zeros_like(row_images[0])] * (num_columns - len(row_images))
    
    grid_images.append(np.hstack(row_images))

collage = np.vstack(grid_images)

counter = 1
output_path = os.path.join(output_folder, f"collage_result_{counter}.jpg")
while os.path.exists(output_path):
    counter += 1
    output_path = os.path.join(output_folder, f"collage_result_{counter}.jpg")

cv2.imwrite(output_path, collage)