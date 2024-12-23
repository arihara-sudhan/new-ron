import os
from PIL import Image, ImageDraw
import random

def add_shape_to_image(image, shape="circle"):
    width, height = image.size
    draw = ImageDraw.Draw(image)
    shape_size = min(width, height) // 4
    center_x, center_y = width // 2 + random.choice([200,-250, 150]), height // 2 + random.choice([200,-200,-140])
    if shape == "circle":
        draw.ellipse((center_x - shape_size, center_y - shape_size, 
                      center_x + shape_size, center_y + shape_size), fill="black")
    elif shape == "square":
        draw.rectangle((center_x - shape_size, center_y - shape_size, 
                        center_x + shape_size, center_y + shape_size), fill="black")

def process_images(input_folder, output_folder, shape="circle"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path).convert("RGB")
            add_shape_to_image(image, shape=shape)
            output_path = os.path.join(output_folder, "MASK4-"+filename)
            image.save(output_path)

input_folder = 'clean'
output_folder = 'masked'
shape = random.choice(['circle', 'square'])
process_images(input_folder, output_folder, shape=shape)
