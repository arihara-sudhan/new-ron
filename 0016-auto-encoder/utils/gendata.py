import numpy as np
import os 
from PIL import Image

"""
    PARAMS: Image read using Image of Pillow
    RETURNS: Noise Added Image
    GUIDE: Adjust SDEV for tuning noise as needed
"""
def add_noise(img):
    img_arr = np.array(img)
    MEAN, SDEV = 0, 500
    noise = np.random.normal(MEAN, SDEV, img_arr.shape)
    nimg_arr = img_arr + noise
    nimg_arr = np.clip(nimg_arr, 0, 255).astype(np.uint8)
    nimg = Image.fromarray(nimg_arr)
    return nimg

"""
    PARAMS: PATH where different types of images exist
    RETURNS: None
    DOES: Generates and saves noise added images in ../dataset/noises folder
"""
def generate_noises(PATH):
    if os.path.exists(PATH):
        FILES = os.listdir(PATH)
        for file in FILES:
            if file.endswith((".png", ".jpg", ".avif", ".gif", ".webp")):
                print(file)
                img = Image.open(PATH+"/"+file)
                nimg = add_noise(img)
                nimg.save(f"../dataset/noises/NOISE_{file}")

PATH = "../dataset/images"
generate_noises(PATH)
