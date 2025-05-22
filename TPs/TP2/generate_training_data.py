import os
import numpy as np
from PIL import Image, ImageDraw
import random

STYLES = ["stripes", "dots", "gradient", "random_shapes"]
IMAGES_PER_STYLE = 50
IMG_SIZE = 128
OUTPUT_DIR = "training_data"


def generate_training_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for style in STYLES:
        style_dir = os.path.join(OUTPUT_DIR, style)
        os.makedirs(style_dir, exist_ok=True)

        for i in range(IMAGES_PER_STYLE):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
            draw = ImageDraw.Draw(img)

            if style == "stripes":
                for y in range(0, IMG_SIZE, 10):
                    color = random.choice(["red", "blue", "green"])
                    draw.rectangle([0, y, IMG_SIZE, y + 5], fill=color)

            elif style == "dots":
                for _ in range(50):
                    x, y = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
                    r = random.randint(3, 8)
                    color = random.choice(["red", "green", "blue"])
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

            elif style == "gradient":
                base_color = np.random.randint(0, 255, size=3)
                for y in range(IMG_SIZE):
                    shade = tuple((base_color * (y / IMG_SIZE)).astype(np.uint8))
                    draw.line([0, y, IMG_SIZE, y], fill=shade)

            elif style == "random_shapes":
                number_of_shapes = np.random.randint(15, 30)
                for _ in range(number_of_shapes):
                    x1, y1 = np.random.randint(0, IMG_SIZE, size=2)
                    x2, y2 = np.random.randint(0, IMG_SIZE, size=2)
                    color = tuple(np.random.randint(0, 255, size=3))
                    x0, x1 = sorted([x1, x2])
                    y0, y1 = sorted([y1, y2])
                    draw.rectangle([x0, y0, x1, y1], fill=color)

            path = os.path.join(style_dir, f"{i:02}.png")
            img.save(path)

generate_training_data()