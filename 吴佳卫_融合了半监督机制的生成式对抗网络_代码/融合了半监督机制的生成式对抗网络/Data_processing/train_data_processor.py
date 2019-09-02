import os
import random
from PIL import Image

name1 = "../Data/Data_raw/A"
name2 = "../Data/Data_processed/A/test"
name3 = "../Data/Data_raw/B"
name4 = "../Data/Data_processed/orange/B"
types = "jpg"

for root, dirs, files in os.walk(name1):
    for f in files:
        if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[1][1:] in types:
            n_path = os.path.join(root, f)
            image = Image.open(n_path)
            image = image.resize((168, 168))

            i = 0
            for i in range(1, 6):
                x = random.randint(0, 120)
                y = random.randint(0, 120)
                print('x:', x, 'y:', y)
                box = (x, y, x + 48, y + 48)
                roi = image.crop(box)
                s_path = os.path.join(name2, str(i)+f)
                roi.save(s_path)

for root, dirs, files in os.walk(name3):
    for f in files:
        if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[1][1:] in types:
            n_path = os.path.join(root, f)
            image = Image.open(n_path)
            image = image.resize((168, 168))

            i = 0
            for i in range(1, 6):
                x = random.randint(0, 120)
                y = random.randint(0, 120)
                print('x:', x, 'y:', y)
                box = (x, y, x + 48, y + 48)
                roi = image.crop(box)
                s_path = os.path.join(name4, str(i)+f)
                roi.save(s_path)
