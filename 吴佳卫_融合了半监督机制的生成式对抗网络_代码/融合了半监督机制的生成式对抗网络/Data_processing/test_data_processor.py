import os
from PIL import Image

load_file_path1 = "../Data/Data_test/raw_data/A"
save_file_path1 = "../Data/Data_test/processed_data/A"
load_file_path2 = "../Data/Data_test/raw_data/B"
save_file_path2 = "../Data/Data_test/processed_data/B"

types = "jpg"
for root, dirs, files in os.walk(load_file_path1):
    img_cnt = 0
    for f in files:
        if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[1][1:] in types:
            n_path = os.path.join(root, f)
            image = Image.open(n_path)
            image = image.resize((168, 168))
            if img_cnt % 10:
                print("Image ", img_cnt)
            crop_cnt = 0
            for i in range(0, 121, 40):
                for j in range(0, 121, 40):
                    crop_cnt += 1
                    box = (j, i, j + 48, i + 48)
                    roi = image.crop(box)
                    if img_cnt < 10:
                        if crop_cnt < 10:
                            s_path = save_file_path1 + "/" + "000" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path1 + "/" + "000" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"
                    elif 10 <= img_cnt < 100:
                        if crop_cnt < 10:
                            s_path = save_file_path1 + "/" + "00" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path1 + "/" + "00" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"
                    elif 100 <= img_cnt < 1000:
                        if crop_cnt < 10:
                            s_path = save_file_path1 + "/" + "0" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path1 + "/" + "0" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"
                    elif img_cnt >= 1000:
                        if crop_cnt < 10:
                            s_path = save_file_path1 + "/" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path1 + "/" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"

                    roi.save(s_path)

            img_cnt += 1

for root, dirs, files in os.walk(load_file_path2):
    img_cnt = 0
    for f in files:
        if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[1][1:] in types:
            n_path = os.path.join(root, f)
            image = Image.open(n_path)
            image = image.resize((168, 168))
            if img_cnt % 10 :
                print("Image ", img_cnt)
            crop_cnt = 0
            for i in range(0, 121, 40):
                for j in range(0, 121, 40):
                    crop_cnt += 1
                    box = (j, i, j + 48, i + 48)
                    roi = image.crop(box)
                    if img_cnt < 10:
                        if crop_cnt < 10:
                            s_path = save_file_path2 + "/" + "000" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path2 + "/" + "000" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"
                    elif 10 <= img_cnt < 100:
                        if crop_cnt < 10:
                            s_path = save_file_path2 + "/" + "00" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path2 + "/" + "00" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"
                    elif 100 <= img_cnt < 1000:
                        if crop_cnt < 10:
                            s_path = save_file_path2 + "/" + "0" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path2 + "/" + "0" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"
                    elif img_cnt >= 1000:
                        if crop_cnt < 10:
                            s_path = save_file_path2 + "/" + str(img_cnt) + "_0" + str(crop_cnt) + ".jpg"
                        else:
                            s_path = save_file_path2 + "/" + str(img_cnt) + "_" + str(crop_cnt) + ".jpg"

                    roi.save(s_path)

            img_cnt += 1


