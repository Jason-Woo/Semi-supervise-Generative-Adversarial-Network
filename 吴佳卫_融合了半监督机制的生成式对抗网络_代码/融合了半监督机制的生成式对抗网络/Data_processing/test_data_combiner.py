import os
from PIL import Image
import cv2
import numpy as np

load_file_path = "../Data/Data_test/cycle_GAN_result/A"
save_file_path = "../Data/Data_test/final_result/A"

# 参数初始化
all_path = []
final_img_num = 0


def comb1(img1, img2):
    res = np.zeros([48, 8, 3], np.uint8)
    for row in range(0, 48):
        for col in range(0, 8):
            srcImgLen = float(abs(col))
            testImgLen = float(abs(col - 8))
            alpha = srcImgLen / (srcImgLen + testImgLen)
            res[row, col] = np.clip(np.asanyarray(img1)[row, col] * (1-alpha) + np.asanyarray(img2)[row, col] * alpha, 0, 255)

    img = Image.fromarray(np.uint8(res))
    return img


def comb2(img1, img2):
    res = np.zeros([8, 168, 3], np.uint8)
    for col in range(0, 168):
        for row in range(0, 8):
            srcImgLen = float(abs(row))
            testImgLen = float(abs(row - 8))
            alpha = srcImgLen / (srcImgLen + testImgLen)
            res[row, col] = np.clip(np.asanyarray(img1)[row, col] * (1-alpha) + np.asanyarray(img2)[row, col] * alpha, 0, 255)

    img = Image.fromarray(np.uint8(res))
    return img


for root, dirs, files in os.walk(load_file_path):
    for file in files:
        if "jpg" in file:
            all_path.append(os.path.join(root, file))

if len(all_path) % 16 != 0:
    print("ERROR!")
else:
    img_num = len(all_path) / 16
    final_img_num = int(img_num)

for n in range(0, final_img_num):

    images1 = [Image.new('RGB', (168, 48)) for x in range(4)]
    images2 = [Image.new('RGB', (8, 48)) for x in range(8)]
    images3 = [Image.new('RGB', (168, 8)) for x in range(8)]
    toImage = Image.new('RGB', (168, 168))
    for i in range(0, 4):
        for j in range(0, 4):
            num = 4 * i + j + 16 * n
            if num < 10:
                img_name = "000" + str(num)
            elif 10 <= num < 100:
                img_name = "00" + str(num)
            elif 100 <= num < 1000:
                img_name = "0" + str(num)
            elif num >= 1000:
                img_name = str(num)
            img_path = load_file_path + "/" + img_name + ".jpg"

            tmp_pic = Image.open(img_path)
            box1 = (0, 0, 8, 48)
            box2 = (40, 0, 48, 48)
            images2[2 * j] = tmp_pic.crop(box1)
            images2[2 * j + 1] = tmp_pic.crop(box2)

            images1[i].paste(tmp_pic, (j * 40, 0))

        images1[i].paste(comb1(images2[1], images2[2]), (40, 0))
        images1[i].paste(comb1(images2[3], images2[4]), (80, 0))
        images1[i].paste(comb1(images2[5], images2[6]), (120, 0))

        box3 = (0, 0, 168, 8)
        box4 = (0, 40, 168, 48)
        images3[2 * i] = images1[i].crop(box3)
        images3[2 * i + 1] = images1[i].crop(box4)
        toImage.paste(images1[i], (0, i * 40))

    toImage.paste(comb2(images3[1], images3[2]), (0, 40))
    toImage.paste(comb2(images3[3], images3[4]), (0, 80))
    toImage.paste(comb2(images3[5], images3[6]), (0, 120))
    
    img = cv2.cvtColor(np.asarray(toImage), cv2.COLOR_RGB2BGR)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 1, 10, 7, 21)
    save_name = save_file_path + "/" + str(n) + ".jpg"
    cv2.imwrite(save_name, dst)
    print("Image ", n)

