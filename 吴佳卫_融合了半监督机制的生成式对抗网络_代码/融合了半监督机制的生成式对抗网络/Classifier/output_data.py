import numpy as np
from PIL import Image


def output(output_dir0, output_dir1, images, labels):
    print("Writing Data!")
    img_size = np.shape(images)[0]
    for i in range(img_size):
        if i % 100 == 0:
            print("Image ", i)
        im = Image.fromarray(np.uint8(images[i]))
        if labels[i] == 0:
            if i <10:
                im.save(output_dir0 + "/000" + str(i) + ".jpg")
            elif 10 <= i < 100:
                im.save(output_dir0 + "/00" + str(i) + ".jpg")
            elif 100 <= i < 1000:
                im.save(output_dir0 + "/0" + str(i) + ".jpg")
            elif i >= 1000:
                im.save(output_dir0 + "/" + str(i) + ".jpg")
        else:
            if i < 10:
                im.save(output_dir1 + "/000" + str(i) + ".jpg")
            elif 10 <= i < 100:
                im.save(output_dir1 + "/00" + str(i) + ".jpg")
            elif 100 <= i < 1000:
                im.save(output_dir1 + "/0" + str(i) + ".jpg")
            elif i >= 1000:
                im.save(output_dir1 + "/" + str(i) + ".jpg")

