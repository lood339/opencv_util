# matplotlit util

import matplotlib.pyplot as plt
import cv2 as cv

def vis_images(images, cols):
    """
    :param images: list of images
    :param cols:
    :return:
    """
    assert cols > 0
    N = len(images)
    rows = N//cols
    if N%cols != 0:
        rows += 1
    fig = plt.figure(figsize=(rows,cols))
    for r in range(rows):
        for c in range(cols):
            index = r * cols + c
            if index < N:
                fig.add_subplot(rows, cols, index + 1)
                plt.imshow(images[index])
                plt.axis('equal')
                plt.axis('off')
    plt.show()

def ut_vis_images():
    import random

    im = cv.imread('/Users/jimmy/Desktop/test_image.png', 1)
    h, w = im.shape[0], im.shape[1]
    patch_size = 128
    N, M = 2, 5
    images = []
    for i in range(N):
        for j in range(M):
            x = random.randint(patch_size, w - patch_size)
            y = random.randint(patch_size, h - patch_size)
            patch = im[y:patch_size + y, x:patch_size + x, :]
            images.append(patch)
    vis_images(images, 5)

if __name__ == '__main__':
    ut_vis_images()



