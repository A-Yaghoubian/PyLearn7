import cv2
import os

os.makedirs('./mnist', exist_ok=True)

mnist_image = cv2.imread('mnist.png')
x, y, _ = mnist_image.shape # 1000, 2000

for ix in range(0, x, 20):
    num = ix // 100 
    os.makedirs(f'./mnist/{num}', exist_ok=True)
    
    for iy in range(0, y, 20):
        cv2.imwrite(f'mnist/{num}/{ix}{iy}.png', mnist_image[ix:ix+20, iy:iy+20])
        