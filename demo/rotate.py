import cv2

image_path = './P0217__1024__541___401.png'
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

for i in range(8):
    M = cv2.getRotationMatrix2D(center, i * 45, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(f'P0217__1024__541___401_{i}.png', rotated)