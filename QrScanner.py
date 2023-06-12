import cv2
import numpy as np
from pyzbar import pyzbar


img = cv2.imread(r"/home/ubuntu/webapp/qr.jpg")


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# for contour in contours:
#     (x, y, w, h) = cv2.boundingRect(contour)
#     crop_img = img[y:y+h, x:x+w]
#     decoded = pyzbar.decode(crop_img)

#     if decoded:
#         print("QR code data:", decoded[0].data.decode('utf-8'))
#         print("QR code type:", decoded[0].type)
#         print("QR code location:", (x, y, w, h))

print("QR code data:", pyzbar.decode(img))