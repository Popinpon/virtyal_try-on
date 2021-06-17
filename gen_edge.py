from PIL.Image import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torchvision.models as models
import torch




id="000005"
path="dataset/"
img = cv2.imread(path+"test_clothes/"+id+"_2.jpg")

OLD_IMG = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)
SIZE = (1, 65)
bgdModle = np.zeros(SIZE, np.float64)

fgdModle = np.zeros(SIZE, np.float64)
rect = (1, 1, img.shape[1], img.shape[0])
cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')[:, :, np.newaxis]


img *= mask2
back=(np.full(img.shape,255,dtype='uint8')*(1-mask2))
img=img+back


#img=background_seg(img)

plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.bitwise_not(back), cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])
cv2.imwrite(path+"test_edge/"+id+"_1.jpg",cv2.bitwise_not(back))
cv2.imwrite(path+"test_clothes/"+id+"_1.jpg",img)

plt.show()