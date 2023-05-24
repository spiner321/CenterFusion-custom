import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches



im = Image.open('/home/user/data/SanminKim/CenterFusion/data/nia/source/S_Clip_01685_05/Camera/CameraFront/blur/2-048_01685_CF_016.png')
fig, ax = plt.subplots()
ax.imshow(im)

xy = (0, 720.8474202081819)
width = 1398.4577846609773
height = 720.8474202081819 - 340.65140777551596

rect = patches.Rectangle(xy, width,height, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()


