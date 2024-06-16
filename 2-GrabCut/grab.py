import numpy as np
import cv2

from matplotlib import pyplot as plt
from grabcut import GrabCut

# def checkout(value,limit):
# 	return min(max(value,0),limit)

# 返回两个点中x较小的坐标
def minx(x1,x2,y1,y2):
	if x1<x2:
		return x1,y1
	else:
		return x2,y2

# 检查x是否超出边界P
def checkout(x,p):
	if x>p:
		return p
	if x<0:
		return 0
	return x

# ------执行GrabCut图像分割算法
def algo_grabcut(filename,foreground=[],background=[],pos1x=1,pos1y=1,pos2x=511,pos2y=511,times=5, algo=True):
	img = cv2.imread(filename)
	print(f"Reading image from: {filename}")
	if img is None:
		raise ValueError(f"Cannot open or read image file: {filename}")

	mask = np.zeros(img.shape[:2],np.uint8)

	height = img.shape[:2][0]
	width = img.shape[:2][1]
	pos1x = checkout(pos1x,width-1)
	pos1y = checkout(pos1y,height-1)
	pos2x = checkout(pos2x,width-1)
	pos2y = checkout(pos2y,height-1)
	mask[min(pos1y,pos2y):max(pos1y,pos2y)+1,min(pos1x,pos2x):max(pos1x,pos2x)+1]=3

	for y1,x1,y2,x2 in foreground:
		x1 = checkout(x1,height-1)
		y1 = checkout(y1,width-1)
		x2 = checkout(x2,height-1)
		y2 = checkout(y2,width-1)
		if x1==x2:
				mask[x1,min(y1,y2):max(y1,y2)+1] = 1
		else:
			k = (y1-y2)/(x1-x2)
			x,y = minx(x1,x2,y1,y2)
			while True:
				mask[x,y] = 1
				x = x+1
				y = checkout(int(round(y+k)),width-1)
				if x>max(x1,x2):
					break

	for y1,x1,y2,x2 in background:
		x1 = checkout(x1,height-1)
		y1 = checkout(y1,width-1)
		x2 = checkout(x2,height-1)
		y2 = checkout(y2,width-1)
		if x1==x2:
				mask[x1,min(y1,y2):max(y1,y2)+1] = 0
		else:
			k = (y1-y2)/(x1-x2)
			x,y = minx(x1,x2,y1,y2)
			while True:
				mask[x,y] = 0
				x = x+1
				y = checkout(int(round(y+k)),height-1)
				if x>max(x1,x2):
					break
	if algo:
		rect = (0, 0, 0, 0)
		bgdModel = np.zeros((1, 65), np.float64)
		fgdModel = np.zeros((1, 65), np.float64)
		cv2.grabCut(img, mask, rect, bgdModel, fgdModel, times, cv2.GC_INIT_WITH_MASK)
		mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
		im = img * mask2[:, :, np.newaxis]
		im += 255 * (1 - cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR))
		cv2.imwrite('out.png', im)

	else:
		mask1 = np.zeros(img.shape[:2],np.uint8)
		mask1[mask==1] = 3
		mask1[mask==2] = 1
		mask1[mask==3] = 2
		GrabCut(filename, mask1, 1)
		return True
	# 显示分割后的图像
	cv2.imshow("Segmented Image", im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os
#
#
# def algo_grabcut(filename, foreground=[], background=[], pos1x=1, pos1y=1, pos2x=511, pos2y=511, times=5, algo=True):
# 	# 输出文件路径调试信息
# 	print(f"Checking file at: {filename}")
#
# 	# 检查文件是否存在
# 	if not os.path.exists(filename):
# 		raise ValueError(f"File does not exist: {filename}")
#
# 	# 读取图像
# 	img = cv2.imread(filename)
#
# 	# 输出读取图像调试信息
# 	print(f"Reading image from: {filename}")
#
# 	# 检查图像是否读取成功
# 	if img is None:
# 		raise ValueError(f"Cannot open or read image file: {filename}")
#
# 	mask = np.zeros(img.shape[:2], np.uint8)
#
# 	height = img.shape[:2][0]
# 	width = img.shape[:2][1]
# 	pos1x = checkout(pos1x, width - 1)
# 	pos1y = checkout(pos1y, height - 1)
# 	pos2x = checkout(pos2x, width - 1)
# 	pos2y = checkout(pos2y, height - 1)
# 	mask[min(pos1y, pos2y):max(pos1y, pos2y) + 1, min(pos1x, pos2x):max(pos1x, pos2x) + 1] = 3
#
# 	for y1, x1, y2, x2 in foreground:
# 		x1 = checkout(x1, height - 1)
# 		y1 = checkout(y1, width - 1)
# 		x2 = checkout(x2, height - 1)
# 		y2 = checkout(y2, width - 1)
# 		if x1 == x2:
# 			mask[x1, min(y1, y2):max(y1, y2) + 1] = 1
# 		else:
# 			k = (y1 - y2) / (x1 - x2)
# 			x, y = minx(x1, x2, y1, y2)
# 			while True:
# 				mask[x, y] = 1
# 				x = x + 1
# 				y = checkout(int(round(y + k)), width - 1)
# 				if x > max(x1, x2):
# 					break
#
# 	for y1, x1, y2, x2 in background:
# 		x1 = checkout(x1, height - 1)
# 		y1 = checkout(y1, width - 1)
# 		x2 = checkout(x2, height - 1)
# 		y2 = checkout(y2, width - 1)
# 		if x1 == x2:
# 			mask[x1, min(y1, y2):max(y1, y2) + 1] = 0
# 		else:
# 			k = (y1 - y2) / (x1 - x2)
# 			x, y = minx(x1, x2, y1, y2)
# 			while True:
# 				mask[x, y] = 0
# 				x = x + 1
# 				y = checkout(int(round(y + k)), height - 1)
# 				if x > max(x1, x2):
# 					break
#
# 	if algo:
# 		rect = (0, 0, 0, 0)
# 		bgdModel = np.zeros((1, 65), np.float64)
# 		fgdModel = np.zeros((1, 65), np.float64)
# 		cv2.grabCut(img, mask, rect, bgdModel, fgdModel, times, cv2.GC_INIT_WITH_MASK)
# 		mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# 		im = img * mask2[:, :, np.newaxis]
# 		im += 255 * (1 - cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR))
# 		cv2.imwrite('out.png', im)
# 	else:
# 		mask1 = np.zeros(img.shape[:2], np.uint8)
# 		mask1[mask == 1] = 3
# 		mask1[mask == 2] = 1
# 		mask1[mask == 3] = 2
# 		GrabCut(filename, mask1, 1)
#
# 	return True
#
#
# # 调用方法进行测试
# filename = "C:\\Users\\冯敏\\Desktop\\GrabCut-master\\image\\dsa.png"
# try:
# 	algo_grabcut(filename)
# except ValueError as e:
# 	print(e)
