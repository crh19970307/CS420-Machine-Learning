"""
This file implements the preprocessing steps on the given MNIST dataset.
"""
import numpy as np
from PIL import Image
import cv2
import os


# save the array to file as a image
def save_as_image(data, filename):
	im = Image.fromarray(data)
	im.save(filename)

# save the array to file as 0/1 text
def save_as_txt(data, filename):
	np.savetxt(filename, data, fmt='%i')

# search all pixels of one connected component
def dfs_search_block(arr, x, y):
	vis[x][y] = True
	if arr[x][y] == 0:	# if the current pixel is black, skip
		return 0
	b_size = 1
	for i in range(8):		# enumerate the 8 adjacent pixels
		new_x = x + dx[i]
		new_y = y + dy[i]
		if new_x >= 0 and new_x < fig_w and new_y >= 0 and new_y < fig_w and arr[new_x][new_y] != 0 \
		and not vis[new_x][new_y]:		# the adjacent pixel is legal
			b_size += dfs_search_block(arr, new_x, new_y)	# recursively search on the adjacent pixel
	return b_size

# color the connected component into black using floodfill
def dfs_remove_block(arr, x, y):
	arr[x][y] = 0
	for i in range(8):		# enumerate the 8 adjacent pixels
		new_x = x + dx[i]
		new_y = y + dy[i]
		if new_x >= 0 and new_x < fig_w and new_y >= 0 and new_y < fig_w and arr[new_x][new_y] != 0:
			dfs_remove_block(arr, new_x, new_y)		# recursively color the adjacent pixel


if __name__ == '__main__':
	n_samples = 60000
	fig_w = 45

	# define the relative coordinates of the 8 adjacent pixels
	dx = [-1, 0, 1, 0, -1, 1, 1, -1]
	dy = [0, -1, 0, 1, -1, -1, 1, 1]

	# load dataset from files
	data = np.fromfile("../mnist_train/mnist_train_data",dtype=np.uint8)
	X_train = data.reshape(n_samples, -1)
	data = data.reshape(n_samples, fig_w, fig_w)
	Y_train = np.fromfile("../mnist_train/mnist_train_label",dtype=np.uint8)
	data = np.fromfile("../mnist_test/mnist_test_data",dtype=np.uint8).reshape(10000, fig_w, fig_w)
	X_test = np.fromfile("../mnist_test/mnist_test_data",dtype=np.uint8).reshape(10000, -1)
	Y_test = np.fromfile("../mnist_test/mnist_test_label" ,dtype=np.uint8)

	images = np.zeros((10000, 45 * 45), dtype=np.uint8)

	# enumerate each image in the testing set
	for id in range(10000):

		image = data[id]
		# save the original image to disk
		# save_as_image(image, str(id) + '_original.png')

		arr = np.asarray(image)

		vis = [[False for i in range(fig_w)] for j in range(fig_w)]

		# enumerate each pixel in the image
		for i in range(fig_w):
			for j in range(fig_w):
				if vis[i][j]:
					continue
				# if a white pixel is found, search for the entire white block
				block_size = dfs_search_block(arr, i, j)
				# if the block contains more than 20 pixels, then it is a noise block
				if block_size > 0 and block_size <= 20:
					dfs_remove_block(arr, i, j)

		# save the denoised result to disk
		# save_as_image(arr, str(id) + '_denoised.png')

		# calculate the up/down left/right paddings
		sum_rows = np.sum(arr, axis=1)
		upper = np.where(sum_rows > 0)[0][0]
		lower = np.where(sum_rows > 0)[0][-1]

		sum_cols = np.sum(arr, axis=0)
		left = np.where(sum_cols > 0)[0][0]
		right = np.where(sum_cols > 0)[0][-1]

		top_padding = (45 - lower + upper - 1) // 2
		bottom_padding = 45 - (top_padding + lower - upper + 1)

		left_padding = (45 - right + left - 1) // 2
		right_padding = 45 - (left_padding + right - left + 1)

		# crop out the digit area from the image
		arr = arr[upper: lower + 1, left: right +1]

		# save the cropped result to disk
		# save_as_image(arr, str(id) + '_cropped.png')

		# add blacking padding to the digit
		arr = cv2.copyMakeBorder(arr, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT,value=0)
		# save_as_image(arr, str(id) + '_centered.png')

		os._exit(0)

		arr = arr.reshape(1, 45 * 45)
		images[id] = arr
		if id % 100 == 0:
			print(id, ' done')
	# save the processed dataset to disk
	images.tofile('new_test_data')

