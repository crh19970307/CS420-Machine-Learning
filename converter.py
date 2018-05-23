import numpy as np
from PIL import Image
import cv2


def save_as_image(data, filename):
    im = Image.fromarray(data)
    im.save(filename)

def save_as_txt(data, filename):
    np.savetxt(filename, data, fmt='%i')


def dfs_search_block(arr, x, y):
    vis[x][y] = True
    if arr[x][y] == 0:
        return 0
    b_size = 1
    for i in range(8):
        new_x = x + dx[i]
        new_y = y + dy[i]
        if new_x >= 0 and new_x < fig_w and new_y >= 0 and new_y < fig_w and arr[new_x][new_y] != 0 \
        and not vis[new_x][new_y]:
            b_size += dfs_search_block(arr, new_x, new_y)
    return b_size


def dfs_remove_block(arr, x, y):
    arr[x][y] = 0
    for i in range(8):
        new_x = x + dx[i]
        new_y = y + dy[i]
        if new_x >= 0 and new_x < fig_w and new_y >= 0 and new_y < fig_w and arr[new_x][new_y] != 0:
            dfs_remove_block(arr, new_x, new_y)


n_samples = 60000
fig_w = 45

dx = [-1, 0, 1, 0, -1, 1, 1, -1]
dy = [0, -1, 0, 1, -1, -1, 1, 1]

data = np.fromfile("mnist_train\\mnist_train_data",dtype=np.uint8)
X_train = data.reshape(n_samples, -1)
data = data.reshape(n_samples, fig_w, fig_w)

Y_train = np.fromfile("mnist_train\\mnist_train_label",dtype=np.uint8)

data = np.fromfile("mnist_test\\mnist_test_data",dtype=np.uint8).reshape(10000, fig_w, fig_w)

X_test = np.fromfile("mnist_test\\mnist_test_data",dtype=np.uint8).reshape(10000, -1)
Y_test = np.fromfile("mnist_test\\mnist_test_label" ,dtype=np.uint8)

images = np.zeros((10000, 45 * 45), dtype=np.uint8)

for id in range(10000):

    image = data[id]
    # save_as_image(image, 'original.png')

    arr = np.asarray(image)
    # save_as_txt(arr, 'original.txt')

    vis = [[False for i in range(fig_w)] for j in range(fig_w)]

    for i in range(fig_w):
        for j in range(fig_w):
            if vis[i][j]:
                continue
            block_size = dfs_search_block(arr, i, j)
            if block_size > 0 and block_size <= 20:
                dfs_remove_block(arr, i, j)
                # print("Remove block of size ", block_size)

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
    # print(top_padding, bottom_padding, left_padding, right_padding)

    arr = arr[upper: lower + 1, left: right +1]
    arr = cv2.copyMakeBorder(arr, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT,value=0)

    arr = arr.reshape(1, 45 * 45)
    images[id] = arr
    if id % 100 == 0:
        print(id, ' done')
    
images.tofile('new_test_data')



# save_as_image(arr, 'new.png')
# save_as_txt(arr, 'new.txt')

