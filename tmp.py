import numpy as np
from PIL import Image


x = np.fromfile("data" ,dtype=np.uint8)
print(x.shape)
x = x.reshape(2, 40, 40)


im = Image.fromarray(x[0])
im.save('tmp.png')