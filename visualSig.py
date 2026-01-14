import numpy as np

rgb_train = np.load("./res/unsw/RGB_TRAIN.npz")["patches"]
rgb_test = np.load("./res/unsw/RGB_TEST.npz")["patches"]

att_train = np.load("./res/unsw/IM.npz")["patches"]
att_test = np.load("./res/unsw/IM_TEST.npz")["patches"]

#CONVERT DTYPE TO NUMPY FLOAT32 TO COMPUTE INERTIA VIA NUMPY
att_train = att_train.astype(np.float32)
att_test = att_test.astype(np.float32)          
rgb_train = rgb_train.astype(np.float32)
rgb_test = rgb_test.astype(np.float32)

print("ATT_TRAIN:", att_train.shape, att_train.dtype)
print("ATT_TEST:", att_test.shape, att_test.dtype)
print("RGB_TRAIN:", rgb_train.shape, rgb_train.dtype)
print("RGB_TEST:", rgb_test.shape, rgb_test.dtype)
print(np.load("./res/unsw/RGB_TRAIN.npz")["labels"].shape)
print(np.load("./res/unsw/RGB_TEST.npz")["labels"].shape)
