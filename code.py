# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Input, Conv2DTranspose
from skimage.io import imread, imshow
from skimage.transform import resize
from tqdm import tqdm

# We have multiple mask for a single image in each folder of a image mask.
# There is a mask for each nuclie of the picture in mask folder
# We need to combine all the mask to make a single mask for a image. 

Train_path = "Nuclie_dataset/stage1_train"
Test_path = "Nuclie_dataset/stage1_test"

# number of training images
n = len(os.listdir(Train_path))

Train_img_dir = os.listdir(Train_path)

#x = images , y = mask

Img_height = 128
Img_width = 128
color_channel = 3

# Now we are creating an X_train array of [670, 128, 128, 3] to store each image.
X_train = np.zeros((n, Img_height, Img_width, color_channel),dtype=np.uint8)

# We need an array for masks also
Y_train = np.zeros((n, Img_height, Img_width, 1), dtype=bool)

for i in tqdm(range(n)):
    img_path = Train_path + "/" + Train_img_dir[i] + "/images/"
    img_name = os.listdir(img_path)[0]
    img = imread(img_path + "/" + img_name)[:,:,:color_channel]
    img = resize(img, (Img_height, Img_width), 
                 mode = "constant", preserve_range=True)
    X_train[i] = img
    
    mask_path = Train_path + "/" + Train_img_dir[i] + "/masks/"
    mask_n = os.listdir(mask_path)
    mask = np.zeros([Img_height, Img_width, 1], dtype=bool)
    
    for j in range(len(mask_n)):
        mask_img = imread(mask_path + "/" + mask_n[j])
        mask_img = np.expand_dims(resize(mask_img,(Img_height, Img_width),
                                         mode="constant",preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_img)
    Y_train[i] = mask    
    
    
# checking train images and masks

imshow(X_train[0])
imshow(Y_train[0])


# number of testing images
test_n = len(os.listdir(Test_path))
Test_img_dir = os.listdir(Test_path)
X_test = np.zeros((test_n, Img_height, Img_width, color_channel),dtype=np.uint8)

# loading of testing data

for i in tqdm(range(test_n)):
    img_path = Test_path + "/" + Test_img_dir[i] + "/images/"
    img_name = os.listdir(img_path)[0]
    img = imread(img_path + "/" + img_name)[:,:,:color_channel]
    img = resize(img, (Img_height, Img_width), 
                 mode = "constant", preserve_range=True)
    X_test[i] = img
    
#Unet Architechture

inputs  = Input((Img_height, Img_width, color_channel))
x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#Downsampling = Applying CNN

c1 = Conv2D(16,(3,3), activation="relu", padding="same")(x)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16,(3,3),activation="relu", padding="same")(c1)
p1 = MaxPool2D((2,2))(c1)

c2 = Conv2D(32,(3,3), activation="relu", padding="same")(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32,(3,3),activation="relu", padding="same")(c2)
p2 = MaxPool2D((2,2))(c2) 

c3 = Conv2D(64,(3,3), activation="relu", padding="same")(p2)
c3 = Dropout(0.1)(c3)
c3 = Conv2D(64,(3,3),activation="relu", padding="same")(c3)
p3 = MaxPool2D((2,2))(c3) 

c4 = Conv2D(128, (3,3), activation="relu", padding="same")(p3)
c4 = Dropout(0.1)(c4)
c4 = Conv2D(128, (3,3),activation="relu", padding="same")(c4)
p4 = MaxPool2D((2,2))(c4) 

c5 = Conv2D(256, (3,3), activation="relu", padding="same")(p4)
c5 = Dropout(0.1)(c5)
c5 = Conv2D(256, (3,3),activation="relu", padding="same")(c5) 

#Upsampling
#Conv2DTranspose is for upsampling(crop and copy) 

from tensorflow.keras.layers import concatenate as Concat

u6 = Conv2DTranspose(128,(2,2), strides=(2,2), padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = Conv2D(128,(3,3), activation="relu", padding="same")(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3,3), activation="relu", padding ="same")(c6)

u7 = Conv2DTranspose(64,(2,2), strides=(2,2), padding="same")(c6)
u7 = Concat([u7,c3])
c7 = Conv2D(64,(3,3), activation="relu", padding="same")(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64,(3,3), activation="relu", padding ="same")(c7)

u8 = Conv2DTranspose(32,(2,2), strides=(2,2), padding="same")(c7)
u8 = Concat([u8,c2])
c8 = Conv2D(32,(3,3), activation="relu", padding="same")(u8)
c8 = Dropout(0.2)(c8)
c8 = Conv2D(32, (3,3), activation="relu", padding ="same")(c8)

u9 = Conv2DTranspose(16,(2,2), strides=(2,2), padding="same")(c8)
u9 = Concat([u9,c1])
c9 = Conv2D(16,(3,3), activation="relu", padding="same")(u9)
c9 = Dropout(0.2)(c9)
c9 = Conv2D(16, (3,3), activation="relu", padding ="same")(c9)

output = Conv2D(1,(1,1), activation="sigmoid")(c9)

from tensorflow.keras import Model 
model = Model(inputs=[inputs], outputs = output)

model.summary()

tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Including checkpoints
checkpoint = tf.keras.callbacks.ModelCheckpoint('model_ckpt.h5', 
                                   save_best_only=True, verbose=1)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.TensorBoard(log_dir="logs"), checkpoint]


results = model.fit(X_train, Y_train, validation_split=0.1,
                    batch_size=16, epochs=20, callbacks=callbacks)




# Test model
y_pred_mask = model.predict(X_test)

imshow(X_test[0])
imshow(y_pred_mask[0])

model.save("Unet_model_nuclie_segmentation.h5")
model.save()



    
    
    
    
    
    
    
    
    
    