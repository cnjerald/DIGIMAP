# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt
import numpy as np

import random
import cv2

# %% [markdown]
# # Load images

# %%
train_path = './seg_train/'
test_path = './seg_test/'

# %%
def randomBlackPatch(img):
    patchSize = random.randint(0,38)
    x, y = img.shape[:2]  # Get image dimensions
    x_start = np.random.randint(0, x - patchSize)
    y_start = np.random.randint(0, y - patchSize)
    img[x_start:x_start + patchSize, y_start:y_start + patchSize] = 0
    return img

# %%
def rotateImg(img):
    angle = random.randint(0, 45)
    
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    
    return rotated_img


# %%
def shiftImg(img):
  configuration = random.choice(["sideward","upward","na"])
  magnitude = random.randint(0,10)
  height, width = img.shape[:2]

  if(configuration == "sideward"):
    M = np.float32([[1,0,magnitude],[0,1,0]])
    dst = cv2.warpAffine(img,M,(width,height))
    return dst
  elif(configuration=="upward"):
    M = np.float32([[1,0,0],[0,1,-magnitude]])
    dst = cv2.warpAffine(img,M,(width,height))
    return dst
  else:
      return img


# %%
def flipImg(img):
    flip_code = random.randint(-1,2)
    if(flip_code == 2):
        return img
    else:
    # flip_code: 0 for vertical, 1 for horizontal, -1 for both
        flipped_img = cv2.flip(img, flip_code)
        return flipped_img

# %%
def reshape(img):
    resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    return resized_img
    

# %%
def combined_preprocessing_function(image):
    image = reshape(image)
    image = randomBlackPatch(image)
    image = rotateImg(image)
    image = shiftImg(image)
    image = flipImg(image)
    return image

# %%
# For training: Include augmentation and preprocessing

# Preprocessing tools. (flips ,rotation, and patch.)
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = combined_preprocessing_function,
    validation_split=0.2 
)

# Load data, reisze of 224, 224
train_data_gen = train_image_generator.flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=42
)

# For validation: No augmentation, only resizing and normalization
val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2  # Same split as training data
)

val_data_gen = val_image_generator.flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42
)


# %%
print("Class Indices:", train_data_gen.class_indices)
print("Classes:", list(train_data_gen.class_indices.keys()))

# %%
class_indices = train_data_gen.class_indices
reversed_class_indices = {v: k for k, v in class_indices.items()}
print(reversed_class_indices)

# %%
image,labels = next(train_data_gen)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(image[i].astype("uint8"))
    plt.axis('off')
    plt.title(f"Class: {reversed_class_indices[labels[i].argmax()]}")
plt.show()


# %%
image,labels = next(val_data_gen)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(image[i].astype("uint8"))
    plt.axis('off')
    plt.title(f"Class: {reversed_class_indices[labels[i].argmax()]}")
plt.show()


# %% [markdown]
# # Building the model

# %%
# Define the model

model = tf.keras.Sequential()

# Input layer
model.add(tf.keras.Input(shape=(224, 224, 3)))  # Input layer

# Layer 1
model.add(tf.keras.layers.Dense(6, activation='relu'))  # Dense layer with ReLU activation

# Layer 2
model.add(tf.keras.layers.Conv2D(128, kernel_size=[2,2], padding='valid', activation='relu'))

# Layer 3 pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2, padding='valid'))

# Layer 4
model.add(tf.keras.layers.Conv2D(64, kernel_size=[2,2], padding='valid', activation='relu'))

# Layer 5 pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2, padding='valid'))

# Layer 6
model.add(tf.keras.layers.Conv2D(32, kernel_size=[2,2], padding='valid', activation='relu'))

# Layer 7 pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2, padding='valid'))

# Layer 8
model.add(tf.keras.layers.Conv2D(32, kernel_size=[2,2], padding='valid', activation='relu'))

# Layer 9 pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2], strides=2, padding='valid'))

# Flatten before
model.add(tf.keras.layers.Flatten())

# Output layer
model.add(tf.keras.layers.Dense(6, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# %%
history = model.fit(train_data_gen,validation_data = val_data_gen, epochs = 50, verbose = 2)

# %%
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(

    
)


# Load the test data
test_data_gen = test_image_generator.flow_from_directory(
    directory=test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,  # Do not shuffle for evaluation
    seed=42
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data_gen, verbose=2)

# Print the test results
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")



