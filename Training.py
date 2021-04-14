# %%
# cargar imagenes
# sacado de: https://www.tensorflow.org/tutorials/load_data/images
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

batch_size = 32
img_height = 96
img_width = 128

train_ds = image_dataset_from_directory(
    "dataset_completo/Classify",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical",
)

val_ds = image_dataset_from_directory(
    "dataset_completo/Classify",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical",
)

class_names = train_ds.class_names
print(class_names)
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch[1])
    break

# %%
# visualizar data (Roto con el categorical)
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#  for i in range(9):
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")

# %%
# Performance del dataset
# sacado del tutorial de keras: https://www.tensorflow.org/tutorials/load_data/images
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# %%
# Definir y entrenar modelo
# modelo de: https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
from tensorflow.keras import layers, optimizers, constraints

num_classes = len(class_names)
model = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu", kernel_constraint=constraints.max_norm(3), padding="same"),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu", kernel_constraint=constraints.max_norm(3), padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation="relu", kernel_constraint=constraints.max_norm(3)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
# compile model
epochs = 25
learning_rate = 0.01
decay = learning_rate / epochs
sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds)

print(model.summary())
# %%
# cargar imagen con opencv y hacer un predict
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("dataset_completo/Classify/Meta/img_005.png")

# usando opencv (abre una ventana nueva)
# cv2.imshow("Test Image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# usando matplotlib (se ejecuta en ipython)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# print(img.shape)
# agregar el espacio de batches (1)
img = np.expand_dims(img, axis=0)
print(img.shape)

prediction = model.predict(img)
print(prediction)

index = np.argmax(prediction)
# print(index)

print(class_names[index])
# %%
# save model
# model.save("model")
