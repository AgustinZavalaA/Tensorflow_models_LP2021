# %%
import numpy as np
import cv2

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow("model_april_20_frozen.pb")

# %%
# Input image
img = cv2.imread("dataset_completo/Classify/Meta/img_005.png")
rows, cols, channels = img.shape

# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(128, 96), swapRB=True, crop=False))

# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()

# %%
# Resultados
cv2.imshow("img test", img)
cv2.waitKey()
cv2.destroyAllWindows()

classes = ["banistas", "Negras", "Verdes", "Metas"]
print(classes[np.argmax(networkOutput)])
# %%
