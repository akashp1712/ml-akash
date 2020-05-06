# This file demonstrates the Data Augmentation on MNIST datasets

from sklearn.datasets import fetch_openml
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import numpy as np


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

print("Fetching Dataset...")
mnist = fetch_openml('mnist_784', version=1)
print("Fetching Dataset completed.")

# Get the data and target
X, y = mnist["data"], mnist["target"]

# Split the train and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

image = X_train[1000]
shifted_image_down = shift_image(image, 0, 1)
shifted_image_left = shift_image(image, -1, 0)

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(132)
plt.title("shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(133)
plt.title("shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")

# Uncomment the follwoing to see the example of shift
#plt.show()

print("Creating Augmented Dataset...")
X_train_augmented = [image for image in X_train]
y_train_augmented = [image for image in y_train]

for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
     for image, label in zip(X_train, y_train):
             X_train_augmented.append(shift_image(image, dx, dy))
             y_train_augmented.append(label)


shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
y_train_augmented = np.array(y_train_augmented)[shuffle_idx]

print("Creating Augmented Dataset completed")

# from sklearn.neighbors import KNeighborsClassifier
# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train_augmented, y_train_augmented)
# y_pred = knn_clf.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

print("Training on the existing dataset")
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)


print("Training on augmented dataset")
rf_clf_for_augmented = RandomForestClassifier(random_state=42)
rf_clf_for_augmented.fit(X_train_augmented, y_train_augmented)
y_pred_after_augmented = rf_clf_for_augmented.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy score after training on existing dataset", accuracy_score(y_test, y_pred))
print("Accuracy score after training on augmented dataset", accuracy_score(y_test, y_pred_after_augmented ))

