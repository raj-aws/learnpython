import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load images (make sure you have "cats" and "dogs" folders with images)
def load_images_from_folder(folder, label, size=(64,64)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, size)  # resize to fixed size
            data.append((img.flatten(), label))  # flatten image into 1D feature vector
    return data

cat_data = load_images_from_folder("cats", 0)  # 0 = cat
dog_data = load_images_from_folder("dogs", 1)  # 1 = dog

dataset = cat_data + dog_data
X = np.array([item[0] for item in dataset])  # features
y = np.array([item[1] for item in dataset])  # labels

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 5: Test on a new image
test_img = cv2.imread("test/cat.jpg")  # replace with your image
test_img = cv2.resize(test_img, (64,64)).flatten().reshape(1, -1)
prediction = model.predict(test_img)
print("Prediction:", "Dog 🐶" if prediction[0] == 1 else "Cat 🐱")
