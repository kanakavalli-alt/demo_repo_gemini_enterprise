import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from PIL import Image

def load_and_preprocess_data(data_dir, img_size=(64, 64)):
    X = []
    y = []
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found. Generating synthetic dummy data for demonstration...")
        X = np.random.rand(1000, img_size[0] * img_size[1] * 3)
        y = np.random.randint(0, 2, 1000)
        return X, y

    classes = {'cats': 0, 'dogs': 1}
    for class_name, label in classes.items():
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                try:
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path).resize(img_size)
                    img_array = np.array(img).flatten()
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    pass
    return np.array(X), np.array(y)

def train_model():
    X, y = load_and_preprocess_data("dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    clf = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    print("Training Decision Tree...")
    clf.fit(X_train, y_train)
    print(f"Best parameters selected: {clf.best_params_}")
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    return clf.best_estimator_

def predict_image(model, img_path, img_size=(64, 64)):
    try:
        img = Image.open(img_path).resize(img_size)
        img_array = np.array(img).flatten().reshape(1, -1)
        prediction = model.predict(img_array)
        result = "Dog" if prediction[0] == 1 else "Cat"
        print(f"The predicted class for {img_path} is: {result}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print(\"Starting Cat vs Dog Classifier Training...\")
    best_model = train_model()