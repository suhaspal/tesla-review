import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset from pickle file
with open('./data.pickle', 'rb') as file:
    dataset = pickle.load(file)

features = np.array(dataset['data'])
targets = np.array(dataset['labels'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, stratify=targets, shuffle=True
)

# Initialize and train the Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Predict on the test set
predictions = rf_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"{accuracy * 100:.2f}% of samples were classified correctly!")

# Save the trained model to a file
with open('model.p', 'wb') as model_file:
    pickle.dump({'model': rf_clf}, model_file)
