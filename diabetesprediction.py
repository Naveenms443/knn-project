import numpy as np
import pandas as pd

# Function to calculate Euclidean distance
def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2) ** 2))

def knn_predict(train_data, train_labels, test_instance, k):
    distances = []
    for i, train_instance in enumerate(train_data):
        distance = euclidean_distance(test_instance, train_instance)
        distances.append((i, distance))

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])

    # Get the indices of the k-nearest neighbors
    neighbors_indices = [index for index, _ in distances[:k]]

    # Get the labels of the k-nearest neighbors
    neighbors_labels = [train_labels[index] for index in neighbors_indices]

    # Predict the class based on majority voting
    prediction = max(set(neighbors_labels), key=neighbors_labels.count)
    return prediction

# Load dataset from CSV file
data = pd.read_csv("Dataset.csv")

# Extract features and labels
train_data = data.iloc[:, :-1].values
train_labels = data.iloc[:, -1].values

# Test instance for prediction
test_instance = np.array([3, 100, 50, 10, 135, 15.05, 0.321, 20])#test data (for having case[[5, 166, 72, 19, 175, 25.8, 0.587, 51]])#(for not have case[[3, 100, 50, 10, 135, 15.05, 0.321, 20])

# Set the value of k for KNN
k_value = 5

# Predict the outcome for the test instance
prediction = knn_predict(train_data, train_labels, test_instance, k_value)

# Display the prediction
if prediction == 1:
    print("The person has diabetes.")
else:
    print("The person does not have diabetes.")

# Calculate Accuracy
test_predictions = [knn_predict(train_data, train_labels, instance, k_value) for instance in train_data]
accuracy = np.mean(test_predictions == train_labels) * 100
print("Accuracy:", accuracy)
