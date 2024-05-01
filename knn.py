# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load the dataset
data = pd.read_csv("Data.csv")

# Preprocess the data
# Encode categorical variables
label_encoder = LabelEncoder()
data['Marital'] = label_encoder.fit_transform(data['Marital'])

# Remove '$' and ',' from 'Income' column and convert it to float
data['Income'] = data['Income'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Map target labels to integers
label_mapping = {'Bad loss': 0, 'Good risk': 1}
data['Risk'] = data['Risk'].map(label_mapping)

# Split the dataset into features and labels
X = data[['Age', 'Marital', 'Income']]
y = data['Risk']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check the shapes of X_train, X_test, y_train, and y_test
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Initialize the k-NN classifier with k=9
k = 7
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Replace NaN value in y_train with the mode
y_train.fillna(y_train.mode()[0], inplace=True)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# New record to classify (#10)
new_record = [[66, 'Married', 36120.34]]

# Convert the new record to a DataFrame
new_record_df = pd.DataFrame(new_record, columns=['Age', 'Marital', 'Income'])

# Print the new record DataFrame
print("\nNew record before preprocessing:")
print(new_record_df)

# Preprocess the new record
new_record_df['Marital'] = label_encoder.transform(new_record_df['Marital'])
new_record_df['Income'] = new_record_df['Income'].astype(float)
new_record_scaled = scaler.transform(new_record_df)

# Print the new record after preprocessing
print("\nNew record after preprocessing:")
print(new_record_df)

# Perform prediction on the new record
prediction = knn_classifier.predict(new_record_scaled)

# Map predicted label back to original label
predicted_label = 'Good risk' if prediction[0] == 1 else 'Bad loss'
print("\nPredicted risk for the new record (#10) using k =", k, ":", predicted_label)


# %%
# Load the dataset
data = pd.read_csv("Data.csv")

# Remove '$' and ',' from 'Income' column and convert it to float
data['Income'] = data['Income'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Calculate min-max standardized values for 'Age' and 'Income' attributes
min_age = data['Age'].min()
max_age = data['Age'].max()
min_income = data['Income'].min()
max_income = data['Income'].max()

# Min-max standardization formula
data['Age_MinMax'] = (data['Age'] - min_age) / (max_age - min_age)
data['Income_MinMax'] = (data['Income'] - min_income) / (max_income - min_income)

# Display the min-max standardized values
print("Min-max standardized values for 'Age':")
print(data['Age_MinMax'])
print("\nMin-max standardized values for 'Income':")
print(data['Income_MinMax'])


# %%
# New record (#10) standardized values
new_record_std = new_record_scaled[0]

# Calculate Euclidean distance for each record in X_train
distances = []
for record in X_train:
    distance = np.sqrt(np.sum((record - new_record_std) ** 2))
    distances.append(distance)

# Display distances
for i, distance in enumerate(distances, start=1):
    print(f"Distance from record #{i}: {distance}")

# %%
from collections import Counter

# Find indices of the k-nearest neighbors
k = 9
nearest_indices = np.argsort(distances)[:k]

# Get labels of the k-nearest neighbors
nearest_labels = y_train.iloc[nearest_indices]

# Perform unweighted voting to classify the risk factor
predicted_label = Counter(nearest_labels).most_common(1)[0][0]

# Map predicted label back to original label
predicted_risk = 'Good risk' if predicted_label == 1 else 'Bad loss'

print("Predicted risk for the new record (#10) using unweighted voting:", predicted_risk)


# %%
