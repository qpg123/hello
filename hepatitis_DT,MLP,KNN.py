# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load and preprocess the dataset
def load_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data.replace("?", np.nan)

    # Impute missing values
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column] = data[column].astype(float)
            data[column].fillna(data[column].mean(), inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

    return data, label_encoders

# Visualize decision boundaries
def plot_decision_boundaries(X, y, model, model_title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='g')
    plt.title(model_title)
    plt.show()

# Main function to run the analysis
def main():
    data, encoders = load_preprocess_data('hepatitis.csv')
    X = data.drop(['class'], axis=1)
    y = data['class']
    y = y.map({1.0: 0, 2.0: 1})
    # Standardizing data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize classifiers
    clf_tree = DecisionTreeClassifier()
    clf_mlp = MLPClassifier(max_iter=1000)
    clf_knn = KNeighborsClassifier(n_neighbors=5)
    ensemble_clf = VotingClassifier(estimators=[('dt', clf_tree), ('mlp', clf_mlp), ('knn', clf_knn)], voting='hard')

    # Fit models
    clf_tree.fit(X_train, y_train)
    clf_mlp.fit(X_train, y_train)
    clf_knn.fit(X_train, y_train)
    ensemble_clf.fit(X_train, y_train)

    # Evaluate models
    for clf, name in [(clf_tree, "Decision Tree"), (clf_mlp, "MLP"), (clf_knn, "kNN"), (ensemble_clf, "Ensemble")]:
        y_pred = clf.predict(X_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"{name} Precision: {precision_score(y_test, y_pred)}")
        print(f"{name} Recall: {recall_score(y_test, y_pred)}")
        print(f"{name} F1-Score: {f1_score(y_test, y_pred)}")

        # ROC and AUC
        if name != "Ensemble":  # Skip ensemble for ROC as it does not support predict_proba by default
            fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} ROC curve (area = {roc_auc:.2f})')

    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
