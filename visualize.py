# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_curve,auc,confusion_matrix
import matplotlib.pyplot as plt


# %%
data=pd.read_csv('hepatitis.csv')
le=LabelEncoder()
data['class']=le.fit_transform(data['class'])
print(data['class'])


# %%
X=data.drop('class',axis=1)
y=data['class']
sc=StandardScaler()
X=sc.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# %%
clf_tree=DecisionTreeClassifier()
clf_mlp=MLPClassifier(max_iter=1000)
clf_knn=KNeighborsClassifier(n_neighbors=5)
ensemble=VotingClassifier(estimators=[('dt',clf_tree),('mlp',clf_mlp),('knn',clf_knn)],voting='hard')
clf_tree.fit(X_train,y_train)
clf_mlp.fit(X_train,y_train)
clf_knn.fit(X_train,y_train)
ensemble.fit(X_train,y_train)

# %%
for clf,name in [(clf_tree,'Decision Tree'),(clf_mlp,'MLP'),(clf_knn,'KNN'),(ensemble,'Ensemble')]:
    y_pred=clf.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"{name} Precision: {precision_score(y_test, y_pred)}")
    print(f"{name} Recall: {recall_score(y_test, y_pred)}")
    print(f"{name} F1-Score: {f1_score(y_test, y_pred)}")

    if name!='Ensemble':
      fpr,tpr,thresholds=roc_curve(y_test,clf.predict_proba(X_test)[:,-1])
      roc_auc=auc(fpr,tpr)
      plt.plot(fpr,tpr,label=f'{name} ROC CURVE ( AREA= {roc_auc:.2f})')

plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# %%
data

# %%
X1 = data[['bilirubin', 'albumin']]
sc=StandardScaler()
X1=sc.fit_transform(X1)
X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y,test_size=0.2,random_state=42)

# %%
clf_tree1=DecisionTreeClassifier()
clf_mlp1=MLPClassifier(max_iter=1000)
clf_knn1=KNeighborsClassifier(n_neighbors=5)
ensemble1=VotingClassifier(estimators=[('dt',clf_tree1),('mlp',clf_mlp1),('knn',clf_knn1)],voting='hard')
clf_tree1.fit(X_train1,y_train1)
clf_mlp1.fit(X_train1,y_train1)
clf_knn1.fit(X_train1,y_train1)
ensemble1.fit(X_train1,y_train1)

# %%
for clf,name in [(clf_tree1,'Decision Tree'),(clf_mlp1,'MLP'),(clf_knn1,'KNN'),(ensemble1,'Ensemble')]:
    h=0.02
    x_min,x_max=X1[:,0].min()-1,X1[:,0].max()+1
    y_min,y_max=X1[:,1].min()-1,X1[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.8)
    plt.scatter(X1[:,0],X1[:,1],c=y,edgecolors='g')
    plt.title(f'{name} Decision Boundary')
    plt.show()


# %%
from sklearn.svm import SVC
sv=SVC(kernel="linear",gamma=0.5,C=0.1)
sv.fit(X_train1,y_train1)

# %%
y_pred=sv.predict(X_test1)

# %%
print(accuracy_score(y_test1,y_pred))
print(precision_score(y_test1,y_pred))
print(recall_score(y_test1,y_pred))
print(f1_score(y_test1,y_pred))

# %%
DecisionBoundaryDisplay.from_estimator(
        sv,
        X1,
        response_method="predict",
        cmap=plt.cm.Spectral,
        alpha=0.8,

    )

# Scatter plot
plt.scatter(X1[:, 0], X1[:, 1],
            c=y,
            s=20, edgecolors="k")
plt.show()

# %%
from sklearn.inspection import DecisionBoundaryDisplay

disp=DecisionBoundaryDisplay.from_estimator(sv,X1,cmap=plt.cm.Spectral,response_method='predict',alpha=0.5)
disp.ax_.scatter(X1[:,0],X1[:,1],s=20,c=y,edgecolors='k')
plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred3 = rf_classifier.predict(X_test)


# %%
from sklearn.metrics import accuracy_score, classification_report
accuracy1 = accuracy_score(y_test, y_pred3)
classification_rep1 = classification_report(y_test, y_pred3)

# Print the results
print(f"Accuracy: {accuracy1:.2f}")
print("\nClassification Report:\n", classification_rep1)

# %%
from sklearn import tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(rf_classifier.estimators_[0],
                   filled=True,feature_names=['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',
       'anorexia', 'liver_big', 'liver_firm', 'spleen_palable', 'spiders',
       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',
       'protime', 'histology'],class_names=['0','1'])

# %%
# Load the important packages
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

# Load the datasets
cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

#Build the model
svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
# Trained the model
svm.fit(X, y)

# Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
		svm,
		X,
		response_method="predict",
		cmap=plt.cm.Spectral,
		alpha=0.8,
		xlabel=cancer.feature_names[0],
		ylabel=cancer.feature_names[1],
	)

# Scatter plot
plt.scatter(X[:, 0], X[:, 1],
			c=y,
			s=20, edgecolors="k")
plt.show()


# %%

d1=pd.read_csv('Wine.csv')
X2 = d1.iloc[:, 0:13].values
y2 = d1.iloc[:, 13].values

# %%
from sklearn.model_selection import train_test_split

X_train2, X_test2, y_train2, y_tes2t = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train2= sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)

# %%
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train2=pca.fit_transform(X_train2)
X_test2=pca.transform(X_test2)
explained_varience=pca.explained_variance_ratio_


# %%
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train2, y_train2)

# %%
y_pred2 = classifier.predict(X_test2)

# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_tes2t, y_pred2)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

disp=DecisionBoundaryDisplay.from_estimator(classifier,X_train2,response_method='predict',alpha=0.5)
disp.ax_.scatter(X_train2[:,0],X_train2[:,1],s=20,c=y,edgecolors='k')
plt.show()

# %%
data.columns

# %%
from sklearn import tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_tree,
                   filled=True,feature_names=['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',
       'anorexia', 'liver_big', 'liver_firm', 'spleen_palable', 'spiders',
       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',
       'protime', 'histology'],class_names=['0','1'])


# %%
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.array([[0.15,0.25],[0.2,0.3]])
        self.weights_hidden_output = np.array([[0.4,0.45],[0.5,0.55]])


        # Initialize the biases
        self.bias_hidden = np.array([[0.35,0.35]])
        self.bias_output = np.array([[0.6,0.6]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        # Input to hidden
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)

        # Hidden to output
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Compute the output layer error
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        # Compute the hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            if epoch == 0 or epoch==99:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")
                print(f"Input_Hidden_Weights {self.weights_input_hidden}")
                print(f"Hidden_Output_Weights {self.weights_hidden_output}")




X = np.array([[0.05,0.1]])
y = np.array([[0.01,0.99]])

nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=2)
nn.train(X, y, epochs=1, learning_rate=0.5)

# Test the trained model
output = nn.feedforward(X)
print("Predictions after training:")
print(output)