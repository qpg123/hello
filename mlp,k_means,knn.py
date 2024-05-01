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


# %%
import numpy as np
class NeuralNetwork:
   def __init__(self,ip,hd,op):
    self.ip=ip
    self.hd=hd
    self.op=op


    self.weights_ip=np.array([[.15,.25],[.2,.3]])
    self.weights_op=np.array([[.4,.5],[.45,.55]])
    self.bias_ip=np.array([[.35,.35]])
    self.bias_op=np.array([[.6,.6]])

   def sig(self,x):
      return 1/(1+ np.exp(-x))

   def sig_der(self,x):
      return x*(1-x)

   def forward(self,X):
      self.hid_ac=np.dot(X,self.weights_ip)  +   self.bias_ip
      self.hid_op=self.sig(self.hid_ac)

      self.output_act = np.dot(self.hid_op, self.weights_op) + self.bias_op
      self.predicted_output = self.sig(self.output_act)

      return self.predicted_output

   def back(self,X,y,le):
    op_err=y-self.predicted_output
    op_del=op_err* self.sig_der(self.predicted_output)
    hid_err=np.dot(op_del,self.weights_op.T)
    hid_del=hid_err* self.sig_der(self.hid_op)

    self.weights_op+=np.dot(self.hid_op.T,op_del)*le
    self.weights_ip+=np.dot(X.T,hid_del)*le
    self.bias_op+=np.sum(op_del,axis=0,keepdims=True)*le
    self.bias_ip+=np.sum(hid_del,axis=0,keepdims=True)*le

   def train(self,X,y,epochs,le):
    for epoch in range(epochs):
      output=self.forward(X)
      self.back(X,y,le)
      if epoch == 0 or epoch==99:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")
                print(f"Input_Hidden_Weights {self.weights_ip}")
                print(f"Hidden_Output_Weights {self.weights_op}")

X = np.array([[0.05,0.1]])
y = np.array([[0.01,0.99]])
nn=NeuralNetwork(ip=2,hd=2,op=2)
nn.train(X,y,epochs=1,le=0.5)
output = nn.forward(X)
print("Predictions after training:")
print(output)


# %%
import pandas as pd

home_data = pd.read_csv('/content/sample_data/california_housing_train.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
home_data.head()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

# %%
from sklearn import preprocessing

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# %%
import sklearn

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

# %%
from sklearn.metrics import silhouette_score

silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

# %%
K = range(2, 8)
fits = []
score = []


for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)

    # append the model to fits
    fits.append(model)

    # Append the silhouette score to scores
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

# %%
import seaborn as sns
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[0].labels_)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # Focus on petal length and petal width
y = iris.target

# Scale features for better k-NN performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Create a mesh grid of values
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict classifications for each point in the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.brg)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.brg)
plt.title('k-NN Decision Boundaries with k=5')
plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
plt.colorbar(ticks=[0, 1, 2])
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # Focus on petal length and petal width
y = iris.target

# Scale features for better k-NN performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the new flower data point
new_flower = np.array([[1.4, 0.2]])  # New flower's petal length and width
new_flower_scaled = scaler.transform(new_flower)  # Scale the new flower data

# Create the k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Predict the class of the new flower
new_flower_class = knn.predict(new_flower_scaled)

# Create a mesh grid of values for the background
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict classifications for each point in the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.brg)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.brg)

# Highlight the new flower
plt.scatter(new_flower_scaled[:, 0], new_flower_scaled[:, 1], c='yellow', s=200, edgecolor='k', label='New Flower Prediction', marker='*')

plt.title('k-NN Decision Boundaries with k=5')
plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
plt.colorbar(ticks=[0, 1, 2])
plt.legend(loc="upper left")
plt.show()


# %%
#wcss
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i , init = "k-means++" , random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.ylabel("WCSS")
plt.xlabel("Number of Clusters")
plt.title("Elbow Curve Method")
plt.show()