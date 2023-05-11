# machinelearning

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['independent_variable1', 'independent_variable2']], 
                                                    df['dependent_variable'], test_size=0.2, random_state=42)

# Create an instance of the LinearRegression class
reg = LinearRegression()

# Fit the linear regression model to the training data
reg.fit(X_train, y_train)

# Generate predictions for the test data
y_pred = reg.predict(X_test)

# Evaluate the accuracy of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-Squared: {r2:.2f}')


---------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['independent_variable1', 'independent_variable2']], 
                                                    df['dependent_variable'], test_size=0.2, random_state=42)

# Create an instance of the LogisticRegression class
logreg = LogisticRegression()

# Fit the logistic regression model to the training data
logreg.fit(X_train, y_train)

# Generate predictions for the test data
y_pred = logreg.predict(X_test)

# Evaluate the accuracy of the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
print(f'Accuracy Score: {accuracy:.2f}')


--------------------------

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Determine the optimal number of clusters
# ... perform elbow method or silhouette analysis ...

# Create an instance of the KMeans class
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans model to the data
kmeans.fit(X)

# Generate cluster assignments for the data
labels = kmeans.predict(X)

# Visualize the clusters (for example, using a scatter plot)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.show()


--------------------------

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the KNeighborsClassifier class
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the KNeighborsClassifier model to the training data
knn.fit(X_train, y_train)

# Generate predictions for the testing data
y_pred = knn.predict(X_test)

# Evaluate the performance of the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the SVM model class
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

# Fit the SVM model to the training data
svm.fit(X_train, y_train)

# Generate predictions for the testing data
y_pred = svm.predict(X_test)

# Evaluate the performance of the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


-----------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the MLPClassifier class
mlp = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, max_iter=200)

# Fit the MLP model to the training data
mlp.fit(X_train, y_train)

# Generate predictions for the testing data
y_pred = mlp.predict(X_test)

# Evaluate the performance of the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

-----------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the Perceptron class
slp = Perceptron(alpha=0.0001, max_iter=1000, tol=1e-3)

# Fit the SLP model to the training data
slp.fit(X_train, y_train)

# Generate predictions for the testing data
y_pred = slp.predict(X_test)

# Evaluate the performance of the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
