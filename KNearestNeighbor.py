#Import scikit-learn dataset library
from sklearn import datasets
import numpy as np

#Load dataset
wine = datasets.load_wine()

# print the names of the features
print(wine.feature_names)

# print the label species(class_0, class_1, class_2)
print(wine.target_names)

print(wine.data[0:5])

print(wine.target)

from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test

#Import knearest neighbors Classifier model. K = 1 -> Overfitting
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("sklearn: n = 1 results")
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Let us now try K = 7 -> Good Fit

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("sklearn: n = 7 results")
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Let us now try K = 100 -> Underfitting

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=100)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("sklearn: n = 100 results")
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Let us normalize the features and continue with K = 7 -> Even Better Fit with Normalized Features

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
y_pred_train = knn.predict(X_train)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("sklearn: n = 7 and normalized results")
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def euclidean_distance(num1, num2):
    return np.sqrt(np.sum((num1 - num2) ** 2))

class MyKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test):
        predictions_arr = []
        for i in range(len(X_test)):
            #Creates a list for all the test points with the euclidean distance between
            #the test point and all data points in the training set

            distance_arr = [euclidean_distance(X_test[i], x_train) for x_train in self.X_train]

            #Creates a list with the k closest indices by sorting the distance list

            index_arr = np.argsort(distance_arr)[:self.k]

            #Crreates an array with the k closest labels

            label_arr =  [self.y_train[j] for j in index_arr]

            #the predicted label based on the label that appears the most frequent
            #in the label_arr
            prediction = max(set(label_arr), key = label_arr.count)

            #adds the prediction the list of predictions
            predictions_arr.append(prediction)

        return predictions_arr




def my_accuracy(predictions_arr, y_test):
    correct_pred = np.sum(predictions_arr == y_test)
    total_pred = len(y_test)
    return  correct_pred / total_pred


# Split dataset into training set and test set
X_train2, X_test2, y_train2, y_test2 = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test

my_knn = MyKNN(k = 1)
my_knn.fit(X_train2, y_train2)

my_y_pred = my_knn.predict(X_test2)
my_y_pred_train = my_knn.predict(X_train2)

print("\nMy K = 1 results")
print("Train Accuracy:",my_accuracy(y_train2, my_y_pred_train))
print("Accuracy:",my_accuracy(y_test2, my_y_pred))


my_knn = MyKNN(k = 7)
my_knn.fit(X_train2, y_train2)

my_y_pred = my_knn.predict(X_test2)
my_y_pred_train = my_knn.predict(X_train2)

print("My K = 7 results")
print("Train Accuracy:",my_accuracy(y_train2, my_y_pred_train))
print("Accuracy:",my_accuracy(y_test2, my_y_pred))


my_knn = MyKNN(k = 100)
my_knn.fit(X_train2, y_train2)

my_y_pred = my_knn.predict(X_test2)
my_y_pred_train = my_knn.predict(X_train2)

print("\nMy K = 100 results")
print("Train Accuracy:",my_accuracy(y_train2, my_y_pred_train))
print("Accuracy:",my_accuracy(y_test2, my_y_pred))


scaler2 = preprocessing.StandardScaler().fit(X_train2)

X_train2 = scaler2.transform(X_train2)
X_test2 = scaler2.transform(X_test2)

my_knn = MyKNN(k = 7)
my_knn.fit(X_train2, y_train2)

my_y_pred = my_knn.predict(X_test2)
my_y_pred_train = my_knn.predict(X_train2)

print("My K = 7 w/ normalization results")
print("Train Accuracy:",my_accuracy(y_train2, my_y_pred_train))
print("Accuracy:",my_accuracy(y_test2, my_y_pred))











