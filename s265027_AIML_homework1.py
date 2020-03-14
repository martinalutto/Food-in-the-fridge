# In[1]:


# Machine Learning and Artificial Intelligence 
# Homework 1 - Martina Alutto s265027

import sklearn.datasets 
from sklearn.preprocessing import StandardScaler

dataset = sklearn.datasets.load_wine()
wine = dataset.data[:,[0, 1]]  # Select the first two attributes
print("Names of labels are: ")
print(dataset.target_names) 

# Standardize dataset for easier parameter selection
wine = StandardScaler().fit_transform(wine)
print("\nThe standardized selected attributes are: ")
print(wine)


# In[2]:


import matplotlib.pyplot as plt
# Select the first and second attribute to plot them in a 2D graph
x1 = wine[:,0] 
x2 = wine[:,1]
label = dataset.target

print("\n2D representation of our data.")
plt.scatter(x1, x2, c = label)
plt.show()


# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split

# Apply the function train_test_split
# With the first run we get the train set, 
# the reamining set is passed to the function again
# for getting validation and test set.
wine_train, wine_s, target_train, target_s = train_test_split(wine, label, test_size=0.5, train_size=0.5, random_state = 0)
wine_cv, wine_test, target_cv, target_test = train_test_split(wine_s, target_s, test_size = 0.6, train_size =0.4, random_state = 0)

# Merge the training and validation split.
wine_train2 = np.concatenate((wine_train, wine_cv))
target_train2 = np.concatenate((target_train, target_cv))

print("\nThe sizes of train, validation and test set are:")
print(wine_train.size, wine_cv.size, wine_test.size)


# In[4]:


from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

X = wine_train
y = target_train
h = .02  # h is the step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# K-NN 
print("\n ---- K-Nearest Neighbors ----")
for n_neighbors in [1,3,5,7]:
    # Create an instance of k Neighbors Classifier and fit the train data.
    clf_knn = KNeighborsClassifier(n_neighbors= n_neighbors)
    clf_knn.fit(X, y)

    # Plot the decision boundary. 
    # For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf_knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot.
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points.
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("K-NN classification (k = %i)" % (n_neighbors))

print("\nPlot of the decision boundaries.")
plt.show()


# In[5]:


# Compute the accuracy of each k-nn classifier and evaluate 
# it on the validation set
score = np.zeros(20)

for k in range(1,21,1):
    clf_knn = KNeighborsClassifier(n_neighbors = k)
    clf_knn.fit(X, y)
    score[k-1] = clf_knn.score(wine_cv, target_cv)    

print("\nAccuracy of the model on the validation set.")
print(score)


# In[6]:


# Plot a graph of the accuracy 
print("\nPlot of the accuracy of the model on the validation set.")
k = range(1,21,1)
plt.plot(k,score)
plt.show()


# In[7]:


# We choose k = 6 because is the best value 
# Evaluate the model on the test set

knn_best = KNeighborsClassifier(n_neighbors = 6)
knn_best.fit(wine_train2, target_train2)
knn_best.predict(wine_test) 

print("\nEvaluation on the test set with the best value of k.")
print(knn_best.score(wine_test, target_test))


# In[8]:


# Linear SVM
print("\n ---- Linear SVM ----")
from sklearn import svm

for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # Create an instance of SVC Classifier with linear kernel and fit the data.
    clf_svm = svm.SVC(C = c, kernel = 'linear', gamma='auto')
    clf_svm.fit(X, y)

    Z = clf_svm.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("SVM with linear kernel classification (c = %.3f)" % (c))

print("Plot of the decision boundaries.")
plt.show()


# In[9]:


score = np.zeros(7)
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for i in range(1,8,1) :
    clf_svm = svm.SVC(C=c[i-1], kernel = 'linear', gamma='auto')
    clf_svm.fit(X, y)
    score[i-1] = clf_svm.score(wine_cv, target_cv)  

print("\nAccuracy of the model on the validation set.")
print(score)


# In[10]:


# Plot a graph of the accuracy 
print("\nPlot of the accuracy of the model on the validation set.")
i = range(1,8,1)
plt.plot(i,score)
plt.show()


# In[11]:


# Best value is c = 10
# Evaluate the model on the test set

best_svm = svm.SVC(C = 10, kernel = 'linear', gamma='auto')
best_svm.fit(wine_train2, target_train2)
best_svm.predict(wine_test) 

print("\nEvaluation on the test set with the best value of c.")
print(best_svm.score(wine_test, target_test))


# In[12]:


# SVM with RBF kernel
print("\n ---- SVM with RBF kernel ----")
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # we create an instance of SVC Classifier with RBF kernel and fit the data.
    clf_rbf = svm.SVC(C = c, kernel = 'rbf', gamma='auto')
    clf_rbf.fit(X, y)

    Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("RBF kernel classification (c = %.3f)" % (c))

print("\nPlot of the decision boundaries.")
plt.show()


# In[13]:


score = np.zeros(7)
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for i in range(1,8,1) :
    clf_rbf = svm.SVC(C=c[i-1], kernel='rbf', gamma='auto')
    clf_rbf.fit(X, y)
    score[i-1] = clf_rbf.score(wine_cv, target_cv)    

print("\nAccuracy of the model on the validation set.")
print(score)


# In[14]:


# Plot a graph of the accuracy 
print("\nPlot of the accuracy of the model on the validation set.")
i = range(1,8,1)
plt.plot(i,score)
plt.show()


# In[15]:


# Best value of c is 100
# evaluate the method with the best value of c on the test set

best_rbf = svm.SVC(C = 100, kernel='rbf', gamma='auto')
best_rbf.fit(wine_train2, target_train2)
best_rbf.predict(wine_test) 

print("\nEvaluation on the test set with the best value of c.")
print(best_rbf.score(wine_test, target_test))


# In[16]:


# Grid search of the best parameters for an RBF kernel.
score = np.zeros([7,7])
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = [0.0001, 0.001, 0.005, 0.01, 0.1, 1.0, 10]

for i in range(1,8,1):
    for j in range(1,8,1):
        clf_rbf = svm.SVC(C=c[i-1], kernel='rbf', gamma=gamma[j-1])
        clf_rbf.fit(X, y)
        score[i-1,j-1] = clf_rbf.score(wine_cv, target_cv) 

print("\nGrid search of the best parameters for an RBF kernel.")
print(score)


# In[17]:


# c = 10 and gamma = 1.0

best_rbf = np.max(score)
print("\nThe best accuracy is: ")
print(best_rbf)
# print(np.argmax(score))

clrbf = svm.SVC(C = 10, kernel='rbf', gamma=1.0)
clrbf.fit(wine_train2, target_train2)
clrbf.predict(wine_test) 

print("\nEvaluation on the test set with the best values of c and gamma.")
print(clrbf.score(wine_test, target_test))


# In[18]:


c = 10
gamma = 1.0

clf_rbf = svm.SVC(C = c, kernel = 'rbf', gamma=gamma)
clf_rbf.fit(X, y)

Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("RBF kernel classification (c = %.3f, gamma= %.3f)" % (c, gamma))
 
print("\nPlot of the decision boundaries.")
plt.show()


# In[19]:


# K-Fold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(5) # Cross validation generator for model selection
c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma = [0.0001, 0.001, 0.005, 0.01, 0.1, 1, 10]

for i in range(1,8,1):
    for j in range(1,8,1):
        clf_rbf = svm.SVC(C=c[i-1], kernel='rbf', gamma=gamma[j-1])
        score[i-1,j-1] = np.mean(cross_val_score(clf_rbf, wine_train2, target_train2, cv=cv))

print("\nGrid Search of the best parameters for an RBF kernel with 5-fold validation.")
print(score)

best_rbf = np.max(score)
print("\nThe best accuracy is: ")
print(best_rbf)
# print(np.argmax(score))


# In[20]:


c = 1
gamma = 1
clrbf = svm.SVC(C =c, kernel='rbf', gamma=gamma)
clrbf.fit(wine_train2, target_train2)
clrbf.predict(wine_test) 

print("\nEvaluation on the test set with the best values of c and gamma.")
print(clrbf.score(wine_test, target_test))