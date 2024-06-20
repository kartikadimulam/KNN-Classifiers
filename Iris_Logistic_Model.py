import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

from sklearn import linear_model, datasets
import pandas as pd

iris = datasets.load_iris()

iris_dataset = pd.DataFrame(iris.data, columns = iris.feature_names)

print(iris_dataset)
iris_dataset['target'] = pd.Series(iris.target)

df0 = iris_dataset[iris_dataset.target==0]
df1 = iris_dataset[iris_dataset.target==1]
df2 = iris_dataset[iris_dataset.target==2]

print('Number of rows corresponding to 0:', len(df0))
print('Number of rows corresponding to 1:', len(df1))
print('Number of rows corresponding to 2:', len(df2))

X = iris.data[:, :2]
Y = iris.target


logistic_regression = linear_model.LogisticRegression(C=1e5)
logistic_regression.fit(X,Y)

h=0.01

x_min, x_max = X[:,0].min()-3.5, X[:,0].max()+3.5
y_min, y_max = X[:,1].min()-3.5, X[:,1].max()+3.5

xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = logistic_regression.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(10,7))
plt.pcolormesh(xx,yy,Z, cmap=plt.cm.Paired)

plt.scatter(X[:,0], X[:,1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.title('Iris Dataset Logistic Regression Classifier Prediction Model', fontsize=14)
plt.show()

