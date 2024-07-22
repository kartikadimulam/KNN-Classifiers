import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

wine = datasets.load_wine()

X = wine.data[:, :2]
y = wine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

param_grid = {
    'n_neighbors': [3,5,10,15,20],
    'weights': ['uniform', 'distance'],
    'p': [1,2]
}

cv = StratifiedKFold(n_splits = 5)

clf = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, cv=cv)
clf.fit(X,y)
print('Ideal Parameters: ', clf.best_params_)

best_clf = clf.best_estimator_

scores = cross_val_score(best_clf, X, y, cv=5)
print('Cross Validation Scores: ', scores)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

n_neighbors = clf.best_params_['n_neighbors']
weights = clf.best_params_['weights']
h = 0.02

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

x_min, x_max = X_2D[:, 0].min()-1, X_2D[:, 0].max()+1
y_min, y_max = X_2D[:, 1].min()-1, X_2D[:, 1].max()+1

xx,yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))

Z = best_clf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))

Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,7))
plt.pcolormesh(xx,yy,Z, cmap = cmap_light)

plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("3-Class classification (k = %i, weights = '%s')"
         % (n_neighbors, weights))

plt.show()
    

