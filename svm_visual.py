import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

df = pd.read_csv('data/training.csv')
data_train = df[['jockey_ave_rank', 'recent_ave_rank', 'horse_rank_top_50_percent']].values
Xtrain, Ytrain_HorseRankTop50Percent = data_train[:,0:2], data_train[:,2:3].ravel()
print ("Start training")

svm_model = svm.SVC(class_weight='balanced', kernel='linear', C = 15000)
svm_model.fit(Xtrain, Ytrain_HorseRankTop50Percent)
print("Plotting")
y = svm_model.predict(Xtrain)

fig, ax = plt.subplots()
X0, X1 = Xtrain[:, 0], Xtrain[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('jockey_ave_rank')
ax.set_ylabel('recent_ave_rank')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('SVC with linear kernel')

plt.show()


fig, ax = plt.subplots()
X0, X1 = Xtrain[:, 0], Xtrain[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=Ytrain_HorseRankTop50Percent, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('jockey_ave_rank')
ax.set_ylabel('recent_ave_rank')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('SVC with linear kernel')

plt.show()