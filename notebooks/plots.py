import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

def decision_regions(x, y, classifier, test_idx=None, resolution=0.02, plot_support=False, plot_custom_support=False):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
                           )
    #z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    xy = np.array([xx1.ravel(), xx2.ravel()]).T
    z = classifier.decision_function(xy) #.reshape(x.shape)
    z = z.reshape(xx1.shape)

    plt.contour(xx1, xx2, z, alpha=0.3, cmap=cmap, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(
            x=x[y == c1, 0], y=x[y == c1, 1],
            alpha=0.8, c=colors[idx],
            marker=markers[idx], label=c1,
            edgecolors='none'
                    )
    if test_idx:
        x_test, y_test = x[test_idx,:], y[test_idx]
        plt.scatter(
            x_test[:,0], 
            x_test[:,1], 
            c='none', 
            edgecolors='green', 
            alpha=1.0, linewidth=1,
            marker='o', 
            s=250, 
            label='test set')

    if plot_support:
        plt.scatter(
            classifier.support_vectors_[:, 0],
            classifier.support_vectors_[:, 1],
            marker='o',
            s=100,
            c='none',
            alpha=1.0,
            linewidth=1,
            edgecolors='purple',
            #facecolors='none',
            label='support set'
            )
    if plot_custom_support:
        preds = classifier.decision_function(x)
        support_vectors = np.where(abs(preds) <= 1, 1, 0)
        #print(support_vectors)
        support_vector_idxs = np.where(support_vectors == 1)[0]
        #print(support_vector_idxs)

        x_support = x[support_vector_idxs, :]
        plt.scatter(
            x_support[:, 0],
            x_support[:, 1],
            marker='o',
            s=200,
            c='none',
            alpha=1.0,
            linewidth=1,
            edgecolors='orange',
            facecolors='none',
            label='custom support set'
            )