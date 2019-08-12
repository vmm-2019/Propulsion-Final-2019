"""
plot_handler.py: Utility script for simple plots

__author__ = "Victor Marco Milli"
__version__ = "0.9.1"
__maintainer__ = "Victor Marco Milli"
__status__ = "Project/study script for project SWISS / Bise"

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd


def plot_confusion_matrix(cm, y_true):
    """ Deprecated, as it only works with binary classifications, use 'print_confusion_matrix' instead"""

    print(cm)
    df_cm = pd.DataFrame(cm, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(5, 4))
    sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font size


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # return fig


def plot_binary_distribution(binary_col_unsampled, binary_col_sampled):

    labels = ['Unsampled', 'Sampled']
    negatives = [binary_col_unsampled.value_counts()[0]] + [binary_col_sampled.value_counts()[0]]
    positives = [binary_col_unsampled.value_counts()[1]] + [binary_col_sampled.value_counts()[1]]

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, negatives, width, label='0')
    rects2 = ax.bar(x + width / 2, positives, width, label='1')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Negative positive ratios in datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()

    plt.show()

def autolabel(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


def plot_correlation_map(df):

    sn.set(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(25, 20))

    # Generate a custom diverging colormap
    cmap = sn.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


def get_correlated_features(df, threshold=0.8):

    corr = df.corr()
    correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
              print(corr.columns[i], corr.columns[j])
              colname = corr.columns[i]
              correlated_features.add(colname)

    return correlated_features