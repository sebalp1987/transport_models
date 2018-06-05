import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import itertools

def hist_utils(file_col_num:pd.DataFrame, name: str):
    mu = file_col_num[name].describe().loc[['mean']]
    sigma = file_col_num[name].describe().loc[['std']]
    max_len = file_col_num[name].describe().loc[['count']]
    mu = float(mu)
    sigma = float(sigma)
    max_len = int(max_len)
    s = np.random.normal(mu, sigma, max_len)
    sns.distplot(s, hist=False, label='Normal Distribution')
    sns.distplot(file_col_num[name], hist=False)
    plot.show()


def plot_confusion_matrix(cm, classes,  name, title='Confusion matrix', cmap=plot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Copyed from a kernel by joparga3 https://www.kaggle.com/joparga3/kernels
    """
    plot.figure()
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=0)
    plot.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    plot.savefig(name)
    plot.close()