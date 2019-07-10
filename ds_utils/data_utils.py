import numpy as np
import pandas as pd

from sklearn import preprocessing

def label_encoder(values, encode=False):
    le = preprocessing.LabelEncoder().fit(values)
    if encode:
        return le.transform(values)
    else:
        classes = list(le.classes_)
        return classes, le

#le.classes_

def one_hot(x):
    return np.array(preprocessing.OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

def discard_outliers(data, x_label, percentile=95):
    ulimit = np.percentile(data[x_label], percentile)
    llimit = np.percentile(data[x_label], 100-percentile)
    print("{} upper limit: {}".format(x_label, ulimit))
    print("{} lower limit: {}".format(x_label, llimit))
    data = data[(data[x_label] < ulimit) & (data[x_label] > llimit)]
    return data

def bin_dataframe(df,label_tobin, n_bins, y_label=None):
    x_bins = np.arange(n_bins)
    _, bins = pd.cut(df[label_tobin], bins=n_bins, retbins=True, right=False)
    bins = df.groupby(np.digitize(df[label_tobin], bins)).mean()
    if y_label:
        bins = bins[y_label]
    print('Bin size= {:.3f} minutes'.format((max(df[df[label_tobin]])/n_bins)))
    return bins

# scale vector values in the fixed range [0, 1]
def vector_scaling(a):
    a = (a - np.min(a))/(np.max(a) - np.min(a))
    return a


# Grid-Search creation of named parameters
from itertools import product, starmap
from collections import namedtuple


def named_configs(items):
    Config = namedtuple('Config', items.keys())
    return starmap(Config, product(*items.values()))
