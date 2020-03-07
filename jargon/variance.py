'''
Utilities for dealing with feature variance. Allows to check feature
variance, remove features with variance below a certain threshold or
remove features with zero variance.

Functions
----------
print_feature_variance
plot_variance_histogram
remove_zero_variance_features
remove_features_below_variance_threshold

'''

import pandas as pd
pd.set_option("max_rows", 1000)
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

def print_feature_variance(data, ascending=True):
    '''
    Measure variance of features in a pandas DataFrame

    Parameters
    ----------
    data (pandas DataFrame): The DataFrame to assess
    ascending (bool, default True): If True, sorts the variances by ascending value

    Returns
    ----------
    variances (pandas Series): Sorted series of feature names and
    variances

    Raises
    ----------
    TypeError if data is not a pandas DataFrame
    '''

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    vt = VarianceThreshold()
    vt.fit(data)
    return pd.Series(dict(zip(data.columns, vt.variances_))).sort_values(ascending=ascending)

def plot_variance_histogram(data, **histargs):
    '''
    Plot a histogram of feature variance in a pandas DataFrame

    Parameters
    ----------
    data (pandas DataFrame): The DataFrame to assess
    **histargs are passed to seaborn.distplot

    Returns
    ----------
    Histogram of feature variances

    Raises
    ----------
    TypeError if data is not a pandas DataFrame
    '''

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    vt = VarianceThreshold()
    vt.fit(data)
    sns.distplot(vt.variances_, **histargs)


def remove_zero_variance_features(data):
    '''
    Removes features with zero variance.

    Parameters
    ----------
    data (pandas DataFrame): The DataFrame to apply the transformation to

    Returns
    ----------
    pandas DataFrame with removed columns

    Raises
    ----------
    TypeError if data is not a pandas DataFrame
    '''

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    selector = VarianceThreshold(threshold=1e-7)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

def remove_features_below_variance_threshold(data, threshold=1e-5):
    '''
    Removes features with variance below a certain threshold (<1e-5 by default).

    Parameters
    ----------
    data (pandas DataFrame): The DataFrame to apply the transformation to
    threshold (float): The variance threshold below which to discard a column

    Returns
    ----------
    pandas DataFrame with removed columns

    Raises
    ----------
    TypeError if data is not a pandas DataFrame
    '''

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]