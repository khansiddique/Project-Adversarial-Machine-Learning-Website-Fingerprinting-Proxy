import numpy as np
#% matplotlibinline
import matplotlib.pyplot as plt


"""
When computing the z-score for each sample on the data set a threshold must be
specified. Some good ‘thumb-rule’ thresholds can be: 2.5, 3, 3.5 or
more standard deviations.
"""


def detect_outliers_with_z_score(data):

    """
    Using Z score
    Formula for Z score = (Observation — Mean)/Standard Deviation

    z = (X — μ) / σ
    """
    outliers = []
    threshold = 2.4  # Some good ‘thumb-rule’ thresholds can be: 2.5, 3, 3.5
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        z_score = (i - mean)/std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers


if __name__ == "__main__":

    #https: // github.com/krishnaik06/Finding-an-Outlier/blob/master/Finding % 20an % 20outlier % 20in % 20a % 20Dataset.ipynb
    dataset = [11, 10, 12, 14, 12, 15, 14, 13, 15, 102, 12, 14, 17, 19, 107, 10,
               13, 12, 14, 12, 108, 12, 11, 14, 13, 15, 10, 15, 12, 10, 14, 13, 15, 10]


    print(detect_outliers_with_z_score(dataset))
    pass

