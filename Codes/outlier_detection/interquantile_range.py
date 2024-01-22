
import numpy as np
#% matplotlibinline
import matplotlib.pyplot as plt

"""
InterQuantile Range
75%- 25% values in a dataset

Steps
1. Arrange the data in increasing order
2. Calculate first(q1) and third quartile(q3)
3. Find interquartile range (q3-q1)
4.Find lower bound q1-iqr*1.5
5.Find upper bound q3-iqr*1.5
Anything that lies outside of lower and upper bound is an outlier
"""


def detect_outliers_with_IQR(data):
  outliers = []

  ## Perform all the steps of IQR
  dataset = sorted(data)
  #Calculate first(q1) and third quartile(q3)
  quantile1, quantile3 = np.percentile(dataset, [25, 75])
  #print(quantile1, quantile3)
  ## Find the IQR
  iqr_value = quantile3-quantile1
  #print(iqr_value)

  ## Find the lower bound value and the higher bound value

  lower_bound_val = quantile1 - (1.5 * iqr_value)
  upper_bound_val = quantile3 + (1.5 * iqr_value)

  #print(lower_bound_val, upper_bound_val)
  for i in data:
      """
      For correct Data:
      Case 1-
      is 45.50 < 5 and is 5 < 71.45 ?? - False
      Case 2-
      is 45.50 < 50 and is 50 < 71.45 ?? - False
      Beyond this limit the Data falls into the Outliers
      """
      if  i < lower_bound_val or upper_bound_val < i:
        outliers.append(i)



  return outliers


if __name__ == "__main__":

    #https: // github.com/krishnaik06/Finding-an-Outlier/blob/master/Finding % 20an % 20outlier % 20in % 20a % 20Dataset.ipynb
    dataset = [11, 10, 12, 14, 12, 15, 14, 13, 15, 102, 12, 14, 17, 19, 107, 10,
               13, 12, 14, 12, 108, 12, 11, 14, 13, 15, 10, 15, 12, 10, 14, 13, 15, 10]
    print(detect_outliers_with_IQR(dataset))

    """


    """

    #pass
