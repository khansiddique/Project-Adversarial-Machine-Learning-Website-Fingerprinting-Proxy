# Simple Linear Regression

# Importing the libraries
#from sklearn.linear_model import LinearRegression
#from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Calculating the Sum of square due to error
# Influential points always reduce R_square
#Research Ref:
#https: // stats.stackexchange.com/questions/12900/when-is-r-squared-negative

"""
Research Ref:
https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python

If the RMSE value goes down over time we are happy because variance is decreasing.
RMSE has zero tolerance for outlier data points which don't belong


"""

"""
Outlier removal algorithm based on RMS distance
This document describes a very simple iterative algorithm to remove data outliers.
No claim about algorithm optimality is made. Anybody shall be free to use this
algorithm for any purpose and without any restriction. Let D be a set of points
in Rn, which may contain outliers.
(1) Calculate the arithmetic mean of all points in D (mean).
(2) Calculate the root-mean-square (RMS) distance of all points in D from the
mean (RMS distance).
(3) Remove from D all points whose distance from the mean is greater than K * RMS
distance (typically, K = 2 but in our case it is 1).
(4) Repeat (1), (2) and (3) until no points are removed from D.
(5) Return D.
Variations of this algorithm include using the median instead of the mean, and
using smaller values of K (normally not less than 1.5) for more aggressive outlier removal.

"""

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

"""
(2) Calculate the root-mean-square(RMS) distance of all points in D from the
mean(RMS distance).
"""
def cal_distance(X=0, Y=0, mean_v=0):

  # d = []  # ideal target distances, these can be all zeros.
  #call rmse(p, Y) ; p stands for prediction and Y is for our original data
  p = []  # your performance goes here

  # ss_t = 0  # Total Sum of square
  # ss_r = 0  # Total Sum of square of residuals

  #mean_x = np.mean(X)
  # Need to Change here, may be pass our original mean value
  #here, for error calculation
  # mean_y = np.mean(Y)

  # Total number of values
  # l = len(X)
  # y_pred = 0
  # for i in range(l):
  #   y_pred = intercept + slope * X[i]
  #   #print(float(y_pred))
  #   # ss_t += (Y[i] - mean_y) ** 2
  #   # ss_r += (Y[i] - y_pred) ** 2
  #   p.append(float(y_pred))

  #print(f"For prediction: {np.array(p)}")
  # print(type(p))
  # print(p)
  # print(type(np.array(p)))
  # print(type(Y))
  # print(type(np.array(Y)))
  #print(f"For Original Data: {np.array(Y)}")
  rmse_val = rmse(np.array(mean_v), Y)
  #R2 = 1 - (ss_r / ss_t)
  #print(type(int(r2)))
  #print(f'Error: {r2}')
  return rmse_val  # , R**2
  #pass


"""
def cal_error(X=0, Y=0, slope=0, intercept=0):

  # d = []  # ideal target distances, these can be all zeros.
  #call rmse(p, Y) ; p stands for prediction and Y is for our original data
  p = []  # your performance goes here

  # ss_t = 0  # Total Sum of square
  # ss_r = 0  # Total Sum of square of residuals

  #mean_x = np.mean(X)
  # Need to Change here, may be pass our original mean value
  #here, for error calculation
  # mean_y = np.mean(Y)

  # Total number of values
  l = len(X)
  y_pred =0
  for i in range(l):
    y_pred = intercept + slope * X[i]
    #print(float(y_pred))
    # ss_t += (Y[i] - mean_y) ** 2
    # ss_r += (Y[i] - y_pred) ** 2
    p.append(float(y_pred))

  #print(f"For prediction: {np.array(p)}")
  # print(type(p))
  # print(p)
  # print(type(np.array(p)))
  # print(type(Y))
  # print(type(np.array(Y)))
  #print(f"For Original Data: {np.array(Y)}")
  rmse_val = rmse(np.array(p), Y)
  #R2 = 1 - (ss_r / ss_t)
  #print(type(int(r2)))
  #print(f'Error: {r2}')
  return rmse_val  # , R**2
  #pass


def cal_r_square_method(X=0, Y=0, slope=0, intercept=0):

  ss_t = 0  #Total Sum of square
  ss_r = 0  #Total Sum of square of residuals

  #mean_x = np.mean(X)
  # Need to Change here, may be pass our original mean value
  #here, for error calculation
  mean_y = np.mean(Y)

  # Total number of values
  l = len(X)

  for i in range(l):
    y_pred = intercept + slope * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2

  R2 = 1 - (ss_r / ss_t)
  #print(type(int(r2)))
  #print(f'Error: {r2}')
  return R2 #, R**2
  #pass
"""
def plot_graphically_LR(X=0, Y=0, slope=0, intercept=0):
  #plt.rcParams['figure.figsize'] = (20.0, 10.0)
  max_x = np.max(X) #+ 100
  min_x = np.min(X) #- 100

  #Calculating the Lnie Values X and Y
  x = np.linspace(min_x, max_x, 50)
  y = intercept + slope * x

  # Visualising the Training set results
  # Ploting Line
  plt.plot(x, y, color='blue', label = 'Regression Line')

  # Ploting Scatter Points
  plt.scatter(X, Y, color='red', label='Scatter Plot')

  plt.title('Outliers Detection Plot')
  plt.xlabel('Index')
  plt.ylabel('Packet Size')
  plt.legend()
  plt.show()



'''

def cal_slope_intercept(X=0, Y=0):
    """
    A linear regression line has an equation of the form Y = c + mX, where X is the
    explanatory variable and Y is the dependent variable. The slope of the line is m,
    and c is the intercept (the value of y when x = 0).
    """
    # Mean X and Y
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    # Total number of values
    l = len(X)

    # Using the formula to calculate m and c
    numer = 0
    denom = 0

    for i in range(l):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2

    # Calculate the slope and intercept
    slope = numer / denom
    intercept = mean_y - (slope * mean_x)

    #print(slope, intercept)

    return slope, intercept


def two_point_for_cal_slope_intercept(X, Y, max_X, max_Y):
    """
    A linear regression line has an equation of the form Y = c + mX, where X is the
    explanatory variable and Y is the dependent variable. The slope of the line is m,
    and c is the intercept (the value of y when x = 0).
    """
    # Mean X and Y
    # mean_x = np.mean(X)
    # mean_y = np.mean(Y)

    # Total number of values
    # l = len(X)

    # Using the formula to calculate m and c
    numer = 0
    denom = 0

    #for i in range(l):
    numer = (max_X-X) * (max_Y-Y)
    denom = (max_X-X) ** 2

    # Calculate the slope and intercept
    slope = numer / denom
    intercept = max_Y - (slope * max_X)

    #print(slope, intercept)

    return slope, intercept
'''

"""
def detect_outliers_with_LR(data,df2):


    # reading Data
    #data = pd.read_csv('Salary_Data.csv')
    #print(data.shape)

    #print(data.head())

    # Importing the dataset
    #dataset = pd.read_csv('Salary_Data.csv')
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, 1].values

    #print(X)
    #print(Y)
    # Feature Scaling

    mean_75 = df2['75%'].mean()
    mean_25 = df2['25%'].mean()
    max_y = (mean_75 - mean_25) * 1.5 + mean_75
    max_x = df2['count'].max()
    # print(type(df2))
    # print(df2.head())
    # Fitting Simple Linear Regression to the Training set
    #slope, intercept = cal_slope_intercept(X, Y)
    slope, intercept = two_point_for_cal_slope_intercept(0, 0, max_x, max_y)

    # Predicting the Test set results
    R_square = cal_r_square_method(X, Y, slope, intercept)

    # Detecting Error with using the RMSE method
    RMSE_error = cal_error(X, Y, slope, intercept)
    print(f'RMSE: {RMSE_error} and Max Data Point: {max_y}')
    # Visualising the Test set results
    #plot_graphically_LR(X, Y, slope, intercept)

    return RMSE_error, max_y
"""

def detect_outliers_with_LR(data, df2):

    # reading Data

    #Outlier removal algorithm based on RMS distance
    #(1) Calculate the arithmetic mean of all points in D(mean).
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, 1].values

    #print(X)
    #print(Y)
    # Feature Scaling
    mean_75 = df2['75%'].mean()
    max_x = df2['count'].max()
    #Here to make a list of 50 points for a mean value.
    m = []
    m.append(mean_75)
    mean_fea_vec = m * int(max_x)
    # mean_75 = df2['75%'].mean()
    # mean_25 = df2['25%'].mean()
    # max_y = (mean_75 - mean_25) * 1.5 + mean_75
    # max_x = df2['count'].max()
    # print(type(df2))
    # print(df2.head())
    # Fitting Simple Linear Regression to the Training set
    #slope, intercept = cal_slope_intercept(X, Y)
    #slope, intercept = two_point_for_cal_slope_intercept(0, 0, max_x, max_y)

    # Detecting Error with using the RMSE method
    # Predicting the Test set results
    RMSE_error = cal_distance(X, Y, mean_fea_vec)


    # RMSE_error = cal_error(X, Y, slope, intercept)
    # print(
    #     f'RMSE: {RMSE_error} -> Doulbe the Error: {RMSE_error*2} and Max Data Point: {mean_75}')
    # # Visualising the Test set results
    #plot_graphically_LR(X, Y, slope, intercept)

    return RMSE_error, mean_75


def get_summary_fv_ds():
    curr_dir = os.getcwd()
    #print(curr_dir)
    get_file_path = os.path.join(curr_dir, 'output')
    #print(get_file_path)
    abs_path = os.path.join(get_file_path, 'feature_vectors.csv')
    #print(abs_path)

    #path = r'F:/git_clone/study_projects/website_fingerprinting_proxy/workspace/codes/output/feature_vectors.csv'

    df = pd.read_csv(abs_path)

    return df
