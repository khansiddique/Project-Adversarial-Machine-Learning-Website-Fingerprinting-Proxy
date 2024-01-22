import csv
import os
import re
import read_file as RF
import write_file as WF
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import seaborn as sns  # need to install in our environment for visualization of plot
import interquantile_range as IQR
import outliers_evaluation as EV
import outlier_detection as DET
import time
"""
Boxplot:
 These box plots produces graphical representation and allows the human auditor
 to usually pinpoint the outlying points, includes the usual inspection,
 they can handle real valued, ordinal and categorical attributes box plots a lot
 the lower extreme, lower quartile, median, upper quartile and upper extreme points.

 These are basically the feature selection techniques that are used essentially
 to remove the noise from the data distribution and focus on the main cluster of
 the normal data points while isolating the outliers [23]
"""

'''
### Defination of the "Simulation mode".
@@@The developed outlier remover should also offer an so called simulation mode.
In this mode your program only detects the outliers but either does not change
the input file nor write an output file.
'''


def visualization_box_plot(df_data):
  #Here we plot the multiple Boxplot at a time to visualize the Outliers
  sns.boxplot(x="variable", y="value", data=pd.melt(df_data))
  plt.title('Detecting Outliers')
  plt.show()

  return

def simulation_mode(URL):
  #call multi_files_names() function to get specific file name for Marged Data
  file_name = multi_files_names(URL)
  #call merge_file_path() function to enter into the directory of "file_name"
  #for find out the file.
  file_name = merge_file_path(file_name)
  #print(file_name)
  df = pd.read_csv(file_name)

  #print(df.head())
  header_list = list(df.columns)
  #print(header_list[0:-1])
  #print(len(header_list[1:41]))
  #https: // stackoverflow.com/questions/12920214/python-for-syntax-block-code-vs-single-line-generator-expressions
  #spliting the header
  new_header_list = []
  for s in header_list[1:41]:
    s = s.split("_")  # Here feature column Header is splited
    #print(s)
    # Here Header identifier and crawl number is concatenated
    new_header_list.append(s[0] + "." + s[1])
  #print(new_header_list)
  #Research Ref:
  # https: // www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
  df_data = pd.DataFrame(data=df)
  # Delete the "Area" column from the dataframe
  df_data = df_data.drop(columns="index")

  #research Ref:
  #https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
  # Rename multiple columns in one go with a larger dictionary
  #header_list_new = df_data.columns.str.replace('_', '.')

  df_data.columns = new_header_list

  return df_data, file_name

########################  Find Min, Max and Mean curve, i.e. feature vector, of
# ######################  the input data set as well as of the adjusted one
# ######################  as well as their respective distances

#Research Ref:
#https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/
#https://stackoverflow.com/questions/19482970/get-list-from-pandas-dataframe-column-headers
#https://stackoverflow.com/questions/49554139/boxplot-of-multiple-columns-of-a-pandas-dataframe-on-the-same-figure-seaborn
def find_min_max_mean(URL):
  #call multi_files_names() function to get specific file name for Marged Data
  file_name = multi_files_names(URL)
  #call merge_file_path() function to enter into the directory of "file_name"
  #for find out the file.
  file_name = merge_file_path(file_name)
  #print(file_name)
  df = pd.read_csv(file_name)
  """
  #print(df.head())
  header_list = list(df.columns)
  #print(header_list[0:-1])
  #print(len(header_list[1:41]))
  #https: // stackoverflow.com/questions/12920214/python-for-syntax-block-code-vs-single-line-generator-expressions
  #spliting the header
  new_header_list =[]
  for s in header_list[1:41]:
    s = s.split("_") # Here feature column Header is splited
    #print(s)
    # Here Header identifier and crawl number is concatenated
    new_header_list.append(s[0] + "." + s[1])
  #print(new_header_list)
  #Research Ref:
  # https: // www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
  df_data = pd.DataFrame(data=df)
  # Delete the "Area" column from the dataframe
  df_data = df_data.drop(columns="index")

  #research Ref:
  #https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
  # Rename multiple columns in one go with a larger dictionary
  #header_list_new = df_data.columns.str.replace('_', '.')

  df_data.columns = new_header_list
  #Here we plot the multiple Boxplot at a time to visualize the Outliers
  sns.boxplot(x="variable", y="value", data=pd.melt(df_data))
  plt.title('Detecting Outliers')
  plt.show()

  """
  #research Ref:
  #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
  #https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
  #https://stackoverflow.com/questions/20461165/how-to-convert-index-of-a-pandas-dataframe-into-a-column
  #Create new dataset to describe the summary of the Dataset.
  s_data = pd.DataFrame(df.describe(include='all'))
  s_data = s_data.drop(columns="index")
  # Using DataFrame.insert() to add a column
  """
  s_data.insert(0, "describe", ['count', 'mean',
                                'std', 'min', '25%', '50%', '75%', 'max'], True)
  """
  #temp = s_data.index.values
  #s_data.insert(0, column="data_summary", value=temp)

  # print(s_data)
  # print(s_data.columns)
  #research Ref:
  #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transpose.html
  #https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html
  #https://stackoverflow.com/questions/12168648/pandas-python-how-to-add-column-to-dataframe-for-index
  #https://pandas-docs.github.io/pandas-docs-travis/reference/api/pandas.DataFrame.reset_index.html
  #https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
  #Here we have transform the rows into the columns
  df_transposed = s_data.T  # or df1.transpose()
  #Set Serial numbers for the features vectors information as column:
  temp = range(1, len(df_transposed) + 1)
  df_transposed.insert(0, column="sl_no_fvs", value=temp)
  #Set the index to become the "Website crawling" as column:
  df_transposed = df_transposed.reset_index().set_index('index', drop=True)
  #Rename the index colum to the 'Website_crawling_IDs'
  #df2 = df.rename({'a': 'X', 'b': 'Y'}, axis='columns')
  #df2 = df.rename({'a': 'X', 'b': 'Y'}, axis=1)  # new method
  #df_transposed.index.name = None
  df_transposed.index.names = ['Website_crawling_IDs']

  #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
  #https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-feather
  #write the Pandas DataFrame into a output file in output Directory.
  WF.write_csv(df_transposed, "output")
  WF.write_ori(df_transposed, "output")

  return df_transposed



def find_callfor_IQR(URL):
  #call multi_files_names() function to get specific file name for Marged Data
  file_name = multi_files_names(URL)
  #call merge_file_path() function to enter into the directory of "file_name"
  #for find out the file.
  file_name = merge_file_path(file_name)
  #print(file_name)
  df = pd.read_csv(file_name)

  print(f"outliers: {IQR.detect_outliers_with_IQR(df['3_1_www.google.com'])}")


  return
  # while True:

  #   pass

  pass

def make_group():

  pass


def make_group():

  pass

# Ref: https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python

#find and replace strings in Multiple FIles using Python

def multi_files_names(URL):
  #print(f"URL:{URL}")
  curr_dir = os.getcwd()
  #print(curr_dir)

  #texttofind = ''
  #texttoreplace = ''
  #Find out the inputfiles list #=  #+ '/'
  # First get the current directory and then add the folder which you want to enter
  sourcepath = os.listdir(os.path.join(
      curr_dir, "data_for_plots"))  # OR read_file_path()
  delimiters = "_" #, "."
  #print(delimiters)
  #re.escape allows to build the pattern automatically and have the delimiters escaped nicely.
  regexPattern = '|'.join(map(re.escape, delimiters))
  #print(regexPattern)
  for sfile in sourcepath:
    split_file_name = re.split(regexPattern, sfile)
    if URL in split_file_name and 'website.csv' in split_file_name:
      #print(sfile)
      return sfile
    #print(split_file_name)
  #print(sourcepath)


  #pass

def merge_file_path(file_name):
  curr_dir = os.getcwd()
  #print(curr_dir)
  get_file_path = os.path.join(curr_dir, "data_for_plots")
  abs_path = os.path.join(get_file_path, file_name)
  #print(abs_path)
  return abs_path

  # # Return to the Parent Directory
  # get_parent_path = os.path.join(curr_dir, "..")

  # print(get_parent_path)

def main_simul(URL):
  df, path_to_file_name = simulation_mode(URL)
  visualization_box_plot(df)
  find_min_max_mean(URL)
  
  print("Outliers based on Z-score based on all points:")
  
  algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features = EV.find_callfor_Z_score(URL, path_to_file_name, True)
  
  DET.print_comparison(algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features)
  print("YES!!!!!!!!!!!!!!!!!!!!")
  
  print("Outliers based on IQR based on all points:")
  algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features = EV.find_callfor_IQR(URL, path_to_file_name, True)
  DET.print_comparison(algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features)
  
  print("Outliers based on Linear regression based on all points:")
  algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features = EV.find_callfor_LR(URL, path_to_file_name, True)
  DET.print_comparison(algo_name, path_to_file_name, outliers_list, number_of_outliers, df, df_removed, number_of_all_features)
  

if __name__ == "__main__":
  # csv_reader = read_csv_file()
  # write_csv_file(csv_reader)

  #multi_files_names()
  df, path_to_file_name  = simulation_mode('www.google.com')
  #print(type(df))
  visualization_box_plot(df)

  find_min_max_mean('www.google.com')
  #find_callfor_IQR('www.google.com')
  #pass
