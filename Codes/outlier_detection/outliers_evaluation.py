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
import z_score_algo as ZS
import linear_regression as LR

def find_callfor_LR(URL, path_to_file_name, is_Simul):
  #call multi_files_names() function to get specific file name for Marged Data
  file_name = search_marge_files_names(URL)
  #call merge_file_path() function to enter into the directory of "file_name"
  #for find out the file.
  file_name1 = merge_file_path(file_name)
  #print(file_name)
  df = pd.read_csv(file_name1)
  df_removed = df #in future here will be stored adjsted dataframe
  
  outliers_list = []
  header_list = list(df.columns)
  index = header_list[0]
  df2 = LR.get_summary_fv_ds()
  page_outliers_dict = {}
  for i in header_list[1:41]:

    # Create the pandas DataFrame
    df1 = pd.DataFrame(data=df, columns=['index', i])
    # print(len(df1.columns))
    # print(df1.head())
    # break
    outliers_RMS, max_point = LR.detect_outliers_with_LR(df1, df2)
    #print(f"count {c}: {outliers_R}")
    # break
    if outliers_RMS > max_point:
      page_outliers_dict[i] = "1"
      
      df_removed = df_removed.drop(columns=i) #remove outlier's column
      #print(f"outliers cal. with LR for Webpage {i}: {outliers_R}")
  #print(f"outliers: {IQR.detect_outliers_with_IQR(df['3_1_www.google.com'])}")
    #break
  if(len(page_outliers_dict) != 0):
    outliers_list.append(page_outliers_dict)
  if(is_Simul):
    WF.write_csv(df_removed, "lr_all_points_removed_" + file_name)
  return "LR_ALL_POINTS", path_to_file_name, outliers_list, len(outliers_list), df, df_removed, len(header_list) - 1


def find_callfor_Z_score(URL, path_to_file_name, is_Simul):
  #call multi_files_names() function to get specific file name for Marged Data
  file_name = search_marge_files_names(URL)
  #call merge_file_path() function to enter into the directory of "file_name"
  #for find out the file.
  file_name1 = merge_file_path(file_name)
  #print(file_name)
  df = pd.read_csv(file_name1)
  df_removed = df #in future here will be stored adjsted dataframe
  header_list = list(df.columns)
  outliers_list = []
  page_outliers_dict = {}
  for z in header_list[1:41]:
    outliers = ZS.detect_outliers_with_z_score(df[z])
    if len(outliers) > 0:
      #print(f"outliers cal. with Z-score for Webpage {z}: {outliers}")
      page_outliers_dict[z] = "1"
      #outliers_list.append(z)
      df_removed = df_removed.drop(columns=z) #remove outlier's column

  # print(
  #     f"outliers: {ZS.detect_outliers_with_z_score(df['2_9_www.google.com'])}")
  # print(
  #     f"outliers: {ZS.detect_outliers_with_z_score(df['3_3_www.google.com'])}")
  # print(
  #     f"outliers: {ZS.detect_outliers_with_z_score(df['3_4_www.google.com'])}")
  # print(
  #     f"outliers: {ZS.detect_outliers_with_z_score(df['1_1_www.google.com'])}")
  outliers_list.append(page_outliers_dict)
  if(is_Simul):
      WF.write_csv(df_removed, "z-score_all_points_removed_" + file_name)
  return "Z-score_ALL_POINTS", path_to_file_name, outliers_list, len(outliers_list), df, df_removed, len(header_list) - 1
  

  return

def find_callfor_IQR(URL, path_to_file_name, is_Simul):
  #call multi_files_names() function to get specific file name for Marged Data
  file_name = search_marge_files_names(URL)
  #call merge_file_path() function to enter into the directory of "file_name"
  #for find out the file.
  file_name1 = merge_file_path(file_name)
  #print(file_name)
  df = pd.read_csv(file_name1)
  df_removed = df #in future here will be stored adjsted dataframe
  header_list = list(df.columns)
  outliers_list = []
  page_outliers_dict = {}
  for i in header_list[1:41]:
    outliers = IQR.detect_outliers_with_IQR(df[i])
    if len(outliers) > 0:
      page_outliers_dict[i] = "1"
      #outliers_list.append(i)
      df_removed = df_removed.drop(columns=i) #remove outlier's column
      #print(f"outliers cal. with IQR for Webpage {i}: {outliers}")
  #print(f"outliers: {IQR.detect_outliers_with_IQR(df['3_1_www.google.com'])}")
  outliers_list.append(page_outliers_dict)
  if(is_Simul):
    WF.write_csv(df_removed, "lqr_all_points_removed_" + file_name)
  return "IQR_ALL_POINTS", path_to_file_name, outliers_list, len(outliers_list), df, df_removed, len(header_list) - 1

# Ref: https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python

#find and replace strings in Multiple FIles using Python


def search_marge_files_names(URL):
  #print(f"URL:{URL}")
  curr_dir = os.getcwd()
  #print(curr_dir)

  #texttofind = ''
  #texttoreplace = ''
  #Find out the inputfiles list #=  #+ '/'
  # First get the current directory and then add the folder which you want to enter
  sourcepath = os.listdir(os.path.join(
      curr_dir, "data_for_plots"))  # OR read_file_path()
  delimiters = "_"  # , "."
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


if __name__ == "__main__":
  # csv_reader = read_csv_file()
  # write_csv_file(csv_reader)

  #multi_files_names()
  # df = simulation_mode('www.google.com')
  # #print(type(df))
  # visualization_box_plot(df)

  # find_min_max_mean('www.google.com')
  # find_callfor_IQR('www.google.com')
  # find_callfor_Z_score('www.google.com')
  find_callfor_LR('www.google.com')
  #pass
