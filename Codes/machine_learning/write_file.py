import pandas as pd
import os


def get_file_path(dir_name):
  curr_dir = os.getcwd()
  #print(curr_dir)
  abs_path = os.path.join(curr_dir, dir_name)

  #abs_path = os.path.join(get_file_path, file_name)
  #print(abs_path)
  return abs_path

#research Ref:
#https://datatofish.com/export-dataframe-to-csv/
def write_csv(df, file_name):
  #create the absolute file path for save the summary data.
  path = get_file_path("output")
  file_name = os.path.join(path, file_name)
  #file_name = get_file_path("output") + '\feature_vectors.csv'
  #print(file_name)
  # Don't forget to add '.csv' at the end of the path
  export_csv = df.to_csv(file_name, index=True, header=True)

# It is designed to make reading and writing data frames efficient,
# and to make sharing data across data analysis languages easy.
# pip install pyarrow --user
def write_ori(df, file_name):
  #create the absolute file path for save the summary data.
  path = get_file_path("output")
  file_name = os.path.join(path, file_name)
  #file_name = get_file_path("output") + '\' feature_vectors.ori'
  #print(file_name)
  # Don't forget to add '.ori' at the end of the path
  #Write to a feather file.
  export_csv = df.to_csv(file_name, index=True, header=True)
