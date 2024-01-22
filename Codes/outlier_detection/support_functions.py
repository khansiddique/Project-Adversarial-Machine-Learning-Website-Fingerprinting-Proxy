import os

def multi_files_names(URL):
  #print(f"URL:{URL}")
  curr_dir = os.getcwd()
  # First get the current directory and then add the folder which you want to enter
  sourcepath = os.listdir(os.path.join(curr_dir, "data_for_plots"))  # OR read_file_path()
  delimiters = "_"
  #re.escape allows to build the pattern automatically and have the delimiters escaped nicely.
  regexPattern = '|'.join(map(re.escape, delimiters))
  for sfile in sourcepath:
    split_file_name = re.split(regexPattern, sfile)
    if URL in split_file_name and 'ML.csv' in split_file_name:
      #print(sfile)
      return sfile
    #print(split_file_name)
  #print(sourcepath)


def merge_file_path(file_name):
  curr_dir = os.getcwd()
  get_file_path = os.path.join(curr_dir, "data_for_plots")
  abs_path = os.path.join(get_file_path, file_name)
  #print(abs_path)
  return abs_path