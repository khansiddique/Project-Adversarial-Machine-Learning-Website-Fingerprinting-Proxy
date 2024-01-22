import csv
import os

def read_csv_file(file_name):
  with open(file_name, "r") as csv_file:
    # For creating a object which holds data of csv file
    csv_reader = csv.reader(csv_file)

    # skip the iteration just for single row
    # We can call the next method().
    next(csv_file)

    csv_data_list = []

    # Iterate over for printing all the data in CSV files
    for line in csv_reader:
      csv_data_list.append(line)
      print(line)

  return csv_data_list


def write_csv_file(csv_reader, header, file_name):

  # Open a CSV file in write mode to write data with
  # comma seperated value : i.e. delimiter=","

  # write the header
  #header = ['fname', 'lname', 'email']
  try:
    with open(file_name, "w", newline='') as csv_file:
      csv_writer = csv.writer(csv_file, delimiter=",")

      csv_writer.writerow(header)
      # write the actual content line by line
      # Iterate over for writing all the data in CSV file
      for line in csv_reader:
        csv_writer.writerow(line)
  except Exception as e:
    print(e)

# write csv for ML file with append functionality
def append_csv_file(csv_reader, file_name):
  try:
    with open(file_name, "a+", newline='') as csv_file:
      csv_writer = csv.writer(csv_file, delimiter=",")
      for line in csv_reader:
        csv_writer.writerow(line)
        #print(line)
  except Exception as e:
    print(e)

# write header functionality
def add_header_in_file(header, file_name):
  try:
    with open(file_name, "a+", newline='') as csv_file:
      csv_writer = csv.writer(csv_file, delimiter=",")
      if (os.stat(file_name).st_size == 0):
        csv_writer.writerow(header)
  except Exception as e:
    print(e)

def write_header_csv_file(header, file_name):
  try:
    with open(file_name, "a+", newline='') as csv_file:
      csv_writer = csv.writer(csv_file, delimiter=",")
      csv_writer.writerow(header)
  except Exception as e:
    print(e)

def write_csv_file_as_columns(csv_reader, header, file_name):
  try:
    with open(file_name, "w", newline='') as csv_file:
      csv_writer = csv.writer(csv_file, delimiter=",")
      csv_writer.writerow(header)
      # write the actual content line by line
      # Iterate over for writing all the data in CSV file
      for seq_number in range(len(csv_reader[0])):
        line = [str(seq_number + 1)]
        for seq_index, seq in enumerate(csv_reader):
          line.append(seq[seq_number])
        csv_writer.writerow(line)
  except Exception as e:
    print(e)

if __name__ == "__main__":
  # csv_reader = read_csv_file()
  # write_csv_file(csv_reader)
  pass

