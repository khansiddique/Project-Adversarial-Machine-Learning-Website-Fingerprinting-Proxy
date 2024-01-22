import csv
import os

# https://stackoverflow.com/questions/16383882/how-to-skip-2-lines-in-a-file-with-python


def read_init_param_file(file_name=""):

    direct = "./logs"
    file_name = os.path.join(direct, 'init_parameter.txt')
    #print(file_name)
    with open(file_name, "r") as txt_file:
        # I usually use next() when I want to skip a single line, usually a header for a file.
        next(txt_file)  # skip 1 line
        # txt_file.readline()
        # read_data = txt_file.read()
        # print(read_data)
        data_list = []

        # Iterate over for printing all the data in txt files
        for line in txt_file: #.readlines():
            #https: // stackoverflow.com/questions/9347419/python-strip-with-n
            #But these don't modify the line variable...they just return the string with the \n and \t stripped.

            data_list.extend([int(number) for number in line.split(",")])
            #data_list.append(line)
            #print(line)
        print(data_list)
    return data_list


if __name__ == "__main__":

    #https://stackoverflow.com/questions/431684/how-do-i-change-directory-cd-in-python
    #https: // stackoverflow.com/questions/13223737/how-to-read-a-file-in-other-directory-in-python

    direct = "./logs"
    abs_path = os.path.join(direct, 'init_parameter.txt')
    #print(abs_path)
    read_init_param_file()
    pass
