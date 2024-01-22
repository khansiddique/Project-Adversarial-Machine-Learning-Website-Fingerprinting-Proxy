from scapy.all import *
import log_dataset_csv as csv
import getIPaddress as GIP
import uuid
import numpy as np
from scipy import interpolate
import itertools
import os

def get_src_mac():
  #extracting MAC usin UUID module
  return ':'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0,8*6,8)][::-1])

def sampling_interpolation(list_index, list_cumul):
  #it will extrapolate to accomadate these values of your's that lie outside the interpolation range
  f = interpolate.interp1d(list_index, list_cumul, axis=-1, fill_value="extrapolate")
  # taking 50 points with a period len(list_index) / 50.0
  xnew = np.arange(0, len(list_index) - 1, len(list_index) / 50.0)
  ynew = f(xnew)
  # print(ynew)
  # print(len(ynew))
  return ynew

def extract_TLS(file_name, src_mac, input_folder, output_folder):
  #internal_ip = GIP.get_inteface_ip(interface_name)
  input_path = input_folder + "/" + file_name
  load_layer('tls')
  pkts = rdpcap(input_path)
  filtered = (pkt for pkt in pkts if TLS in pkt)

  #output_path = output_folder + "/" + file_name
  output_folder_for_tls = "/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/tls_pcap_files"
  filtered_tls_pcap = output_folder_for_tls + "/" + file_name + '_filtered_tls.pcap'
  wrpcap(filtered_tls_pcap, filtered)
  ex_pcaps = rdpcap(filtered_tls_pcap)
  
  list_index = [] # list of index = x value
  list_cumul = [] # cumul length = y value
  tls_list = []
  cumul = 0
  index = 0
  for ex_pcap in ex_pcaps:
    index += 1
    #ex_pcap[0].show()
    if(src_mac == ex_pcap[0][Ether].src): #Ether == Ethernet
      direction = 0 # outgoing traffic
      cumul -= ex_pcap[0][TLS].len
    else:
      direction = 1 # incoming traffic
      cumul += ex_pcap[0][TLS].len
    list_index.append(index)
    list_cumul.append(cumul)
    tls_list.append([index, ex_pcap[0].time, ex_pcap[0][IP].src, ex_pcap[0][IP].dst, ex_pcap[0][TCP].seq, ex_pcap[0][TCP].ack, ex_pcap[0][TLS].version, ex_pcap[0][TLS].len, direction, cumul]);
  # print(list_index)
  # print(list_cumul)
  return tls_list, list_index, list_cumul

def split_files_to_array(input_files):
  files_list = []
  for id1, file1 in enumerate(input_files):    
    file = []
    for id1, file2 in enumerate(input_files):
      if file1.split('_')[0] == file2.split('_')[0]:
        file.append(file2)
    files_list.append(file)
  files_list.sort()
  #itertools is module which checks and removes the same lists inside list 
  all_lists = list(files_list for files_list,_ in itertools.groupby(files_list))
  
  main_list = []
  sub_main_list = []
  t = all_lists[0][0].split("_")[2]
  for group_of_pages in all_lists:
    if group_of_pages[0].split("_")[2] == t:
      sub_main_list.append(group_of_pages)
    else:
      t = group_of_pages[0].split("_")[2]
      main_list.append(sub_main_list)
      sub_main_list = []
      sub_main_list.append(group_of_pages)
  main_list.append(sub_main_list)
  return main_list





def create_seq_header(file_names_for_cumul):
  seq_header = ['index']
  for index, file_name in enumerate(file_names_for_cumul):
    #column_name = "cumul" + str(index + 1) + file_name
    seq_header.append(file_name)
  return seq_header

def extract_features(input_folder, output_folder):  
  input_files = os.listdir(input_folder)
  input_files.sort()

  main_list = split_files_to_array(input_files)
  print(main_list)

  output_folder_for_plots = "/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/data_for_plots"
  
  for website_index, file_groups in enumerate(main_list):
    seq_list_for_one_website = []
    file_names_for_website = []
    for group_index, group in enumerate(file_groups):
      seq_list_for_one_page = []
      file_names_for_webpage = []
      for file_index, file_name in enumerate(group):
        meta_data, list_index, list_cumul = extract_TLS(file_name, get_src_mac(), input_folder, output_folder)
        header = ['index', 'time', 'src_ip', 'dst_ip', 'tcp_seq', 'tcp_ack', 'tls_version', 'tls_len', 'direction', 'cumul_sum']
        features_path =  output_folder + "/" + file_name + "_tls.csv"
        csv.write_csv_file(meta_data, header, features_path)
        seq = sampling_interpolation(list_index, list_cumul)
        seq_list_for_one_page.append(seq)
        file_names_for_webpage.append(file_name)
      
      file_names_for_website.extend(file_names_for_webpage)
      seq_list_for_one_website.extend(seq_list_for_one_page)
      # file_names_for_website.append(group[0].split("_")[2])
      # print(file_names_for_website)
      
      seq_header = create_seq_header(file_names_for_webpage);
      merge_path_for_one_page = output_folder_for_plots + "/" + str(group[0].split("_")[0]) + "_merge_page.csv"
      csv.write_csv_file_as_columns(seq_list_for_one_page, seq_header, merge_path_for_one_page)

    website_seq_header = create_seq_header(file_names_for_website);
    merge_path_for_one_website = output_folder_for_plots + "/" + str(website_index + 1) + "_merge_website.csv"
    csv.write_csv_file_as_columns(seq_list_for_one_website, website_seq_header, merge_path_for_one_website)

  return seq_list_for_one_page


#################################################################################

def split_files_to_array_from_log_file(log_file_path):
  log_file_list = csv.read_csv_file(log_file_path)
  #print(log_file_list)

  page_counter = 0
  site_counter = 0
  page_list = []
  site_list = []
  main_list = []
  for index1, crawling in enumerate(log_file_list):
    page_list.append(crawling[1])
    if (index1 + 1) % 10 == 0:
      site_list.append(page_list)
      page_list = []
    if (index1 + 1) % 40 == 0:
      main_list.append(site_list)
      site_list = []
  #print(main_list)
  return main_list

def extract_TCP_features(input_folder, output_folder):  
  input_files = os.listdir(input_folder)
  input_files.sort()

  os.chdir(os.path.dirname(os.getcwd()))
  curr_dir = os.getcwd()

  output_folder_for_plots = os.path.join(curr_dir, "data_for_plots")
  output_folder_for_ML = os.path.join(curr_dir, "machine_learning/data_for_ML")
  log_file_path = os.path.join(curr_dir, "logs")

  main_list = split_files_to_array_from_log_file(log_file_path + '/log_files_name.csv')

  counter = 129
  cumul_features = []
  for website_index, file_groups in enumerate(main_list):
    seq_list_for_one_website_ML = []
    seq_list_for_one_website = []
    file_names_for_website = []
    for group_index, group in enumerate(file_groups):
      seq_list_for_one_page_ML = []
      seq_list_for_one_page = []
      file_names_for_webpage = []
      for file_index, file_name in enumerate(group):
        print(file_name)
        meta_data, list_index, list_cumul, number_of_outgoing,  number_of_incoming, sum_of_outgoing, sum_of_incoming = extract_TLS_from_TCP(file_name, get_src_mac(), input_folder, output_folder)
        header = ['index', 'time', 'src_ip', 'dst_ip', 'tcp_seq', 'tcp_ack', 'tcp_len', 'direction', 'cumul_sum']
        features_path =  output_folder + "/" + file_name + "_tls.csv"
        csv.write_csv_file(meta_data, header, features_path)
        seq = sampling_interpolation(list_index, list_cumul)
        seq_list_for_one_page.append(seq)
        seq_list_for_one_page_ML.append(list(seq) + [number_of_outgoing, number_of_incoming, sum_of_outgoing, sum_of_incoming])
        file_names_for_webpage.append(file_name)

        for i, v in enumerate(seq):
          cumul_features.append([counter, group_index + 1, i + 1, v])
          counter += 1
      
      file_names_for_website.extend(file_names_for_webpage)
      seq_list_for_one_website.extend(seq_list_for_one_page)
      seq_list_for_one_website_ML.extend(seq_list_for_one_page_ML)
      # file_names_for_website.append(group[0].split("_")[2])
      # print(file_names_for_website)
      
      seq_header = create_seq_header(file_names_for_webpage);
      merge_path_for_one_page = output_folder_for_plots + "/" + str(group[0].split("_")[0]) + "_" + str(group[0].split("_")[2]) + "_merge_page.csv"
      csv.write_csv_file_as_columns(seq_list_for_one_page, seq_header, merge_path_for_one_page)

    website_seq_header = create_seq_header(file_names_for_website);
    merge_path_for_one_website = output_folder_for_plots + "/" + str(website_index + 1)  + "_" + str(group[0].split("_")[2]) + "_merge_website.csv"
    csv.write_csv_file_as_columns(seq_list_for_one_website, website_seq_header, merge_path_for_one_website)

    merge_path_for_one_website_ML = output_folder_for_ML + "/" + str(website_index + 1)  + "_" + str(group[0].split("_")[2]) + "_merge_website_ML.csv"
    csv.write_csv_file_as_columns(seq_list_for_one_website_ML, website_seq_header, merge_path_for_one_website_ML)


  header_for_ML = ['row_counter', 'identifier', 'index', 'time', 'cumul_sum']
  cumul_features_path =  output_folder + "/cumul_feature_file.csv"
  csv.add_header_in_file(header_for_ML, cumul_features_path)
  csv.append_csv_file(cumul_features, cumul_features_path)

  return seq_list_for_one_page

def extract_TLS_from_TCP(file_name, src_mac, input_folder, output_folder):
  #internal_ip = GIP.get_inteface_ip(interface_name)
  src_mac = '08:00:27:d6:38:49'
  input_path = input_folder + "/" + file_name
  load_layer('tcp')
  pkts = rdpcap(input_path)
  filtered = (pkt for pkt in pkts if TCP in pkt)
  
  curr_dir = os.getcwd()
  output_folder_for_tls = os.path.join(curr_dir, "tls_pcap_files")
  filtered_tls_pcap = output_folder_for_tls + "/" + file_name + '_filtered_tls.pcap'
  wrpcap(filtered_tls_pcap, filtered)
  ex_pcaps = rdpcap(filtered_tls_pcap)
  

  number_of_outgoing = 0
  number_of_incoming = 0
  sum_of_outgoing = 0
  sum_of_incoming = 0

  list_index = [] # list of index = x value
  list_cumul = [] # cumul length = y value
  tls_list = []
  cumul = 0
  index = 0
  for ex_pcap in ex_pcaps:
    try:
      t = len(ex_pcap[0][Raw])
    except Exception as e:
      continue
    index += 1
    #print(index)
    # ex_pcap[0].show()
    # print(len(ex_pcap[0][Raw]))

    if(src_mac == ex_pcap[0][Ether].src): #Ether == Ethernet
      direction = 0 # outgoing traffic
      cumul -= len(ex_pcap[0][Raw])
      number_of_outgoing += 1
      sum_of_outgoing += len(ex_pcap[0][Raw])
    else:
      direction = 1 # incoming traffic
      cumul += len(ex_pcap[0][Raw])
      number_of_incoming += 1
      sum_of_incoming += len(ex_pcap[0][Raw])
    list_index.append(index)
    list_cumul.append(cumul)
    tls_list.append([index, ex_pcap[0].time, ex_pcap[0][IP].src, ex_pcap[0][IP].dst, ex_pcap[0][TCP].seq, ex_pcap[0][TCP].ack, len(ex_pcap[0][Raw]), direction, cumul]);
  # print(list_index)
  # print(list_cumul)
  return tls_list, list_index, list_cumul, number_of_outgoing,  number_of_incoming, sum_of_outgoing, sum_of_incoming
if __name__ == '__main__':
  extract_TCP_TLS("1_1_www.google.com", get_src_mac(), "/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/pcap_files", "/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/features_files")