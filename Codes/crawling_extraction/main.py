import argparse
import log_dataset_csv as csv
import browserautomation as BA
import getIPaddress as GIP
import socket
import os
import signal
import subprocess
import glob as gl
import extract_features as EX
import plot as PL

def numOfPassedArguments(args):
  sum = 0;
  for arg in vars(args):
    if(getattr(args, arg) != None):
      sum += 1
  return sum

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="File Name")
  

  parser.add_argument("-l", "--logfile", action='store_const', 
    const="/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/logs", 
    required=False, help="Shows log file")
  #'store_const' - This stores the value specified by the const keyword argument. The 'store_const' action is most commonly used with optional arguments that specify some sort of flag.
  parser.add_argument("-v", "--diagram", required=False, help="Give input file for extracting features")

  parser.add_argument("-c", "--file", required=False, help="Input file for page crawling")
  parser.add_argument("-n", "--number", type=int, default=3, required=False, help="Number of crawling per one page")
  parser.add_argument("-i", "--interface", required=False, help="Name of interface")
  parser.add_argument("-o", "--pcapout", required=False, help="Output folder path for storing pcap-files")
  
  parser.add_argument("-e", "--pcapin", required=False, help="Input folder path for extracting features")
  parser.add_argument("-s", "--featurespath", required=False, 
    help="Output folder path for storing extracted features")
  
  args = parser.parse_args()

  numOfInit = numOfPassedArguments(args)

  if(args.logfile != None and numOfInit == 2):
    print(os.listdir(args.logfile)) #return a list of all files and directories in "somedirectory".
  elif(args.diagram != None and numOfInit == 2):
    print("diagram")
    PL.diagram(args.diagram)
  elif((args.file != None and args.number != None and args.interface != None and args.pcapout != None) and numOfInit == 4):
    print("crawling")
    BA.main(args.file, args.number, args.interface, args.pcapout)
  elif((args.pcapin != None and args.featurespath != None) and numOfInit == 3):
    print("extracting features")
    EX.extract_TCP_features(args.pcapin, args.featurespath)
	#EX.extract_features(args.pcapin, args.featurespath)
  else:
    print("please, do follow the following rule: -l or -v or (-c and -n and -i and -o) or (-e and -s)")

# for showing logfiles:
# python3 main.py -l

# for crawling:
# python3 main.py -c "/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/urls.txt" -n 10 -i "enp0s3" -o "/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/pcap_files"

# for extracting:
# python3 main.py -e "/root/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/pcap_files" -s "/root/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/features_files"
# python3 main.py -e "/home/batyi/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/pcap_files" -s "/home/batyi/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/features_files"

# for diagram of webpage:
# python3 main.py -v "/root/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/data_for_plots/1_www.google.com_merge_page.csv"

# for diagram of website:
# python3 main.py -v "/root/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/data_for_plots/1_www.google.com_merge_website.csv"

# sudo chown -R $USER /root/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/pcap_files