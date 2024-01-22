import argparse
import ML_algo as ML

def numOfPassedArguments(args):
  sum = 0;
  for arg in vars(args):
    if(getattr(args, arg) != None):
      sum += 1
  return sum

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="File Name")

  parser.add_argument("-a", "--algo", required=False, help="Machine Learning algorithm name")
  
  args = parser.parse_args()

  numOfInit = numOfPassedArguments(args)

  if(args.algo != None and numOfInit == 1):
    ML.algorithms_selection(args.algo)
  else:
    print("please, do follow the following rule: -a")

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