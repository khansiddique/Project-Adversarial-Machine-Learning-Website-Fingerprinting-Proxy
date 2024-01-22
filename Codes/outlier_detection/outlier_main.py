import argparse
import os
import outlier_detection as Out

#Siddique's libraries
import outliers_simulation as OLS

def numOfPassedArguments(args):
  sum = 0;
  for arg in vars(args):
    if(getattr(args, arg) != None):
      sum += 1
      #print(arg)
  return sum

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="File Name")
  
  parser.add_argument("-l", "--logfile", action='store_const', 
    const="/home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/logs", 
    required=False, help="Shows log file")
  #'store_const' - This stores the value specified by the const keyword argument. The 'store_const' action is most commonly used with optional arguments that specify some sort of flag.
  parser.add_argument("-s", "--simulation", required=False, help="Simulation mode")

  parser.add_argument("-i", "--input_file", required=False, help="Single feature file to detect/remove outliers from.")

  parser.add_argument("-a", "--algorithm_name", required=False, help="Name of the Outlier detection algorithm")
  parser.add_argument("-n", "--min_number_to_remove", type=int, required=False, help="Minimum number of feature vectors to remove")
  parser.add_argument("-N", "--max_number_to_remove", type=int, required=False, help="Maximum number of feature vectors to remove")
  parser.add_argument("-v", "--variance_boundary", required=False, help="Upper boundary of variance after outlier removal")
  
  #parser.add_argument("-o", "--output_file", required=False, help="File to store the adjusted features.")
  
  args = parser.parse_args()

  numOfInit = numOfPassedArguments(args)
  #print(numOfInit)
  if(args.logfile != None and numOfInit == 1):
    print(os.listdir(args.logfile)) #return a list of all files and directories in "somedirectory".
  elif(args.simulation != None and numOfInit == 1):
    print("simulation")
    Out.simulation_mode(args.simulation, "output")
    OLS.main_simul(args.simulation)
  elif((args.algorithm_name != None and args.min_number_to_remove != None  and args.input_file != None) and (args.max_number_to_remove != None or args.variance_boundary != None) and numOfInit == 4):
    print("algorithm_running")
    variance = -1
    max_number = -1
    if(args.max_number_to_remove != None):
      max_number = args.max_number_to_remove
    elif(args.variance_boundary != None):
      variance = args.variance_boundary

    with open("urls.txt") as f:
      content = f.readlines()
    content = [x.strip().split('/')[2] for x in content]
    for index, url in enumerate(content):
      Out.algorithm_running(args.algorithm_name, args.min_number_to_remove, max_number, variance, url)
     # break
    #Out.simulation_mode(args.input_file, args.output_file)
  else:
    print("please, follow the rule: -l OR (-s and -i and -o) OR ((-a and -n and -i and -o) AND (-N or -V))")

# for showing logfiles:
# python3 outlier_main.py -l

# for simulation:
# python3 outlier_main.py -s "www.google.com"

# for algorithm_running:
# python3 outlier_main.py -i "www.google.com" -a "IQR_CUMUL_SUM" -n 0 -N 3
# python3 outlier_main.py -i "www.google.com" -a "IQR_CUMUL_SUM" -n 0 -v 4476007226991

# python3 outlier_main.py -i "www.google.com" -a "Z-score_CUMUL_SUM" -n 0 -N 3
# python3 outlier_main.py -i "www.google.com" -a "Z-score_CUMUL_SUM" -n 0 -v 4476007226991

# python3 outlier_main.py -i "www.google.com" -a "Z-score_ALL_POINTS" -n 0 -N 3

