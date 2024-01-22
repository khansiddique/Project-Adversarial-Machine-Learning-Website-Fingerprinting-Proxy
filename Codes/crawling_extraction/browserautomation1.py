from time import time
from random import randrange
import json
from selenium import webdriver
import getIPaddress as GIP
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import log_dataset_csv as csv
import browserautomation as BA
import getIPaddress as GIP
import socket
import os
import signal
import subprocess
import re

def main(file_urls, num_instances, net_interface, output_folder):
    with open(file_urls, 'r') as c:
      c_contents = c.read().splitlines()
      
      header = ['row_counter', 'ID', 'index', 'full_url', 'start_time', 'end_time', 'load_time', 'website_loading_error', 'local_ip', 'remote_ip']
      header_for_files = ['identifier', 'file_name', 'full_url']

      path_to_log_folder = "/home/khan/website_fingerprinting_proxy/workspace/codes/logs/"
      csv.add_header_in_file(header, path_to_log_folder + "log_dataset.csv")
      csv.add_header_in_file(header_for_files, path_to_log_folder + "log_files_name.csv")

      row_counter = 2451
      identifier = 246
      count = 0
      depth = 4 #depth of sub-paging
      for url in c_contents:
        if count < 14:
          links_list = BA.define_path(url, depth)
          print(links_list)
          for link_in_path in links_list:
            runnumber = 1 # number of crawling per one url
            while runnumber != num_instances + 1:
              print(link_in_path)
              file_name = str(identifier) + "_" + str(runnumber) + "_" + url.split('/')[2]
              path = output_folder + "/" + file_name
              #sudo tcpdump -i wlan0 -s0 -w /root/file_name
              p = subprocess.Popen(('sudo', 'tcpdump', '-i', net_interface, '-s0', '-w', path), stdout=subprocess.PIPE, preexec_fn=os.setsid)
              error_list = BA.crawl_website(link_in_path)
              print(error_list[3])
              if(error_list[3] != "ok"):
                os.remove(path)
                print("killed")
                os.system("sudo kill %s" % os.getpgid(p.pid))
                # to crawl again in case of error
                continue
              #os.killpg(os.getpgid(p.pid), signal.SIGTERM)
              os.system("sudo kill %s" % os.getpgid(p.pid))

              ip_list = GIP.get_ips(url, net_interface)

              csv.append_csv_file([[row_counter, identifier, runnumber, link_in_path] + error_list + ip_list], path_to_log_folder + "log_dataset.csv")
              csv.append_csv_file([[identifier, file_name, link_in_path]], path_to_log_folder + "log_files_name.csv")

              row_counter += 1
              runnumber += 1
            identifier += 1
        count += 1
      q = subprocess.Popen(('sudo', 'chown', '-R', 'khan', '/home/khan/website_fingerprinting_proxy/workspace/codes/pcap_files'), stdout=subprocess.PIPE, preexec_fn=os.setsid)
      #os.system("sudo kill %s" % os.getpgid(q.pid))
      print("Successfully finished!")
      
#sudo chown -R $USER /home/khan/Desktop/website_fingerprinting_proxy-stable/workspace/pcap_files

def links_from_website(url):
  links = []
  try:
    cap = DesiredCapabilities.CHROME
    cap['goog:loggingPrefs'] = {'browser': 'ALL', 'performance': 'ALL'}
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--disable-application-cache')
    #chrome_options.add_argument('--no-sandbox')
    
    driver = webdriver.Chrome(desired_capabilities=cap, options=chrome_options, executable_path="/usr/bin/chromedriver")
    driver.delete_all_cookies() 
    driver.implicitly_wait(10)
    
    driver.get(url)
    for a in driver.find_elements_by_xpath('.//a'):
      # z = re.match('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', a.get_attribute('href'))
      # if z:
      #   links.append(a.get_attribute('href'))
      links.append(a.get_attribute('href'))
    
    driver.close()
    return ["ok"], links
  except StaleElementReferenceException as ex:
    driver.refresh()
    try:
      for a in driver.find_elements_by_xpath('.//a'):
        links.append(a.get_attribute('href'))
      driver.close()
    except Exception as e:
      driver.close()
      return [str(e)], links
    return ["ok"], links
  except Exception as e:
    driver.close()
    return [str(e)], links

def crawl_website(url):
  links = []
  start_time = time()
  try:
    cap = DesiredCapabilities.CHROME
    cap['goog:loggingPrefs'] = {'browser': 'ALL', 'performance': 'ALL'}
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--disable-application-cache')
    #chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(desired_capabilities=cap, options=chrome_options, executable_path="/usr/bin/chromedriver")
    driver.delete_all_cookies()
    
    start_time = time()
    driver.maximize_window()
    driver.implicitly_wait(10)
    driver.get(url)
    end_time = time()

    path_to_log_folder = "/home/khan/website_fingerprinting_proxy/workspace/codes/logs/"
    sample = open(path_to_log_folder + 'samplefile.txt', 'a+') 
    sample.write(json.dumps(driver.get_log('performance'), indent=2))
    driver.close()
    sample.close()
    return [start_time, end_time, end_time - start_time, "ok"]
  except Exception as e:
    return [start_time, "null", "null", str(e)]

def define_path(url, depth):
  u = url
  d = depth - 1
  links_list = [] #path
  links_list.append(u)
  while d > 0:
    print(u)
    list1, list2 = links_from_website(u)
    print(len(list2))
    print(list2)
    #print(list1)
    l = len(list2)
    if (l == 0 or list1[0] != "ok"):
      d = depth - 1
      u = url
      continue
    else:
      u = checkUrl(list2, url)
      
    links_list.append(u)
    print(links_list)
    d -= 1
    if(d == 0 and check_links(links_list) == False):
      d = depth - 1
      u = url
      links_list = []
      links_list.append(url)
  return links_list

def checkUrl(list2, url):
	time = 10
	while(time != 0):
		#Change the condition for None Error
		u = list2[randrange(len(list2))]
		if u != None:
			if(u.split('/')[2] == url.split('/')[2]):
				return u
		time -= 1
	return list2[randrange(len(list2))]

def check_links(links_list):
  for id1, url1 in enumerate(links_list):
    for id2, url2 in enumerate(links_list):
       if(url1 == url2 and id1 != id2):
        return False
  return True



