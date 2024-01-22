import socket
import netifaces as ni

def get_host_ip():
  try:
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    return host_ip
  except socket.error as e:
    return str(e)

def get_remote_ip(remote_url):
  try:
    u = remote_url
    remote_ip = socket.gethostbyname(u.split('/')[2])
    return remote_ip
  except socket.error as e:
    #return str(e)
    return '192.168.1.4'

def get_ips(remote_url, interface_name):
    return [get_inteface_ip(interface_name), get_remote_ip(remote_url)]

def get_inteface_ip(interface_name):
  ni.ifaddresses(interface_name)
  ip = ni.ifaddresses(interface_name)[ni.AF_INET][0]['addr']
  return ip

if __name__ == "__main__":
  print(get_inteface_ip('enp0s3'))
  print(get_ips('https://www.google.com/'))