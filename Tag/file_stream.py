import socket               # Import socket module
import subprocess
s = socket.socket()         # Create a socket object
host = 'localhost' # Get local machine name
port = 12345                 # Reserve a port for your service.
s.connect((host, port)) 

f = open('3/example_350.dat','rb')
l = f.read()
input("Press enter to start...")
HOST="pi@192.168.10.10"
# Ports are handled in ~/.ssh/config since we use OpenSSH
COMMAND=r"python3 /home/pi/Desktop/parse_switch_seq.py /home/pi/Desktop/example_350.txt"

ssh = subprocess.Popen(["ssh", "%s" % HOST, COMMAND],
                       shell=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
while (l):
    s.send(l)
    l = f.read()
f.close()
s.close()
print(str(ssh.communicate()))
ssh.terminate()