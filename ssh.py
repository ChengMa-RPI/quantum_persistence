""" transfer files between local machine and remote server"""
import paramiko
import os 
import numpy as np 

client = paramiko.SSHClient()
client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

#client.connect('ganxis3.nest.rpi.edu', username='mac6', password='woods*score&sister')
client.connect('ganxis3.nest.rpi.edu', username='mac6', password='973366MaChengRPI')

def transfer_files(directory, filenames):

    server_des = '/home/mac6/RPI/research/quantum_se/data/' 
    local_des = '/home/mac/RPI/research/quantum_se/data/'
    if not os.path.exists(local_des):
        os.makedirs(local_des)
    sftp = client.open_sftp()
    if '/' in directory:
        if not os.path.exists(local_des + directory):
            os.makedirs(local_des + directory)
        filenames = sftp.listdir(server_des+directory) 
    for i in filenames:
        sftp.get(server_des + directory + i, local_des + directory +i)
        #sftp.put(local_des + directory +i, server_des + directory + i)
    sftp.close()


quantum_or_not = True
network_type = '1D'

if quantum_or_not:
    quantum_des = 'quantum'
else:
    quantum_des = 'classical'
des = '../data/' + quantum_des + '/meta_data/' + network_type + '/' 
des = '../data/' + quantum_des + '/state_distribution/' + network_type + '/' 
des = '../data/' + quantum_des + '/persistence/' + network_type + '/' 
if not os.path.exists(des):
    os.makedirs(des)

des = '../transfer_figure/'

transfer_files(des, [])
