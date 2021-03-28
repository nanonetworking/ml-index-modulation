# -*- coding: utf-8 -*-
"""
@author: Ozgur Kara
"""

from os.path import isfile, join
from os import listdir
import csv
import numpy as np
import os
import random
from util import sph2cart

pi = np.pi

class DataLoader():
    def __init__(self, opt):
        self.training_path = opt.training_path
        self.time = opt.simulation_time
        self.window_numbers = opt.window_numbers
        self.dataset_size = opt.dataset_size
        self.downsample_rate = opt.downsample_rate
        self.training_mol_num = opt.training_mol_num
    def _read_file(self, filepath):
        tri = []
        temp = []
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            i = 1
            for row in reader:
                temp.append(row)
                if(i%3==0):
                    tri.append(np.array(temp,dtype=float))
                    temp = []
                i=i+1
                if(i==-1): #3n + 1 for result
                    break
        return tri    
    
    
    def read_folder(self):
        folder_path = self.training_path
        tri = []
        files_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        for i in files_list:
            if(i[-1] != "v"):
                continue
            tri2 = self._read_file(folder_path + i)
            tri.extend(tri2)
        data = {}
        for i in range(1,9):
            data[str(i)] = []
        for lis in tri:
            data[str(lis[0,-1])[0]].append(lis)            
        return data
    
    def _preprocess_data(self,  tri):
        time = self.time
        downsample = self.downsample_rate
        size = time / downsample
        output = np.zeros((len(tri),8 + 1,int(size)))
#        print(output.shape)
        classes = np.zeros((len(tri),1))
        output[:,8,:] = np.linspace(0,time - downsample,int(size))
        for x,lis in enumerate(tri):
            if(x%100==0):
                print(str(x) + "/" + str(len(tri)))
            i = 0
            for timex in range(int(size)):
                while((timex * downsample) <= lis[2,i] and (lis[2,i] < (timex + 1) * downsample) and (i <= lis.shape[1]-2)):
                    _, y, z = sph2cart(lis[0,i], lis[1,i], 5)
                    aci = np.arctan2(y,z)
                    if(aci < pi/8 and aci >= -pi/8):
                        output[x,2,timex] += 1
                    elif (aci >= pi/8 and aci < 3*pi/8):
                        output[x,1,timex] += 1
                    elif (aci >= 3*pi/8 and aci < 5*pi/8):
                        output[x,0,timex] += 1
                    elif (aci >= 5*pi/8 and aci < 7*pi/8):
                        output[x,7,timex] += 1
                    elif (aci >= 7*pi/8 and aci <= pi) or ( aci < -7*pi/8):
                        output[x,6,timex] += 1
                    elif (aci >= -7*pi/8 and aci < -5*pi/8):
                        output[x,5,timex] += 1 
                    elif (aci >= -5*pi/8 and aci < -3*pi/8):
                        output[x,4,timex] += 1
                    elif (aci >= -3*pi/8 and aci < -pi/8):
                        output[x,3,timex] += 1 
                    i += 1
            classes[x,0] = lis[0,-1] 
        return classes, output    
    
    
    def training_data_creater(self, data):
        time = self.time
        for numberr in self.window_numbers:
            window_size = self.time / numberr 
#            window = int(window_size*10)
            upper_limit = max(4,numberr)
            
            data_folder = "window_" + str(window_size) + "_upper_" + str(upper_limit) + "\\"
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            isi_data = []
            train_size = self.dataset_size
            classes = np.zeros((train_size,int(numberr)))
            for j in range(0,train_size):
                z = np.zeros((1,int(numberr)))
                if(j%100==0):
                    print("j equals to: " + str(j) + "/" + str(train_size))
                for i in range(0,int(numberr)):
                    rand = random.randint(1,8)
                    z[0,i] = rand
                    rand2 = random.randint(0,149)
                    x = data[str(rand)][rand2]
                    if(i == 0):
                        isi_data.append(np.squeeze(x[:,np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))]))
                        
                    else:
                        length = len(np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))[0])
                        temp = np.zeros((3,length))
                        row = np.ones((1,length)) * ((i)*window_size)
                        temp[2,:] = row
                        y = np.squeeze(x[:,np.where((x[2,:-1] + ((i)*window_size) < (window_size*(i+upper_limit))) & (x[2,:-1] + ((i)*window_size) < (time)))]) + temp
                        isi_data[j] = np.hstack((isi_data[j], y))
                isi_data[j] = isi_data[j][ :, isi_data[j][2].argsort()]
                classes[j,:] = z
    
            tri = isi_data
            _, datax = self._preprocess_data(tri) 
            data_reshaped = datax[:,:-1,:]
            data_reshaped = data_reshaped / self.training_mol_num
            data_reshaped = data_reshaped.reshape(data_reshaped.shape[0],data_reshaped.shape[1],data_reshaped.shape[2],1)
    
            np.save(data_folder + "data_reshaped" + ".npy", data_reshaped)
            np.save(data_folder + "classes_reshaped" + ".npy", classes)             
            
    
    