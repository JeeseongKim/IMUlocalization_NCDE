import torch
from torch import nn
import random
import math
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms
import torchvision
import cv2
cv2.ocl.setUseOpenCL(False)
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from scipy.signal import butter, filtfilt

class my_dataset_gyroz_M9_test(Dataset):
    def __init__(self, window_size, imu_filename, vicon_filename):
        super(my_dataset_gyroz_M9_test, self).__init__()
        self.imu_t = []
        self.angle_z = []
        self.imu_x = []
        self.imu_y = []
        self.ofs_x = []
        self.ofs_y = []
        self.gyro_z = []

        self.vicon_t = []
        self.vicon_x = []
        self.vicon_y = []
        self.vicon_angle_z = []

        self.imu = []
        self.vicon= []

        #self.dataset_filename = []
        #self.dataset_filename_vicon = []

        self.window_size = window_size

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9/train/imu/*.txt'))):
        filename = imu_filename
        self.dataset_imu = []
        with open(filename) as f:
            self.dataset_imu.append(f.readlines())

            #if(len(self.dataset_imu)<5):
            self.dataset_imu = (self.dataset_imu[0])

            for i in range(len(self.dataset_imu)):
                self.imu_t.append(self.dataset_imu[i].split('\t')[0]) #time
                self.angle_z.append(self.dataset_imu[i].split('\t')[7]) #angle
                self.imu_x.append(self.dataset_imu[i].split('\t')[4]) #acceleration(x or y)
                self.imu_y.append(self.dataset_imu[i].split('\t')[5]) #acceleration(x or y)
                self.ofs_x.append(self.dataset_imu[i].split('\t')[13]) #ofs x -> vel
                self.ofs_y.append(self.dataset_imu[i].split('\t')[14].split('\n')[0]) #ofs y -> vel
                self.gyro_z.append(self.dataset_imu[i].split('\t')[10])

        #np_imu_t = np.array(self.imu_t, dtype=np.float32)
        #np_angle_z = np.array(self.angle_z, dtype=np.float32)
        #np_imu_x = np.array(self.imu_x, dtype=np.float32)
        #np_imu_y = np.array(self.imu_y, dtype=np.float32)
        #np_ofs_x = np.array(self.ofs_x, dtype=np.float32)
        #np_ofs_y = np.array(self.ofs_y, dtype=np.float32)
        #np_gyro_z = np.array(self.gyro_z, dtype=np.float32)

        a_np_imu_t = np.array(self.imu_t, dtype=np.float32)
        a_np_angle_z = (np.array(self.angle_z, dtype=np.float32))
        a_np_imu_x = self.butter_lowpass_filter_imu(np.array(self.imu_x, dtype=np.float32))
        a_np_imu_y = self.butter_lowpass_filter_imu(np.array(self.imu_y, dtype=np.float32))
        a_np_ofs_x = np.array(self.ofs_x, dtype=np.float32)
        a_np_ofs_y = np.array(self.ofs_y, dtype=np.float32)
        a_np_gyro_z = self.butter_lowpass_filter_gyro(np.array(self.gyro_z, dtype=np.float32))

        tensor_imu_t = torch.from_numpy(a_np_imu_t) * 0.01 #time (sec)
        tensor_angle_z = torch.from_numpy(a_np_angle_z) * 0.1 #angle z (degree)
        tensor_imu_x = torch.from_numpy(a_np_imu_x.copy()) * 1e-6 #acc_x (m/s2)
        tensor_imu_y = torch.from_numpy(a_np_imu_y.copy()) * 1e-6 #acc_y (m/s2)
        tensor_ofs_x = torch.from_numpy(a_np_ofs_x) / 20000 #ofs_x (m/s)
        tensor_ofs_y = torch.from_numpy(a_np_ofs_y) / 20000 #ofs_y (m/s)
        tensor_gyro_z = torch.from_numpy(a_np_gyro_z.copy()) * 0.1 #gyro_z (degree)

        tensor_imu_t = tensor_imu_t.view(tensor_imu_t.shape[0], 1)
        tensor_angle_z = tensor_angle_z.view(tensor_angle_z.shape[0], 1)
        tensor_imu_x = tensor_imu_x.view(tensor_imu_x.shape[0], 1)
        tensor_imu_y = tensor_imu_y.view(tensor_imu_y.shape[0], 1)
        tensor_ofs_x = tensor_ofs_x.view(tensor_ofs_x.shape[0], 1)
        tensor_ofs_y = tensor_ofs_y.view(tensor_ofs_y.shape[0], 1)
        tensor_gyro_z = tensor_gyro_z.view(tensor_gyro_z.shape[0], 1)

        self.imu = torch.cat([tensor_imu_t, tensor_angle_z, tensor_imu_x, tensor_imu_y, tensor_gyro_z, tensor_ofs_x, tensor_ofs_y], dim=1)

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9/train/vicon/*.txt'))):
        filename = vicon_filename
        self.dataset_vicon = []
        with open(filename) as f:
            self.dataset_vicon.append(f.readlines())

            self.dataset_vicon = (self.dataset_vicon[0])

            for i in range(len(self.dataset_vicon)):
                self.vicon_t.append(self.dataset_vicon[i].split('\t')[0])
                self.vicon_x.append(self.dataset_vicon[i].split('\t')[5])
                self.vicon_y.append(self.dataset_vicon[i].split('\t')[6])
                self.vicon_angle_z.append(self.dataset_vicon[i].split('\t')[4])

        np_vicon_t = np.array(self.vicon_t, dtype=np.float32)
        np_vicon_x = np.array(self.vicon_x, dtype=np.float32)
        np_vicon_y = np.array(self.vicon_y, dtype=np.float32)
        np_vicon_angle_z = np.array(self.vicon_angle_z, dtype=np.float32)

        tensor_vicon_t = torch.from_numpy(np_vicon_t) * 0.001  # time (sec)
        tensor_vicon_x = torch.from_numpy(np_vicon_x) * 0.001   #position (m)
        tensor_vicon_y = torch.from_numpy(np_vicon_y) * 0.001   #position (m)
        tensor_vicon_angle_z = torch.from_numpy(np_vicon_angle_z) * 180/math.pi #angle_z (degree)

        tensor_vicon_t = tensor_vicon_t.view(tensor_vicon_t.shape[0], 1)
        tensor_vicon_x = tensor_vicon_x.view(tensor_vicon_x.shape[0], 1)
        tensor_vicon_y = tensor_vicon_y.view(tensor_vicon_y.shape[0], 1)
        tensor_vicon_angle_z = tensor_vicon_angle_z.view(tensor_vicon_angle_z.shape[0], 1)

        new_tensor_vicon_x = tensor_vicon_y
        new_tensor_vicon_y = -1 * tensor_vicon_x

        self.vicon = torch.cat([tensor_vicon_t, new_tensor_vicon_x, new_tensor_vicon_y, tensor_vicon_angle_z], dim=1)

        self.processing_imu = []
        for idx in range(self.imu.shape[0]):
            if(len(self.imu)-idx < window_size):
                self.processing_imu.append(self.imu[len(self.imu)-window_size:len(self.imu), :])
                break
            self.processing_imu.append(self.imu[idx:idx+window_size, :])

        self.processing_target = []
        for idx_trg in range(self.vicon.shape[0]):
            if(len(self.vicon)-idx_trg < window_size):
                #self.processing_target.append(self.vicon[idx_trg:len(self.vicon), :])
                self.processing_target.append(self.vicon[len(self.vicon)-window_size:len(self.vicon), :])
                break
            self.processing_target.append(self.vicon[idx_trg:idx_trg+window_size, :])

        self.len = len(self.processing_imu)

    def butter_lowpass_filter_imu(self, data):

        T = 0.01 # period(time interval)
        fs = 1000/10 #rate(total number of samples/period)
        cutoff = 0.15  # cutoff frequency

        nyq = 0.5 * fs
        order = 2  # filter order
        n = int(T * fs)  # total number of samples

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def butter_lowpass_filter_gyro(self, data):

        T = 0.01 # period(time interval)
        fs = 1000/10 #rate(total number of samples/period)
        cutoff = 35  # cutoff frequency

        nyq = 0.5 * fs
        order = 3  # filter order
        n = int(T * fs)  # total number of samples

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def __getitem__(self, index):
        return self.processing_imu[index], self.processing_target[index]

    def __len__(self):
        return len(self.processing_imu)
class my_dataset_gyroz_M9_test(Dataset):
    def __init__(self, window_size, imu_filename, vicon_filename):
        super(my_dataset_gyroz_M9_test, self).__init__()
        self.imu_t = []
        self.angle_z = []
        self.imu_x = []
        self.imu_y = []
        self.ofs_x = []
        self.ofs_y = []
        self.gyro_z = []

        self.vicon_t = []
        self.vicon_x = []
        self.vicon_y = []
        self.vicon_angle_z = []

        self.imu = []
        self.vicon= []

        #self.dataset_filename = []
        #self.dataset_filename_vicon = []

        self.window_size = window_size

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9/train/imu/*.txt'))):
        filename = imu_filename
        self.dataset_imu = []
        with open(filename) as f:
            self.dataset_imu.append(f.readlines())

            #if(len(self.dataset_imu)<5):
            self.dataset_imu = (self.dataset_imu[0])

            for i in range(len(self.dataset_imu)):
                self.imu_t.append(self.dataset_imu[i].split('\t')[0]) #time
                self.angle_z.append(self.dataset_imu[i].split('\t')[7]) #angle
                self.imu_x.append(self.dataset_imu[i].split('\t')[4]) #acceleration(x or y)
                self.imu_y.append(self.dataset_imu[i].split('\t')[5]) #acceleration(x or y)
                self.ofs_x.append(self.dataset_imu[i].split('\t')[13]) #ofs x -> vel
                self.ofs_y.append(self.dataset_imu[i].split('\t')[14].split('\n')[0]) #ofs y -> vel
                self.gyro_z.append(self.dataset_imu[i].split('\t')[10])

        #np_imu_t = np.array(self.imu_t, dtype=np.float32)
        #np_angle_z = np.array(self.angle_z, dtype=np.float32)
        #np_imu_x = np.array(self.imu_x, dtype=np.float32)
        #np_imu_y = np.array(self.imu_y, dtype=np.float32)
        #np_ofs_x = np.array(self.ofs_x, dtype=np.float32)
        #np_ofs_y = np.array(self.ofs_y, dtype=np.float32)
        #np_gyro_z = np.array(self.gyro_z, dtype=np.float32)

        a_np_imu_t = np.array(self.imu_t, dtype=np.float32)
        a_np_angle_z = (np.array(self.angle_z, dtype=np.float32))
        a_np_imu_x = self.butter_lowpass_filter_imu(np.array(self.imu_x, dtype=np.float32))
        a_np_imu_y = self.butter_lowpass_filter_imu(np.array(self.imu_y, dtype=np.float32))
        a_np_ofs_x = np.array(self.ofs_x, dtype=np.float32)
        a_np_ofs_y = np.array(self.ofs_y, dtype=np.float32)
        a_np_gyro_z = self.butter_lowpass_filter_gyro(np.array(self.gyro_z, dtype=np.float32))

        tensor_imu_t = torch.from_numpy(a_np_imu_t) * 0.01 #time (sec)
        tensor_angle_z = torch.from_numpy(a_np_angle_z) * 0.1 #angle z (degree)
        tensor_imu_x = torch.from_numpy(a_np_imu_x.copy()) * 1e-6 #acc_x (m/s2)
        tensor_imu_y = torch.from_numpy(a_np_imu_y.copy()) * 1e-6 #acc_y (m/s2)
        tensor_ofs_x = torch.from_numpy(a_np_ofs_x) / 20000 #ofs_x (m/s)
        tensor_ofs_y = torch.from_numpy(a_np_ofs_y) / 20000 #ofs_y (m/s)
        tensor_gyro_z = torch.from_numpy(a_np_gyro_z.copy()) * 0.1 #gyro_z (degree)

        tensor_imu_t = tensor_imu_t.view(tensor_imu_t.shape[0], 1)
        tensor_angle_z = tensor_angle_z.view(tensor_angle_z.shape[0], 1)
        tensor_imu_x = tensor_imu_x.view(tensor_imu_x.shape[0], 1)
        tensor_imu_y = tensor_imu_y.view(tensor_imu_y.shape[0], 1)
        tensor_ofs_x = tensor_ofs_x.view(tensor_ofs_x.shape[0], 1)
        tensor_ofs_y = tensor_ofs_y.view(tensor_ofs_y.shape[0], 1)
        tensor_gyro_z = tensor_gyro_z.view(tensor_gyro_z.shape[0], 1)

        self.imu = torch.cat([tensor_imu_t, tensor_angle_z, tensor_imu_x, tensor_imu_y, tensor_gyro_z, tensor_ofs_x, tensor_ofs_y], dim=1)

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9/train/vicon/*.txt'))):
        filename = vicon_filename
        self.dataset_vicon = []
        with open(filename) as f:
            self.dataset_vicon.append(f.readlines())

            self.dataset_vicon = (self.dataset_vicon[0])

            for i in range(len(self.dataset_vicon)):
                self.vicon_t.append(self.dataset_vicon[i].split('\t')[0])
                self.vicon_x.append(self.dataset_vicon[i].split('\t')[5])
                self.vicon_y.append(self.dataset_vicon[i].split('\t')[6])
                self.vicon_angle_z.append(self.dataset_vicon[i].split('\t')[4])

        np_vicon_t = np.array(self.vicon_t, dtype=np.float32)
        np_vicon_x = np.array(self.vicon_x, dtype=np.float32)
        np_vicon_y = np.array(self.vicon_y, dtype=np.float32)
        np_vicon_angle_z = np.array(self.vicon_angle_z, dtype=np.float32)

        tensor_vicon_t = torch.from_numpy(np_vicon_t) * 0.001  # time (sec)
        tensor_vicon_x = torch.from_numpy(np_vicon_x) * 0.001   #position (m)
        tensor_vicon_y = torch.from_numpy(np_vicon_y) * 0.001   #position (m)
        tensor_vicon_angle_z = torch.from_numpy(np_vicon_angle_z) * 180/math.pi #angle_z (degree)

        tensor_vicon_t = tensor_vicon_t.view(tensor_vicon_t.shape[0], 1)
        tensor_vicon_x = tensor_vicon_x.view(tensor_vicon_x.shape[0], 1)
        tensor_vicon_y = tensor_vicon_y.view(tensor_vicon_y.shape[0], 1)
        tensor_vicon_angle_z = tensor_vicon_angle_z.view(tensor_vicon_angle_z.shape[0], 1)

        new_tensor_vicon_x = tensor_vicon_y
        new_tensor_vicon_y = -1 * tensor_vicon_x

        self.vicon = torch.cat([tensor_vicon_t, new_tensor_vicon_x, new_tensor_vicon_y, tensor_vicon_angle_z], dim=1)

        self.processing_imu = []
        for idx in range(self.imu.shape[0]):
            if(len(self.imu)-idx < window_size):
                self.processing_imu.append(self.imu[len(self.imu)-window_size:len(self.imu), :])
                break
            self.processing_imu.append(self.imu[idx:idx+window_size, :])

        self.processing_target = []
        for idx_trg in range(self.vicon.shape[0]):
            if(len(self.vicon)-idx_trg < window_size):
                #self.processing_target.append(self.vicon[idx_trg:len(self.vicon), :])
                self.processing_target.append(self.vicon[len(self.vicon)-window_size:len(self.vicon), :])
                break
            self.processing_target.append(self.vicon[idx_trg:idx_trg+window_size, :])

        self.len = len(self.processing_imu)

    def butter_lowpass_filter_imu(self, data):

        T = 0.01 # period(time interval)
        fs = 1000/10 #rate(total number of samples/period)
        cutoff = 0.15  # cutoff frequency

        nyq = 0.5 * fs
        order = 2  # filter order
        n = int(T * fs)  # total number of samples

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def butter_lowpass_filter_gyro(self, data):

        T = 0.01 # period(time interval)
        fs = 1000/10 #rate(total number of samples/period)
        cutoff = 35  # cutoff frequency

        nyq = 0.5 * fs
        order = 3  # filter order
        n = int(T * fs)  # total number of samples

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def __getitem__(self, index):
        return self.processing_imu[index], self.processing_target[index]

    def __len__(self):
        return len(self.processing_imu)

class my_dataset_gyroz_M9(Dataset):
    def __init__(self, window_size):
        super(my_dataset_gyroz_M9, self).__init__()
        self.imu_t = []
        self.angle_z = []
        self.imu_x = []
        self.imu_y = []
        self.ofs_x = []
        self.ofs_y = []
        self.gyro_z = []

        self.vicon_t = []
        self.vicon_x = []
        self.vicon_y = []
        self.vicon_angle_z = []

        self.imu = []
        self.vicon= []

        #self.dataset_filename = []
        #self.dataset_filename_vicon = []

        self.window_size = window_size

        for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9/train/imu/*.txt'))):
        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9_tmp/train/imu/*.txt'))):
            self.dataset_imu = []
            with open(filename) as f:
                self.dataset_imu.append(f.readlines())

            #if(len(self.dataset_imu)<5):
            self.dataset_imu = (self.dataset_imu[0])

            for i in range(len(self.dataset_imu)):
                self.imu_t.append(self.dataset_imu[i].split('\t')[0]) #time
                self.angle_z.append(self.dataset_imu[i].split('\t')[7]) #angle
                self.imu_x.append(self.dataset_imu[i].split('\t')[4]) #acceleration(x or y)
                self.imu_y.append(self.dataset_imu[i].split('\t')[5]) #acceleration(x or y)
                self.ofs_x.append(self.dataset_imu[i].split('\t')[13]) #ofs x -> vel
                self.ofs_y.append(self.dataset_imu[i].split('\t')[14].split('\n')[0]) #ofs y -> vel
                self.gyro_z.append(self.dataset_imu[i].split('\t')[10])

        a_np_imu_t = np.array(self.imu_t, dtype=np.float32)
        a_np_angle_z = (np.array(self.angle_z, dtype=np.float32))
        a_np_imu_x = self.butter_lowpass_filter_imu(np.array(self.imu_x, dtype=np.float32))
        a_np_imu_y = self.butter_lowpass_filter_imu(np.array(self.imu_y, dtype=np.float32))
        a_np_ofs_x = np.array(self.ofs_x, dtype=np.float32)
        a_np_ofs_y = np.array(self.ofs_y, dtype=np.float32)
        a_np_gyro_z = self.butter_lowpass_filter_gyro(np.array(self.gyro_z, dtype=np.float32))

        tensor_imu_t = torch.from_numpy(a_np_imu_t) * 0.01 #time (sec)
        tensor_angle_z = torch.from_numpy(a_np_angle_z) * 0.1 #angle z (degree)
        tensor_imu_x = torch.from_numpy(a_np_imu_x.copy()) * 1e-6 #acc_x (m/s2)
        tensor_imu_y = torch.from_numpy(a_np_imu_y.copy()) * 1e-6 #acc_y (m/s2)
        tensor_ofs_x = torch.from_numpy(a_np_ofs_x) / 20000 #ofs_x (m/s)
        tensor_ofs_y = torch.from_numpy(a_np_ofs_y) / 20000 #ofs_y (m/s)
        tensor_gyro_z = torch.from_numpy(a_np_gyro_z.copy()) * 0.1 #gyro_z (degree)

        tensor_imu_t = tensor_imu_t.view(tensor_imu_t.shape[0], 1)
        tensor_angle_z = tensor_angle_z.view(tensor_angle_z.shape[0], 1)
        tensor_imu_x = tensor_imu_x.view(tensor_imu_x.shape[0], 1)
        tensor_imu_y = tensor_imu_y.view(tensor_imu_y.shape[0], 1)
        tensor_ofs_x = tensor_ofs_x.view(tensor_ofs_x.shape[0], 1)
        tensor_ofs_y = tensor_ofs_y.view(tensor_ofs_y.shape[0], 1)
        tensor_gyro_z = tensor_gyro_z.view(tensor_gyro_z.shape[0], 1)
        '''
        for idx in range(tensor_imu_t.shape[0]):
            if (torch.abs(tensor_angle_z[idx, 0]) >= torch.tensor(180.0)):
                if (tensor_angle_z[idx, 0] > 0):
                    tensor_angle_z[idx, 0] = tensor_angle_z[idx, 0] - (2 * torch.tensor(180.0))
                elif (tensor_angle_z[idx, 0] < 0):
                    tensor_angle_z[idx, 0] = tensor_angle_z[idx, 0] + (2 * torch.tensor(180.0))

            if (torch.abs(tensor_gyro_z[idx, 0]) > torch.tensor(180.0)):
                if (tensor_gyro_z[idx, 0] > 0):
                    tensor_gyro_z[idx, 0] = tensor_gyro_z[idx, 0] - (2 * torch.tensor(180.0))
                elif (tensor_gyro_z[idx, 0] < 0):
                    tensor_gyro_z[idx, 0] = tensor_gyro_z[idx, 0] + (2 * torch.tensor(180.0))
        '''
        self.imu = torch.cat([tensor_imu_t, tensor_angle_z, tensor_imu_x, tensor_imu_y, tensor_gyro_z, tensor_ofs_x, tensor_ofs_y], dim=1)

        for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9/train/vicon/*.txt'))):
        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/M9_tmp/train/vicon/*.txt'))):
            self.dataset_vicon = []
            with open(filename) as f:
                self.dataset_vicon.append(f.readlines())

            self.dataset_vicon = (self.dataset_vicon[0])

            for i in range(len(self.dataset_vicon)):
                self.vicon_t.append(self.dataset_vicon[i].split('\t')[0])
                self.vicon_x.append(self.dataset_vicon[i].split('\t')[5])
                self.vicon_y.append(self.dataset_vicon[i].split('\t')[6])
                self.vicon_angle_z.append(self.dataset_vicon[i].split('\t')[4])

        np_vicon_t = np.array(self.vicon_t, dtype=np.float32)
        np_vicon_x = np.array(self.vicon_x, dtype=np.float32)
        np_vicon_y = np.array(self.vicon_y, dtype=np.float32)
        np_vicon_angle_z = np.array(self.vicon_angle_z, dtype=np.float32)

        tensor_vicon_t = torch.from_numpy(np_vicon_t) * 0.001  # time (sec)
        tensor_vicon_x = torch.from_numpy(np_vicon_x) * 0.001   #position (m)
        tensor_vicon_y = torch.from_numpy(np_vicon_y) * 0.001   #position (m)
        tensor_vicon_angle_z = torch.from_numpy(np_vicon_angle_z) * 180/math.pi #angle_z (degree)

        tensor_vicon_t = tensor_vicon_t.view(tensor_vicon_t.shape[0], 1)
        tensor_vicon_x = tensor_vicon_x.view(tensor_vicon_x.shape[0], 1)
        tensor_vicon_y = tensor_vicon_y.view(tensor_vicon_y.shape[0], 1)
        tensor_vicon_angle_z = tensor_vicon_angle_z.view(tensor_vicon_angle_z.shape[0], 1)
        '''
        for idx in range(tensor_vicon_angle_z.shape[0]):
            if (torch.abs(tensor_vicon_angle_z[idx, 0]) > torch.tensor(180.0)):
                if (tensor_vicon_angle_z[idx, 0] > 0):
                    tensor_vicon_angle_z[idx, 0] = tensor_vicon_angle_z[idx, 0] - (2 * torch.tensor(180.0))
                elif (tensor_vicon_angle_z[idx, 0] < 0):
                    tensor_vicon_angle_z[idx, 0] = tensor_vicon_angle_z[idx, 0] + (2 * torch.tensor(180.0))
        '''
        new_tensor_vicon_x = tensor_vicon_y
        new_tensor_vicon_y = -1 * tensor_vicon_x

        self.vicon = torch.cat([tensor_vicon_t, new_tensor_vicon_x, new_tensor_vicon_y, tensor_vicon_angle_z], dim=1)

        self.processing_imu = []
        for idx in range(self.imu.shape[0]):
            if(len(self.imu)-idx < window_size):
                self.processing_imu.append(self.imu[len(self.imu)-window_size:len(self.imu), :])
                break
            self.processing_imu.append(self.imu[idx:idx+window_size, :])

        self.processing_target = []
        for idx_trg in range(self.vicon.shape[0]):
            if(len(self.vicon)-idx_trg < window_size):
                #self.processing_target.append(self.vicon[idx_trg:len(self.vicon), :])
                self.processing_target.append(self.vicon[len(self.vicon)-window_size:len(self.vicon), :])
                break
            self.processing_target.append(self.vicon[idx_trg:idx_trg+window_size, :])

        self.len = len(self.processing_imu)

    def butter_lowpass_filter_imu(self, data):

        T = 0.01 # period(time interval)
        fs = 1000/10 #rate(total number of samples/period)
        cutoff = 0.15  # cutoff frequency

        nyq = 0.5 * fs
        order = 2  # filter order
        n = int(T * fs)  # total number of samples

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def butter_lowpass_filter_gyro(self, data):

        T = 0.01 # period(time interval)
        fs = 1000/10 #rate(total number of samples/period)
        cutoff = 35  # cutoff frequency

        nyq = 0.5 * fs
        order = 3  # filter order
        n = int(T * fs)  # total number of samples

        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def __getitem__(self, index):
        return self.processing_imu[index], self.processing_target[index]

    def __len__(self):
        return len(self.processing_imu)

class loss_cosim(nn.Module):
    def __init__(self):
        super(loss_cosim, self).__init__()

    def forward(self, DetectionMap, tf_DetectionMap):

        cosim = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        my_val = (cosim(DetectionMap, tf_DetectionMap)).mean()
        #cosim_loss = torch.exp(-1.0 * my_val)
        cosim_loss = 1 - my_val

        return cosim_loss

