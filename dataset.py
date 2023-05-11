import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
from torch.utils.data import Dataset


def filter_signal(raw_ecg_signal):
    if np.all(raw_ecg_signal == 0): # no need to do anything for 0 signals
        return raw_ecg_signal
    
    a, b = butter(5, .05, 'highpass')
    filt_ecg_signal = filtfilt(a, b, raw_ecg_signal)
    norm_ecg_signal = (filt_ecg_signal - np.mean(filt_ecg_signal)) / np.std(filt_ecg_signal)
    
    return norm_ecg_signal


class ECGDataset(Dataset):
    def __init__(self, csv_path, label_column):
        super(ECGDataset, self).__init__()
        data_df = pd.read_csv(csv_path)
        self.file_list = [Path(sig_path) for sig_path in data_df['signal_file']]
        self.dx_labels = list(data_df[label_column])
        
    def __len__(self):
        return len(self.file_list)
    
    
    def __getitem__(self, i):       
        # Load the given signals
        raw_ecg_signals = loadmat(str(self.file_list[i]))['val']
        
        # Filter them
        norm_ecg_signals = np.array([filter_signal(s) for s in raw_ecg_signals])
        
        
        return norm_ecg_signals, self.dx_labels[i]

