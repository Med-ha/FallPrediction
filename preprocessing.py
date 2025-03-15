import numpy as np
import os
import datetime
import scipy.signal as signal

class Preprocessing:
    
    @staticmethod
    def create_eeg_acc_windows(subjects_list, edfs_name_local, event_table_local, label_table_local, channel, window_size, delay_time, overlap, train_nofall):
        eeg_preprocessed_windows_length = len(subjects_list) * len(overlap) * len(event_table_local[0][2:]) * 2
        eeg_preprocessed_windows = np.zeros((eeg_preprocessed_windows_length, window_size))
        acc_preprocessed_windows = np.zeros((eeg_preprocessed_windows_length, window_size))
        label_preprocessed_windows = np.zeros(eeg_preprocessed_windows_length)

        insert_idx = 0
        for subject_number in subjects_list:
            print(f'Subject {subject_number} is being preprocessed')
            
            edfs_name, label_table_subject, event_table_subject, eeg_sig, eeg_cap = Preprocessing.get_eeg_parameters(
                subject_number, edfs_name_local, label_table_local, event_table_local, channel)
            acc_sig = Preprocessing.get_acc_signal(edfs_name)
            onset_values, label_values = Preprocessing.get_onset_values(len(eeg_sig), event_table_subject, label_table_subject)
            
            if train_nofall != 'True':
                mask = label_values != 2
                onset_values = onset_values[mask]
                label_values = label_values[mask]
            
            eeg_window, acc_window, label_list = Preprocessing.chop_eeg_acc_to_window(
                eeg_sig, acc_sig, eeg_cap, onset_values, label_values, window_size, delay_time, overlap)
            
            eeg_preprocessed_windows[insert_idx:insert_idx+len(label_list), :] = eeg_window
            acc_preprocessed_windows[insert_idx:insert_idx+len(label_list), :] = acc_window
            label_preprocessed_windows[insert_idx:insert_idx+len(label_list)] = label_list
            insert_idx += len(label_list)
        
        eeg_preprocessed_windows = eeg_preprocessed_windows[:insert_idx, :]
        acc_preprocessed_windows = acc_preprocessed_windows[:insert_idx, :]
        label_preprocessed_windows = label_preprocessed_windows[:insert_idx]
        
        mapped_label_preprocessed_windows = np.array(['Expected' if lbl == 0 else 'Unexpected' if lbl == 1 else 'Expected' for lbl in label_preprocessed_windows])
        return eeg_preprocessed_windows, acc_preprocessed_windows, mapped_label_preprocessed_windows
    
    @staticmethod
    def get_eeg_parameters(subject, edfs_name_local, label_table, event_table, channel):
        edf_index = [i for i, name in enumerate(edfs_name_local) if str(subject).zfill(2) in name][0]
        table_index = np.where(label_table[:, 0] == subject)[0][0]
        
        label_table_subject = label_table[table_index, 2:]
        event_table_subject = event_table[table_index, 2:]
        edfs_name = edfs_name_local[edf_index]
        eeg_sig = Preprocessing.get_edf_signal(edfs_name, channel)
        eeg_cap = Preprocessing.get_eeg_cap(eeg_sig, event_table_subject[0])
        return edfs_name, label_table_subject, event_table_subject, eeg_sig, eeg_cap
    
    @staticmethod
    def get_edf_signal(edfs_name, channel):
        from pyedflib import highlevel
        signals, _, _ = highlevel.read_edf(edfs_name)
        return signals[channel]
    
    @staticmethod
    def get_acc_signal(edfs_name):
        acc_sig_x = Preprocessing.get_edf_signal(edfs_name, 'x_dir') / 980
        acc_sig_y = Preprocessing.get_edf_signal(edfs_name, 'y_dir') / 980
        acc_sig_z = Preprocessing.get_edf_signal(edfs_name, 'z_dir') / 980
        return np.array([acc_sig_x, acc_sig_y, acc_sig_z])
    
    @staticmethod
    def get_eeg_cap(eeg_sig, timing_1):
        per_1 = eeg_sig[timing_1 + 1:timing_1 + 1000]
        per_1b = per_1 - np.mean(per_1[:40])
        return np.max(per_1b)
    
    @staticmethod
    def normalize_eeg(eeg_epoch, eeg_cap):
        eeg_epoch_bn = (eeg_epoch - np.mean(eeg_epoch[:40])) / eeg_cap
        return signal.savgol_filter(eeg_epoch_bn, 21, 3)
    
    @staticmethod
    def normalize_acc(acc_epoch):
        acc_epoch_x = signal.savgol_filter(acc_epoch[0, :], 21, 3)
        acc_epoch_y = signal.savgol_filter(acc_epoch[1, :], 21, 3)
        acc_epoch_z = signal.savgol_filter(acc_epoch[2, :], 21, 3)
        return np.sqrt(acc_epoch_x**2 + acc_epoch_y**2 + acc_epoch_z**2)
    
    @staticmethod
    def get_onset_values(eeg_length, event_table_subject, label_table_subject):
        onset_nofall = np.random.randint(1000, eeg_length - 2000, len(event_table_subject))
        onset_values = np.concatenate((onset_nofall, event_table_subject))
        label_values = np.concatenate((np.full(len(event_table_subject), 2), label_table_subject))
        sorted_indices = np.argsort(onset_values)
        return onset_values[sorted_indices], label_values[sorted_indices]
    
    @staticmethod
    def chop_eeg_acc_to_window(eeg_sig, acc_sig, eeg_cap, onset_values, label_values, window_size, delay_time, overlap):
        num_loops = len(onset_values)
        eeg_window = np.zeros((num_loops * len(overlap), window_size))
        acc_window = np.zeros((num_loops * len(overlap), window_size))
        label_list = np.zeros(num_loops * len(overlap))
        row_index = 0
        
        for i in range(num_loops):
            for j in overlap:
                start_idx = onset_values[i] + delay_time + j
                end_idx = start_idx + window_size
                if end_idx <= len(eeg_sig):
                    eeg_epoch = eeg_sig[start_idx:end_idx]
                    eeg_window[row_index, :] = Preprocessing.normalize_eeg(eeg_epoch, eeg_cap)
                    acc_epoch = acc_sig[:, start_idx:end_idx]
                    acc_window[row_index, :] = Preprocessing.normalize_acc(acc_epoch)
                    label_list[row_index] = label_values[i]
                    row_index += 1
        
        return eeg_window[:row_index, :], acc_window[:row_index, :], label_list[:row_index]
