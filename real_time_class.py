import numpy as np
import os
import pandas as pd
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.signal import savgol_filter

class RealTimeClass:
    
    @staticmethod
    def real_time_steps(eeg_sig, acc_sig, label_table_subject, event_table_subject, eeg_cap,
                         sys_order, window_size, stepsize, thresholds, trained_classifier):
        unexpected_locations = np.where(label_table_subject == 1)[0]
        unexpected_events = event_table_subject[unexpected_locations]
        
        start_point, end_point = 0, len(eeg_sig)
        num_iterations = (end_point - start_point - window_size) // stepsize
        
        P1 = np.zeros((num_iterations, 3))
        P2 = np.zeros((num_iterations, 4))
        
        current_point = start_point
        k = 0
        while current_point + window_size < end_point:
            eeg_epoch = eeg_sig[current_point:current_point + window_size]
            acc_epoch = acc_sig[:, current_point:current_point + window_size]
            
            if np.all(eeg_epoch == eeg_epoch[0]) or np.all(acc_epoch == acc_epoch[:, 0]):
                current_point += stepsize
                continue
            
            eeg_epoch_norm = Preprocessing.normalize_eeg(eeg_epoch, eeg_cap)
            acc_epoch_norm = Preprocessing.normalize_acc(acc_epoch)
            
            if np.all(eeg_epoch_norm < thresholds[0]) and np.all(acc_epoch_norm < thresholds[1]):
                current_point += stepsize
                continue
            
            eeg_feature = Preprocessing.feature_extraction(eeg_epoch_norm, sys_order)
            acc_feature = Preprocessing.feature_extraction(acc_epoch_norm, sys_order)
            features = np.hstack((eeg_feature, acc_feature))
            
            p1, p2 = trained_classifier.predict(features.reshape(1, -1))
            
            result_value = int(np.any((current_point >= (unexpected_events - 100)) & 
                                      (current_point <= (unexpected_events + 280))))
            
            P2[k] = [current_point, *p2[:2], result_value]
            P1[k] = [current_point, 1 if p1[0] != 'Expected' else 0, result_value]
            
            if p1[0] != 'Expected':
                current_point += 2000
            
            k += 1
            current_point += stepsize
        
        return P1[:k], P2[:k]
    
    @staticmethod
    def load_trainer(folder_loc, pathi):
        trained_classifier_loc = os.path.join(folder_loc, pathi)
        trained_classifier = joblib.load(trained_classifier_loc)
        return trained_classifier
    
    @staticmethod
    def get_result_table(results, window_size_list, sys_order_list, subjects_list):
        all_values = []
        for window_size in window_size_list:
            for sys_order in sys_order_list:
                for subject in subjects_list:
                    P1 = results[(window_size, sys_order, subject)]["P1"]
                    values = RealTimeClass.get_result_matrix(P1, window_size, sys_order, subject)
                    all_values.append(values)
        
        results_table = pd.DataFrame(all_values, columns=["window", "sys_order", "subject", "TP", "TN", "FP", "FN"])
        return results_table
    
    @staticmethod
    def get_result_matrix(P1, window_size, sys_order, subject):
        values = np.zeros((P1.shape[0], 7))
        values[:, 0] = window_size
        values[:, 1] = sys_order
        values[:, 2] = subject
        
        for i in range(P1.shape[0]):
            event_result = P1[i, 2]
            if np.any(P1[:, 1] == 1):
                if event_result == 1:
                    values[i, 3] = 1
                else:
                    values[i, 5] = 1
            else:
                if event_result == 1:
                    values[i, 6] = 1
                else:
                    values[i, 4] = 1
        
        return values
    
    @staticmethod
    def get_sum_result_table(results_table, window_size_list, sys_order_list, subjects_validation_list):
        all_summaries = []
        for window_size in window_size_list:
            for sys_order in sys_order_list:
                for subject in subjects_validation_list:
                    subset = results_table[(results_table["window"] == window_size) &
                                           (results_table["sys_order"] == sys_order) &
                                           (results_table["subject"] == subject)]
                    TP, TN, FP, FN = subset[["TP", "TN", "FP", "FN"]].sum()
                    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
                    specificity = TN / (TN + FP) if TN + FP > 0 else 0
                    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
                    
                    all_summaries.append([window_size, sys_order, subject, TP, TN, FP, FN, sensitivity, specificity, accuracy])
        
        summary_table = pd.DataFrame(all_summaries, 
                                     columns=["window", "sys_order", "subject", "TP", "TN", "FP", "FN", "Sensitivity", "Specificity", "Accuracy"])
        return summary_table
