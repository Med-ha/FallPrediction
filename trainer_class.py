import numpy as np
import os
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
import joblib

class TrainerClass:
    
    @staticmethod
    def train_xx(subjects_list, edfs_name_local, event_table_local, label_table_local, channel,
                 window_size_list, sys_order_list, delay_time, overlap, train_nofall, train_acc, file_name):
        
        k_ind = 1
        results = []
        
        for window_size in window_size_list:
            if train_acc:
                eeg_preprocessed_windows, acc_preprocessed_windows, label_preprocessed_windows = Preprocessing.create_eeg_acc_windows(
                    subjects_list, edfs_name_local, event_table_local, label_table_local, channel, window_size, delay_time, overlap, train_nofall)
            else:
                eeg_preprocessed_windows, label_preprocessed_windows = Preprocessing.create_eeg_windows(
                    subjects_list, edfs_name_local, event_table_local, label_table_local, channel, window_size, delay_time, overlap, train_nofall)
            
            for sys_order in sys_order_list:
                if train_acc:
                    eeg_feature = Preprocessing.feature_extraction(eeg_preprocessed_windows, sys_order)
                    acc_feature = Preprocessing.feature_extraction(acc_preprocessed_windows, sys_order)
                    features = np.hstack((eeg_feature, acc_feature))
                else:
                    features = Preprocessing.feature_extraction(eeg_preprocessed_windows, sys_order)
                
                predictors, predictor_names = TrainerClass.get_predictors(features)
                trained_classifier = TrainerClass.train_classifier(features, predictor_names, predictors, label_preprocessed_windows)
                results_values = TrainerClass.validation(trained_classifier, label_preprocessed_windows)
                
                params_values = [k_ind, channel, delay_time, window_size, overlap, sys_order]
                results.append(params_values + results_values)
                
                path = os.path.join(os.path.dirname(file_name), f'sys{sys_order}_win{window_size}_ch{channel}.pkl')
                TrainerClass.save_trained_classifier(path, trained_classifier)
                
                General.log_info(file_name, results)
                k_ind += 1
        
        return results
    
    @staticmethod
    def get_predictors(features):
        col_length = features.shape[1]
        predictor_names = [f'column_{i}' for i in range(col_length)]
        predictors = pd.DataFrame(features, columns=predictor_names)
        return predictors, predictor_names
    
    @staticmethod
    def train_classifier(features, predictor_names, predictors, mapped_response):
        classifier = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=30
        )
        classifier.fit(predictors, mapped_response)
        
        return classifier
    
    @staticmethod
    def validation(trained_classifier, eeg_labels):
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        validation_accuracy = np.mean(cross_val_score(trained_classifier, eeg_labels, eeg_labels, cv=kfold))
        
        predictions = trained_classifier.predict(eeg_labels)
        cm = confusion_matrix(eeg_labels, predictions)
        
        if cm.shape == (2, 2):
            true_negative = cm[0, 0] / np.sum(cm[0, :])
            true_positive = cm[1, 1] / np.sum(cm[1, :])
            positive_detective = cm[1, 1] / np.sum(cm[:, 1])
        else:
            true_negative = true_positive = positive_detective = 0.0
        
        return [round(true_positive * 100, 4), round(true_negative * 100, 4), round(positive_detective * 100, 4), round(validation_accuracy * 100, 4)]
    
    @staticmethod
    def save_trained_classifier(path, classifier):
        joblib.dump(classifier, path)
