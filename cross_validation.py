import os
import numpy as np
import pandas as pd
import scipy.io as sio
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import shutil

# Load Data
CD_Loc = "path_to_GraceFall_folder"  # Update this with actual path

# Load .mat files
All34_table_loc = os.path.join(CD_Loc, 'Raw_Data', 'All34_table.mat')
Label_table_loc = os.path.join(CD_Loc, 'Raw_Data', 'Label_table.mat')

data_all34 = sio.loadmat(All34_table_loc)
data_label = sio.loadmat(Label_table_loc)

# Load EEG files
where_my_edfs = os.path.join(CD_Loc, 'Raw_Data', 'Filtered', '*.edf')
edfs_names = glob.glob(where_my_edfs)

# Parameters
Batch_Num = 13
channel = "R6"
window_size_list = [256]
sys_order_list = [3]
validation_loops = 1
subject_validation_number = 12
delay_time = 80
overlap = 1  # Overlap percentage
train_nofall = True
train_acc = True

# Create Results Directory
folder_loc = os.path.join(CD_Loc, 'Results_Classifiers', f'Cross_validation_{Batch_Num}')
os.makedirs(folder_loc, exist_ok=True)
validation_filename = os.path.join(folder_loc, 'Validation_Subjects.txt')

# Cross-validation setup
validation_title = ['Validation_Loop'] + [f'V{i}' for i in range(1, subject_validation_number + 1)]
pd.DataFrame([validation_title]).to_csv(validation_filename, index=False, header=False)

# Cross-validation loop
for i in range(1, validation_loops + 1):
    # Randomly exclude subjects for validation
    exclude_numbers = np.random.choice(range(2, 42), subject_validation_number, replace=False)
    subjects_list_train = list(set(range(2, 42)) - set(exclude_numbers))
    edfs_name_local = [edfs_names[idx - 1] for idx in subjects_list_train]

    # Save the validation information
    pd.DataFrame([[i] + list(exclude_numbers)]).to_csv(validation_filename, mode='a', index=False, header=False)

    # Training placeholder
    print(f"Training on subjects: {subjects_list_train}")


    # Placeholder for classifier training and evaluation
    def train_classifier(subjects_list_train, edfs_name_local):
        print("Training model...")
        # Implement the training logic here
        return None


    model = train_classifier(subjects_list_train, edfs_name_local)


    # Placeholder for evaluation
    def evaluate_model(model, validation_data):
        print("Evaluating model...")
        # Implement evaluation logic here
        return np.random.rand(), np.random.rand(), np.random.rand()


    accuracy, precision, recall = evaluate_model(model, subjects_list_train)
    print(f"Validation {i}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}")

# Visualization placeholder
plt.figure()
plt.title("Cross-validation Results")
plt.xlabel("Validation Loop")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
