import os
import numpy as np
import pandas as pd

# Define batch number and file locations
Batch_Num = 13
train_acc = True  # Assuming True is a boolean
# CD_Loc = "path_to_directory"  # Define your base directory
CD_Loc = "C:\Users\medha\PycharmProjects\PythonProject\Results_Classifiers"  # Define your base directory

batch_folder_loc = os.path.join(CD_Loc, 'Results_Classifires', f'Cross_validation_{Batch_Num}')
validation_filename = os.path.join(batch_folder_loc, 'Validation_Subjects.txt')
validation_table = pd.read_csv(validation_filename, delimiter='\t')

folder_loc_validation = os.path.join(CD_Loc, 'Results_Classifires', f'Cross_validation_{Batch_Num}', f'V{1}')
validation_loop_filename = os.path.join(folder_loc_validation, f'Batch_{Batch_Num}_V{1}.txt')
Info = pd.read_csv(validation_loop_filename, delimiter='\t')

continuous_filename = os.path.join(batch_folder_loc, 'Continuous_results.txt')
cross_validation_info = pd.read_csv(continuous_filename, delimiter='\t')
validation_size = validation_table.shape

channel = Info.loc[0, 'Channel']
delay_time = Info.loc[0, 'delay_time']  # ms
num_trainers = len(cross_validation_info['window'].unique()) * len(cross_validation_info['sys_order'].unique())
num_subjects = validation_size[1] - 1  # Example value

window_size_list = cross_validation_info['window'].unique()
sys_order_list = cross_validation_info['sys_order'].unique()
stepsize = Info['sys_order'].unique()

# Initialize arrays
Sensitivity = np.zeros(validation_size[0] + 1)
Specificity = np.zeros(validation_size[0] + 1)
Accuracy = np.zeros(validation_size[0] + 1)
Balanced_Accuracy = np.zeros(validation_size[0] + 1)

# Compute averages per validation
for i in range(validation_size[0]):
    start = 1 + (validation_size[1] - 1) * i
    stop = (validation_size[1] - 1) * (i + 1)
    range_idx = range(start, stop)

    Sensitivity[i] = cross_validation_info.loc[range_idx, 'Sensitivity'].mean()
    Specificity[i] = cross_validation_info.loc[range_idx, 'Specificity'].mean()
    Accuracy[i] = cross_validation_info.loc[range_idx, 'Accuracy'].mean()
    Balanced_Accuracy[i] = cross_validation_info.loc[range_idx, 'Balanced_Accuracy'].mean()

Sensitivity[-1] = Sensitivity[:-1].mean()
Specificity[-1] = Specificity[:-1].mean()
Accuracy[-1] = Accuracy[:-1].mean()
Balanced_Accuracy[-1] = Balanced_Accuracy[:-1].mean()

Validation_loop = ['1', '2', '3', '4', '5', 'means']
T = pd.DataFrame({
    'Validation_loop': Validation_loop,
    'Sensitivity': Sensitivity,
    'Specificity': Specificity,
    'Accuracy': Accuracy,
    'Balanced_Accuracy': Balanced_Accuracy
})

# Initialize arrays for TP, TN, FP, FN
TP = np.zeros(validation_size[0] + 1)
TN = np.zeros(validation_size[0] + 1)
FP = np.zeros(validation_size[0] + 1)
FN = np.zeros(validation_size[0] + 1)

# Compute averages per validation
for i in range(validation_size[0]):
    start = 1 + (validation_size[1] - 1) * i
    stop = (validation_size[1] - 1) * (i + 1)
    range_idx = range(start, stop)

    TP[i] = cross_validation_info.loc[range_idx, 'TP'].mean()
    TN[i] = cross_validation_info.loc[range_idx, 'TN'].mean()
    FP[i] = cross_validation_info.loc[range_idx, 'FP'].mean()
    FN[i] = cross_validation_info.loc[range_idx, 'FN'].mean()

TP[-1] = TP[:-1].mean()
TN[-1] = TN[:-1].mean()
FP[-1] = FP[:-1].mean()
FN[-1] = FN[:-1].mean()

T2 = pd.DataFrame({
    'Validation_loop': Validation_loop,
    'TP': TP,
    'TN': TN,
    'FP': FP,
    'FN': FN
})

# Continuous Real-time Evaluation
non_zero_sensitivity_ratio = (cross_validation_info['Sensitivity'] == 1).sum() / len(
    cross_validation_info['Sensitivity'])

# Display results
import ace_tools as tools

tools.display_dataframe_to_user(name="Validation Metrics", dataframe=T)
tools.display_dataframe_to_user(name="Confusion Matrix Metrics", dataframe=T2)
