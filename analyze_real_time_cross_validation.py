import os
import pandas as pd
import numpy as np

# Define variables
Batch_Num = 13
train_acc = True
CD_Loc = r"C:\Users\medha\PycharmProjects\PythonProject"

# Load validation file
batch_folder_loc = os.path.join(CD_Loc, 'Results_Classifires', f'Cross_validation_{Batch_Num}')
validation_filename = os.path.join(batch_folder_loc, 'Validation_Subjects.txt')
validation_table = pd.read_csv(validation_filename, sep=r"\s+", engine="python")

num_subjects = validation_table.shape[1] - 1

folder_loc_validation = os.path.join(CD_Loc, 'Results_Classifires', f'Cross_validation_{Batch_Num}', f'V{1}')
validation_loop_filename = os.path.join(folder_loc_validation, f'Batch_{Batch_Num}_V{1}.txt')
Info = pd.read_csv(validation_loop_filename, sep=r"\s+", engine="python")

continuous_filename = os.path.join(batch_folder_loc, 'Continuous_results.txt')
if not os.path.exists(continuous_filename):
    raise FileNotFoundError("Continuous results file not found.")

cross_validation_info = pd.read_csv(continuous_filename, sep=r"\s+", engine="python")
if 'window' not in cross_validation_info.columns:
    raise KeyError("Column 'window' not found in cross_validation_info. Check file format or delimiter.")

channel = Info.iloc[0]['Channel']
delay_time = Info.iloc[0]['delay_time']
window_size_list = np.unique(cross_validation_info['window'])
sys_order_list = np.unique(cross_validation_info['sys_order'])
stepsize = np.unique(Info['sys_order'])

num_trainers = len(window_size_list) * len(sys_order_list)
num_validations = validation_table.shape[0]

Sensitivity = np.zeros(num_validations + 1)
Specificity = np.zeros(num_validations + 1)
Accuracy = np.zeros(num_validations + 1)
Balanced_Accuracy = np.zeros(num_validations + 1)

for i in range(num_validations):
    if len(cross_validation_info) > 0:
        Sensitivity[i] = cross_validation_info['Sensitivity'].iloc[0]
        Specificity[i] = cross_validation_info['Specificity'].iloc[0]
        Accuracy[i] = cross_validation_info['Accuracy'].iloc[0]
        Balanced_Accuracy[i] = cross_validation_info['Balanced_Accuracy'].iloc[0]
    else:
        print(f"Skipping validation {i+1}: No valid data in cross_validation_info.")

Sensitivity[-1] = np.nanmean(Sensitivity[:-1])
Specificity[-1] = np.nanmean(Specificity[:-1])
Accuracy[-1] = np.nanmean(Accuracy[:-1])
Balanced_Accuracy[-1] = np.nanmean(Balanced_Accuracy[:-1])

Validation_loop = ['1', '2', '3', '4', '5', 'means'][:len(Sensitivity)]
T = pd.DataFrame({
    'Validation_loop': Validation_loop,
    'Sensitivity': Sensitivity,
    'Specificity': Specificity,
    'Accuracy': Accuracy,
    'Balanced_Accuracy': Balanced_Accuracy
})

TP = np.zeros(num_validations + 1)
TN = np.zeros(num_validations + 1)
FP = np.zeros(num_validations + 1)
FN = np.zeros(num_validations + 1)

for i in range(num_validations):
    if len(cross_validation_info) > 0:
        TP[i] = cross_validation_info['TP'].iloc[0]
        TN[i] = cross_validation_info['TN'].iloc[0]
        FP[i] = cross_validation_info['FP'].iloc[0]
        FN[i] = cross_validation_info['FN'].iloc[0]
    else:
        print(f"Skipping validation {i+1}: No valid data in cross_validation_info.")

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

sensitivity_ratio = sum(cross_validation_info['Sensitivity'] == 1) / len(cross_validation_info['Sensitivity'])
print("Sensitivity Ratio:", sensitivity_ratio)
