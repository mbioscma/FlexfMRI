# Author: Marc Biosca on 20.10.2014

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import shapiro
import json

# Load the data
func_path = '/pool/home/AD_Multimodal/Estudio_A4/A4_BIDS/fmri_slicetiming_outputderivatives/rs_preproc'
df = pd.read_csv(os.path.join(func_path, 'motion_voxel_data.csv'))

rejections_df = pd.DataFrame(columns=['subject_id', 'FD_Pass', 'FD_max_Pass', 'GM_lower_Pass', 'GM_upper_Pass', 'WM_lower_Pass', 'WM_upper_Pass', 'CSF_lower_Pass', 'CSF_upper_Pass', 'Total_lower_Pass', 'Total_upper_Pass', 'Pass'])

times = 3
alpha = 0.05

mean_FD = df['FD_mean'].tolist()

# Perform the Shapiro-Wilk test
stat, p = shapiro(mean_FD)

print('Statistics=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#Graph the frequency of the mean FD
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(mean_FD, bins=30, color='blue', kde=True)
plt.title('Mean FD distribution')
plt.xlabel('Mean FD')
plt.ylabel('Frequency')
plt.show()
plt.close()

threshold_FD = np.mean(mean_FD) + times*np.std(mean_FD)
print(f'The threshold for the FD is {round(threshold_FD,3)}')
count = 0

for subject in df['subject_id'].unique():
    subject_df = df[df['subject_id'] == subject]
    if np.any(subject_df['FD_mean'] > threshold_FD):
        count += 1
        rejections_df = pd.concat([rejections_df, pd.DataFrame([{'subject_id': subject, 'Pass': False}])], ignore_index=True)
    else:
        rejections_df = pd.concat([rejections_df, pd.DataFrame([{'subject_id': subject, 'Pass': True}])], ignore_index=True)

print(count)
percentage = round(count/len(df['subject_id'].unique())*100, 2)
print(f'{percentage}% of the subjects have a FD > {threshold_FD}')


FD_max = df['FD_max'].tolist()

# Perform the Shapiro-Wilk test
stat, p = shapiro(FD_max)

print('Statistics=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#Graph the frequency of the max FD
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(FD_max, bins=30, color='blue', kde=True)
plt.title('Max FD distribution')
plt.xlabel('Max FD')
plt.ylabel('Frequency')
plt.show()
plt.close()

threshold_FD_max = np.mean(FD_max) + times*np.std(FD_max)

print(f'The threshold for the FD_max is {round(threshold_FD_max,3)}')

count = 0

for subject in df['subject_id'].unique():
    subject_df = df[df['subject_id'] == subject]
    if np.any(subject_df['FD_max'] > threshold_FD_max):
        count += 1
        rejections_df.loc[rejections_df['subject_id'] == subject, 'FD_max_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'FD_max_Pass'] = True

print(count)
percentage = round(count/len(df['subject_id'].unique())*100, 2)
print(f'{percentage}% of the subjects have a FD_max > {threshold_FD_max}')

GM_perc = df['GM_voxels_perc'].tolist()

# Perform the Shapiro-Wilk test
stat, p = shapiro(GM_perc)

print('Statistics=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#Graph the frequency of the mean FD
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(GM_perc, bins=30, color='blue', kde=True)
plt.title('GM percentage distribution')
plt.xlabel('GM percentage')
plt.ylabel('Frequency')
plt.show()
plt.close()

threshold_GM_lower = np.mean(GM_perc) - times*np.std(GM_perc)
threshold_GM_upper = np.mean(GM_perc) + times*np.std(GM_perc)
print(f'The threshold for the GM percentage is {round(threshold_GM_lower,3)} and {round(threshold_GM_upper,3)}')
count = 0

for subject in df['subject_id'].unique():
    if np.any((subject_df['GM_voxels_perc'] < threshold_GM_lower) | (subject_df['GM_voxels_perc'] > threshold_GM_upper)):
        count += 1

print(count)
percentage = round(count/len(df['subject_id'].unique())*100, 2)
print(f'{percentage}% of the subjects have a GM percentage out of the threshold')

# Frequency distribution of the WM percentage
WM_perc = df['WM_voxels_perc'].tolist()

# Perform the Shapiro-Wilk test
stat, p = shapiro(WM_perc)

print('Statistics=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#Graph the frequency
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(WM_perc, bins=30, color='blue', kde=True)
plt.title('WM percentage distribution')
plt.xlabel('WM percentage')
plt.ylabel('Frequency')
plt.show()
plt.close()

threshold_WM_lower = np.mean(WM_perc) - times*np.std(WM_perc)
threshold_WM_upper = np.mean(WM_perc) + times*np.std(WM_perc)
print(f'The threshold for the WM percentage is {round(threshold_WM_lower,3)} and {round(threshold_WM_upper,3)}')
count = 0

for subject in df['subject_id'].unique():
    if np.any((subject_df['WM_voxels_perc'] < threshold_WM_lower) | (subject_df['WM_voxels_perc'] > threshold_WM_upper)):
        count += 1

print(count)
percentage = round(count/len(df['subject_id'].unique())*100, 2)
print(f'{percentage}% of the subjects have a WM percentage out of the threshold')

CSF_perc = df['CSF_voxels_perc'].tolist()

# Perform the Shapiro-Wilk test
stat, p = shapiro(CSF_perc)

print('Statistics=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#Graph the frequency
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(CSF_perc, bins=30, color='blue', kde=True)
plt.title('CSF percentage distribution')
plt.xlabel('CSF percentage')
plt.ylabel('Frequency')
plt.show()
plt.close()

threshold_CSF_lower = np.mean(CSF_perc) - times*np.std(CSF_perc)
threshold_CSF_upper = np.mean(CSF_perc) + times*np.std(CSF_perc)
print(f'The threshold for the CSF percentage is {round(threshold_CSF_lower,3)} and {round(threshold_CSF_upper,3)}')
count = 0

for subject in df['subject_id'].unique():
    if np.any((subject_df['CSF_voxels_perc'] < threshold_CSF_lower) | (subject_df['CSF_voxels_perc'] > threshold_CSF_upper)):
        count += 1

print(count)
percentage = round(count/len(df['subject_id'].unique())*100, 2)
print(f'{percentage}% of the subjects have a CSF percentage out of the threshold')

total_voxels = df['Total_voxels'].tolist()

# Perform the Shapiro-Wilk test
stat, p = shapiro(total_voxels)

print('Statistics=%.3f, p=%.3f' % (stat, p))

if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

#Graph the frequency
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(total_voxels, bins=30, color='blue', kde=True)
plt.title('Total voxels distribution')
plt.xlabel('Total voxels')
plt.ylabel('Frequency')
plt.show()
plt.close()

threshold_total_lower = np.mean(total_voxels) - times*np.std(total_voxels)
threshold_total_upper = np.mean(total_voxels) + times*np.std(total_voxels)
print(f'The threshold for the Total voxels is {round(threshold_total_lower)} and {round(threshold_total_upper)}')
count = 0

for subject in df['subject_id'].unique():
    if np.any((subject_df['Total_voxels'] < threshold_total_lower) | (subject_df['Total_voxels'] > threshold_total_upper)):
        count += 1

print(count)
percentage = round(count/len(df['subject_id'].unique())*100, 2)
print(f'{percentage}% of the subjects have a Total voxels out of the threshold')

for subject in df['subject_id'].unique():
    subject_df = df[df['subject_id'] == subject]
    if np.any(subject_df['FD_mean'] > threshold_FD):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'FD_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'FD_Pass'] = True

    if np.any(subject_df['GM_voxels_perc'] < threshold_GM_lower):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'GM_lower_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'GM_lower_Pass'] = True

    if np.any(subject_df['GM_voxels_perc'] > threshold_GM_upper):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'GM_upper_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'GM_upper_Pass'] = True
    
    if np.any(subject_df['WM_voxels_perc'] < threshold_WM_lower):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'WM_lower_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'WM_lower_Pass'] = True
    
    if np.any(subject_df['WM_voxels_perc'] > threshold_WM_upper):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'WM_upper_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'WM_upper_Pass'] = True

    if np.any(subject_df['CSF_voxels_perc'] < threshold_CSF_lower):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'CSF_lower_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'CSF_lower_Pass'] = True
    
    if np.any(subject_df['CSF_voxels_perc'] > threshold_CSF_upper):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'CSF_upper_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'CSF_upper_Pass'] = True
    
    if np.any(subject_df['Total_voxels'] < threshold_total_lower):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'Total_lower_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'Total_lower_Pass'] = True

    if np.any(subject_df['Total_voxels'] > threshold_total_upper):
        rejections_df.loc[rejections_df['subject_id'] == subject, 'Total_upper_Pass'] = False
    else:
        rejections_df.loc[rejections_df['subject_id'] == subject, 'Total_upper_Pass'] = True

rejections_df['Pass'] = rejections_df['FD_Pass'] & rejections_df['GM_lower_Pass'] & rejections_df['GM_upper_Pass'] & rejections_df['WM_lower_Pass'] & rejections_df['WM_upper_Pass'] & rejections_df['CSF_lower_Pass'] & rejections_df['CSF_upper_Pass'] & rejections_df['Total_lower_Pass'] & rejections_df['Total_upper_Pass'] & rejections_df['FD_max_Pass'] 

general_count = 0

for subject in df['subject_id'].unique():
    if np.any(rejections_df[rejections_df['subject_id'] == subject]['Pass'] == False):
        general_count += 1

percentage_general = round(general_count/len(df['subject_id'].unique())*100, 2)
print(f'The total number of subjects that are rejected is {general_count} ({percentage_general}%)')

# Save the rejected subjects in a csv file
save = True

if save:
    rejections_df[rejections_df['Pass'] == False].to_csv(os.path.join(func_path, 'automatic_rejections.csv'), index=False)

    # Save a json file with the thresholds formatted in a proper way

thresholds = {'FD': round(threshold_FD, 3),
              'FD_max': round(threshold_FD_max, 3),
              'GM_lower': round(threshold_GM_lower, 3),
              'GM_upper': round(threshold_GM_upper, 3),
              'WM_lower': round(threshold_WM_lower, 3),
              'WM_upper': round(threshold_WM_upper, 3),
              'CSF_lower': round(threshold_CSF_lower, 3),
              'CSF_upper': round(threshold_CSF_upper, 3),
              'Total_lower': round(threshold_total_lower),
              'Total_upper': round(threshold_total_upper)}

save_threshold = True

if save_threshold:
    with open(os.path.join(func_path, 'automatic_thresholds.json'), 'w') as f:
        json.dump(thresholds, f)
