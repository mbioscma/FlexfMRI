# Author: Marc Biosca on 20.10.24

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def get_data_directory(func_path, mode = 'output_T1seg.txt'):
    directories = []
    for folder in os.listdir(func_path):
        if folder.startswith('sub-'):
            report_folder = os.path.join(func_path, folder, 'ses-01', 'report')
            if os.path.exists(report_folder) and os.path.exists(os.path.join(report_folder, mode)):
                directories.append(os.path.join(report_folder, mode))
    return directories

def get_data(file_path, mode = 'output_T1seg.txt'):
    if mode == 'output_T1seg.txt':
        voxel_counts = {
            'GM': 0,
            'WM': 0,
            'CSF': 0,
            'Total': 0
        }
        with open(file_path, 'r') as file:
            for line in file:
                if 'Sum of GM voxels:' in line:
                    voxel_counts['GM'] = int(re.search(r'\d+', line).group())
                elif 'Sum of WM voxels:' in line:
                    voxel_counts['WM'] = int(re.search(r'\d+', line).group())
                elif 'Sum of CSF voxels:' in line:
                    voxel_counts['CSF'] = int(re.search(r'\d+', line).group())
                elif 'Total number of voxels:' in line:
                    voxel_counts['Total'] = int(re.search(r'\d+', line).group())
        return voxel_counts
    elif mode == 'output_mc.txt':
        mc_params = {
            'Xrot_max': 0, 'Xrot_mean': 0, 'Xrot_std': 0,
            'Yrot_max': 0, 'Yrot_mean': 0, 'Yrot_std': 0,
            'Zrot_max': 0, 'Zrot_mean': 0, 'Zrot_std': 0,
            'Xtrans_max': 0, 'Xtrans_mean': 0, 'Xtrans_std': 0,
            'Ytrans_max': 0, 'Ytrans_mean': 0, 'Ytrans_std': 0,
            'Ztrans_max': 0, 'Ztrans_mean': 0, 'Ztrans_std': 0,
            'FD_max': 0, 'FD_mean': 0, 'FD_std': 0
        }

        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r'MC param (\w+) \(max, mean, std\): ([\d.-]+), ([\d.-]+), ([\d.-]+)', line)
                if match:
                    param, max_val, mean_val, std_val = match.groups()
                    mc_params[f'{param}_max'] = float(max_val)
                    mc_params[f'{param}_mean'] = float(mean_val)
                    mc_params[f'{param}_std'] = float(std_val)
        
        return mc_params
    else:
        return None
    
def integrate_dataframe(func_path):
    # First get the voxel counts
    directories = get_data_directory(func_path, mode = 'output_T1seg.txt')
    data = {}
    for directory in directories:
        #Excract the subject id
        subject_id = directory.split('/')[-4]
        data[subject_id] = get_data(directory, mode = 'output_T1seg.txt')
    df_voxels = pd.DataFrame(data).T
    df_voxels.columns = ['GM_voxels', 'WM_voxels', 'CSF_voxels', 'Total_voxels']
    
    # Now get the motion correction parameters
    directories = get_data_directory(func_path, mode = 'output_mc.txt')
    data = {}
    for directory in directories:
        #Excract the subject id
        subject_id = directory.split('/')[-4]
        data[subject_id] = get_data(directory, mode = 'output_mc.txt')
    df_mc = pd.DataFrame(data).T
    df_mc.columns = ['Xrot_max', 'Xrot_mean', 'Xrot_std',
            'Yrot_max', 'Yrot_mean', 'Yrot_std',
            'Zrot_max', 'Zrot_mean', 'Zrot_std',
            'Xtrans_max', 'Xtrans_mean', 'Xtrans_std',
            'Ytrans_max', 'Ytrans_mean', 'Ytrans_std',
            'Ztrans_max', 'Ztrans_mean', 'Ztrans_std',
            'FD_max', 'FD_mean', 'FD_std']

    # Merge the two dataframes
    df = pd.merge(df_voxels, df_mc, left_index=True, right_index=True, how='outer')

    # Make the subject_id a column
    df = df.reset_index()
    df = df.reset_index().rename(columns={'index': 'subject_id'})
    df.drop('level_0', axis=1, inplace=True)
    
    return df


def main():
    func_path = '/pool/home/AD_Multimodal/Estudio_A4/A4_BIDS/fmri_slicetiming_outputderivatives/rs_preproc'
    df = integrate_dataframe(func_path)

    df['GM_voxels_perc'] = df['GM_voxels'] / df['Total_voxels']
    df['WM_voxels_perc'] = df['WM_voxels'] / df['Total_voxels']
    df['CSF_voxels_perc'] = df['CSF_voxels'] / df['Total_voxels']

    # Save the dataframe as a csv
    file_name = 'motion_voxel_data.csv'
    df.to_csv(os.path.join(func_path, file_name), index=False)
    df.describe().to_csv(os.path.join(func_path, 'motion_voxel_data_summary.csv'))


if __name__ == '__main__':
    main()

    
        
