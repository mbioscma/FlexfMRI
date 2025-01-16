# Author: Marc Biosca on 21.10.2014

# mbioscma7@alumnes.ub.edu


import csv
import re
import os
import subprocess
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import base64
import pandas as pd
import json

csv_file = '/pool/home/AD_Multimodal/Estudio_A4/Scripts Preproc Slicing/subject_list.csv'
path_to_func = '/pool/home/AD_Multimodal/Estudio_A4/A4_BIDS/fmri_slicetiming_outputderivatives/rs_preproc'
path_to_anat = '/pool/home/AD_Multimodal/Estudio_A4/A4_BIDS/fmri_slicetiming_outputderivatives/anat'

problematic_subjects = False

if problematic_subjects:
    problematic_df = pd.read_csv('/mnt/B468D0C568D0878E/usuarios/MarcBiosca/problematic_subjects.csv')
    problematic_sub = problematic_df['subject'].tolist()
else:
    problematic_sub = []

thres = True

if thres:
    #Open the json file with the thresholds
    with open(os.path.join(path_to_func, 'automatic_thresholds.json'), 'r') as f:
        thresholds = json.load(f)
    print('Thresholds loaded')
else:
    thresholds = {"FD": 0.458, "FD_max": 2.599, "GM_lower": 0.171, "GM_upper": 0.312,
                    "WM_lower": 0.341, "WM_upper": 0.463, "CSF_lower": 0.31,
                    "CSF_upper": 0.402, "Total_lower": 719936, "Total_upper": 1320292}


def create_directory(directory_path):
    """ Create a directory if it does not exist """
    try:
        os.makedirs(directory_path)
        print(f'Report directory created: {directory_path}')
    except FileExistsError:
        print(f'Report directory already exists: {directory_path}')

        if os.path.exists(os.path.join(directory_path, 'output_T1seg_Warning.txt')):
            # Delete the warning file if it exists
            os.remove(os.path.join(directory_path, 'output_T1seg_Warning.txt'))
        
        if os.path.exists(os.path.join(directory_path, 'output_mc_Warning.txt')):
            # Delete the warning file if it exists
            os.remove(os.path.join(directory_path, 'output_mc_Warning.txt'))

def load_first_volume(nii_path):
    ''' Load the first volume of a 4D NIfTI file '''
    img = nib.load(nii_path)
    if img.ndim == 4:
        volume_fmri = 10
        return img.slicer[..., volume_fmri].get_fdata(caching='unchanged')  # Select 10th volume if 4D
    return img.get_fdata(caching='unchanged')

def extract_ten_volume_and_save(nii_path, output_path):
    ''' Extract the 10th volume of a 4D NIfTI file and save it to a new file '''
    volume = 10
    img = nib.load(nii_path)
    ten_vol_data = img.slicer[..., volume].get_fdata(caching='unchanged')  # Extract the 10th volume
    nib.save(nib.Nifti1Image(ten_vol_data, img.affine), output_path)

def image_to_data_url(image_path):
    """ Convert an image file to a data URL """
    # Determine the image type based on the file extension
    file_type = image_path.split('.')[-1]
    if file_type.lower() == 'png':
        mime_type = 'image/png'
    elif file_type.lower() == 'gif':
        mime_type = 'image/gif'
    else:
        raise ValueError("Unsupported file type")

    with open(image_path, "rb") as img_file:
        return f"data:{mime_type};base64,{base64.b64encode(img_file.read()).decode('utf-8')}"


def get_slices(subject, report_dir):

    ''' Get slices from the different images and save them as PNG files '''

    # try changing the subprocess for the python fsl version
    # anat segmentation 
    nii_in = path_to_anat + '/' + subject + '/ses-01/' + 'segmentation/' + subject +'_ses-01_T1w_fast_seg_pveseg.nii.gz'
    if not os.path.exists(nii_in):
        nii_in = path_to_anat + '/' + subject + '/ses-01/' + 'segmentation/' + subject +'_ses-01_run-02_T1w_fast_seg_pveseg.nii.gz'
    png_out= report_dir + '/T1_segmentation.png'
    slicer_run = subprocess.run(['slicer', nii_in, 
                                '-l render2',
                                 '-a', png_out])

    # registration anat to template 
    nii_in = path_to_anat + '/' + subject + '/ses-01/' + subject +'_ses-01_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'
    if not os.path.exists(nii_in):
        nii_in = path_to_anat + '/' + subject + '/ses-01/' + subject +'_ses-01_run-02_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'
    png_out= report_dir + '/T1_2template.png'
    slicer_run = subprocess.run(['slicer', nii_in, 
                                 '-a', png_out])
    
    # functional BET mask 
    nii_in = path_to_func + '/' + subject + '/ses-01/' + subject +'_ses-01_task-rest_bold_desc-reference-rs_seg-brain-bet_mask.nii.gz'
    if not os.path.exists(nii_in):
        nii_in = path_to_func + '/' + subject + '/ses-01/' + subject +'_ses-01_run-02_task-rest_bold_desc-reference-rs_seg-brain-bet_mask.nii.gz'
    png_out= report_dir + '/func_bet_mask.png'
    slicer_run = subprocess.run(['slicer', nii_in, 
                                 '-a', png_out])

    # functional data registered to T1
    nii_in = path_to_func + '/' + subject + '/ses-01/' + subject +'_ses-01_task-rest_bold_desc-rs-ref-reg-2-t1.nii.gz'
    if not os.path.exists(nii_in):
        nii_in = path_to_func + '/' + subject + '/ses-01/' + subject +'_ses-01_run-02_task-rest_bold_desc-rs-ref-reg-2-t1.nii.gz'
    png_out= report_dir + '/func_2T1.png'
    slicer_run = subprocess.run(['slicer', nii_in, 
                                 '-a', png_out])
    
    # anatomical data regsitered to template GIF
    create_gif_registration(subject, report_dir, path_to_anat, num_slices=15, axis=2)

    create_gif_registration2(subject, report_dir, path_to_anat, num_slices=15, axis=2)

    # functional data registered to anatomical data GIF
    create_gif_segmentation(subject, report_dir, num_slices=15)

    # final cleaned data registered to template
    
    #nii_vol0 = path_to_func + '/' + subject + '/' + subject +'_task-rest_bold_desc-ts-mc-cleaned-final-reg-template_vol0.nii.gz'
    #roi_run = subprocess.run(['fslroi', nii_in, nii_vol0, '0 1'])
    
    nii_in = path_to_func + '/' + subject + '/ses-01/' + subject +'_ses-01_task-rest_bold_desc-ts-mc-cleaned-final-reg-template.nii.gz'
    if not os.path.exists(nii_in):
        nii_in = path_to_func + '/' + subject + '/ses-01/' + subject +'_ses-01_run-02_task-rest_bold_desc-ts-mc-cleaned-final-reg-template.nii.gz'
    png_out= report_dir + '/func_final_cleaned_2template.png'
    temp_nii_path = os.path.join(report_dir, 'temp_first_vol.nii.gz')
    extract_ten_volume_and_save(nii_in, temp_nii_path)
    slicer_run = subprocess.run(['slicer', temp_nii_path, '-a', png_out])
    os.remove(temp_nii_path)


def QCmetrics(subject, report_dir, path_to_anat):

    ''' Calculate the motion correction metrics and segmentation metrics and generate txt files. 
        Check if the values are within the thresholds and generate warning files if necessary. 
        Lastly, plot the motion correction summary figure.'''

    # Load the segmentation file
    nii_in = os.path.join(path_to_anat, subject, 'ses-01', 'segmentation', f'{subject}_ses-01_T1w_fast_seg_seg.nii.gz')

    if not os.path.exists(nii_in):
        nii_in = os.path.join(path_to_anat,subject, 'ses-01', 'segmentation', f'{subject}_ses-01_run-02_T1w_fast_seg_seg.nii.gz')

    data = load_first_volume(nii_in)
    
    # Calculate number of voxels for each segmentation
    num_vox_seg0 = np.sum(data == 1)  # GM voxels
    num_vox_seg1 = np.sum(data == 2)  # WM voxels
    num_vox_seg2 = np.sum(data == 3)  # CSF voxels
    total = num_vox_seg0 + num_vox_seg1 + num_vox_seg2
    
    # Create plain text report
    output_file_path = os.path.join(report_dir, 'output_T1seg.txt')
    with open(output_file_path, 'w') as output_file:
        output_file.write(f"Sum of GM voxels: {num_vox_seg0}\n")
        output_file.write(f"Sum of WM voxels: {num_vox_seg1}\n")
        output_file.write(f"Sum of CSF voxels: {num_vox_seg2}\n")
        output_file.write(f"Total number of voxels: {total}\n")

    if (num_vox_seg0/total < thresholds['GM_lower'] or num_vox_seg1/total < thresholds['WM_lower'] or
        num_vox_seg2/total < thresholds['CSF_lower'] or total < thresholds['Total_lower'] or
        num_vox_seg0/total > thresholds['GM_upper'] or num_vox_seg1/total > thresholds['WM_upper'] or
        num_vox_seg2/total > thresholds['CSF_upper'] or total > thresholds['Total_upper']):

        output_file_path_w = os.path.join(report_dir, 'output_T1seg_Warning.txt')
        # Append warnings to the report if any metric is out of range
        with open(output_file_path_w, 'w') as output_file:
            if num_vox_seg0/total < thresholds['GM_lower']:
                output_file.write("WARNING: GM VOLUME IS LESS THAN " + str(thresholds['GM_lower']*100) + "% OF TOTAL VOLUME!!!\n")
            if num_vox_seg1/total < thresholds['WM_lower']:
                output_file.write("WARNING: WM VOLUME IS LESS THAN " + str(thresholds['WM_lower']*100) + "% OF TOTAL VOLUME!!!\n")
            if num_vox_seg2/total < thresholds['CSF_lower']:
                output_file.write("WARNING: CSF VOLUME IS LESS THAN " + str(thresholds['CSF_lower']*100) + "% OF TOTAL VOLUME!!!\n")
            if total < thresholds['Total_lower']:
                output_file.write("WARNING: TOTAL VOLUME IS LESS THAN " + str(thresholds['Total_lower']) + " VOXELS!!!\n")
            if num_vox_seg0/total > thresholds['GM_upper']:
                output_file.write("WARNING: GM VOLUME IS MORE THAN " + str(thresholds['GM_upper']*100) + "% OF TOTAL VOLUME!!!\n")
            if num_vox_seg1/total > thresholds['WM_upper']:
                output_file.write("WARNING: WM VOLUME IS MORE THAN " + str(thresholds['WM_upper']*100) + "% OF TOTAL VOLUME!!!\n")
            if num_vox_seg2/total > thresholds['CSF_upper']:
                output_file.write("WARNING: CSF VOLUME IS MORE THAN " + str(thresholds['CSF_upper']*100) + "% OF TOTAL VOLUME!!!\n")
            if total > thresholds['Total_upper']:
                output_file.write("WARNING: TOTAL VOLUME IS MORE THAN " + str(thresholds['Total_upper']) + " VOXELS!!!\n")

    mc_names = ['Xrot', 'Yrot', 'Zrot', 'Xtrans', 'Ytrans', 'Ztrans', 'FD']
    y_names = ['rad', 'rad', 'rad', 'mm', 'mm', 'mm', 'mm']
    warnings = []
    
    # Summary values of motion correction
    mc_in = os.path.join(path_to_func, subject, 'ses-01', f'{subject}_ses-01_task-rest_bold_desc-mc.nii.gz.par')
    mc_data = np.loadtxt(mc_in)
    max_all = np.max(mc_data, axis=0)
    mean_all = np.mean(mc_data, axis=0)
    std_all = np.std(mc_data, axis=0)

    # ------------- FD CALCULATION -------------
    translations = mc_data[:, 3:6]  # Xtrans, Ytrans, Ztrans
    rotations = mc_data[:, 0:3]     # Xrot, Yrot, Zrot

    head_radius = 50.0
    rotations_mm = rotations * head_radius # Convert rotations from radians to millimeters on the surface of a sphere

    fd = np.zeros(translations.shape[0])
    for i in range(1, translations.shape[0]):
        # Compute differences in translations and rotations
        trans_diff = np.abs(translations[i] - translations[i-1])
        rot_diff = np.abs(rotations_mm[i] - rotations_mm[i-1])
        fd[i] = np.sum(trans_diff) + np.sum(rot_diff)


    mean_fd = np.mean(fd)
    max_fd = np.max(fd)
    std_fd = np.std(fd)

    # Append FD values to the motion correction summary
    mc_data = np.column_stack((mc_data, fd))
    max_all = np.append(max_all, max_fd)
    mean_all = np.append(mean_all, mean_fd)
    std_all = np.append(std_all, std_fd)

    # Round the values
    max_all = np.round(max_all, 4)
    mean_all = np.round(mean_all, 4)
    std_all = np.round(std_all, 4)

    output_file_path = os.path.join(report_dir, 'output_mc.txt')
    with open(output_file_path, 'w') as output_file:
        for i in range(len(mc_names)):
            output_file.write(f"MC param {mc_names[i]} (max, mean, std): {max_all[i]}, {mean_all[i]}, {std_all[i]} {y_names[i]}\n")
            # Check if the parameter exceeds the threshold
        if mean_fd > thresholds['FD']:
            warnings.append(f"WARNING: FD exceeds threshold with mean value {mean_fd} mm\n")
        if max_fd > thresholds['FD_max']:
            warnings.append(f"WARNING: FD exceeds threshold with max value {max_fd} mm\n")

    # Write warnings to a separate warning file if any warnings exist
    if warnings:
        warning_file_path = os.path.join(report_dir, 'output_mc_Warning.txt')
        with open(warning_file_path, 'w') as warning_file:
            for warning in warnings:
                warning_file.write(warning)

    # Plot the motion correction summary figure
    plot_motion_correction(mc_data, mc_names, y_names, report_dir, mean_all)

def plot_motion_correction(mc_data, mc_names, y_names, report_dir, mean_all):

    ''' Plot the motion correction summary figure '''

    num_tp, num_cols = mc_data.shape
    tp = np.arange(num_tp)
    plt.figure(figsize=(10, 15))
    for i in range(len(mc_names)):
        plt.subplot(len(mc_names), 1, i+1)
        plt.plot(tp, mc_data[:, i])
        plt.title(f'{mc_names[i]}, Mean = {mean_all[i]}', fontsize=10)
        plt.xlabel('Timepoints', fontsize=6)
        plt.ylabel(y_names[i], fontsize=6)
    plt.tight_layout()
    output_fig_path = os.path.join(report_dir, 'output_mc_summary.png')
    plt.savefig(output_fig_path)


def generate_html(subject, report_dir, path_to_func):

    ''' Integrate all the images and metrics into an HTML file '''
    
    # Convert image paths to data URLs
    anat_segmentation = image_to_data_url(os.path.join(report_dir, 'T1_segmentation.png'))
    anat_to_template = image_to_data_url(os.path.join(report_dir, 'T1_2template.png'))
    func_bet_mask = image_to_data_url(os.path.join(report_dir, 'func_bet_mask.png'))
    func_to_T1 = image_to_data_url(os.path.join(report_dir, 'func_2T1.png'))
    func_final_cleaned_to_template = image_to_data_url(os.path.join(report_dir, 'func_final_cleaned_2template.png'))
    gif_path_anat = image_to_data_url(os.path.join(report_dir, 'T1_template_slices.gif'))
    gif_path_anat2 = image_to_data_url(os.path.join(report_dir, 'T1_template_slices_overlay.gif'))
    gif_path_seg = image_to_data_url(os.path.join(report_dir, 'T1_segmented_slices.gif'))
    mc_summary = image_to_data_url(os.path.join(report_dir, 'output_mc_summary.png'))


    seg_warning_path = os.path.join(report_dir, 'output_T1seg_Warning.txt')
    mc_warning_path = os.path.join(report_dir, 'output_mc_Warning.txt')

    t1seg_metrics_warning = ''
    mc_metrics_warning = ''

    if os.path.exists(seg_warning_path):
        with open(seg_warning_path, 'r') as file:
            t1seg_metrics_warning = file.read()

    if os.path.exists(mc_warning_path):
        with open(mc_warning_path, 'r') as file:
            mc_metrics_warning = file.read()


    seg_names = ['GM', 'WM', 'CSF', 'Total']
    seg_voxels = []

    pattern = re.compile(r'Sum of GM voxels: (\d+)|Sum of WM voxels: (\d+)|Sum of CSF voxels: (\d+)|Total number of voxels: (\d+)')

    with open(os.path.join(report_dir, 'output_T1seg.txt'), 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract the numbers from the matched groups, ignoring None values
                numbers = [int(num) for num in match.groups() if num is not None]
                seg_voxels.extend(numbers)



    mc_names = ['Xrot', 'Yrot', 'Zrot', 'Xtrans', 'Ytrans', 'Ztrans', 'FD']
    y_names = ['rad', 'rad', 'rad', 'mm', 'mm', 'mm', 'mm']

    

    # Initialize empty lists to store the values
    max_all = []
    mean_all = []
    std_all = []

    # Regular expression to match the values in the file
    pattern = re.compile(r'MC param \w+ \(max, mean, std\): ([\d\.\-e]+), ([\d\.\-e]+), ([\d\.\-e]+)')

    # Read the file and extract the values
    with open(os.path.join(report_dir, 'output_mc.txt'), 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                max_val, mean_val, std_val = match.groups()
                max_all.append(float(max_val))
                mean_all.append(float(mean_val))
                std_all.append(float(std_val))

    # HTML content construction
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Report for Subject {subject}</title>
        <style>
            body {{ font-family: 'Calibri', sans-serif; }}
            img {{ width: 450px; height: auto; }} /* Adjusted width to be 1.5 times bigger */
            .metrics-text {{ background: #f4f4f4; padding: 10px; border-radius: 5px; }}
            .warning-text {{ color: red; font-size: 24px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Processing Report for Subject {subject}</h1>
        <h2>Anatomical Processing</h2>
        <h3>T1-weighted Image to MNI Template</h3>
        <img src="{anat_to_template}" alt="T1 to MNI Template">
        <h3>Comparison of T1 slices in GIF format</h3>
        <img src="{gif_path_anat2}" alt="GIF of T1 Slices" style="width: 250px;"> <!-- Adjusted width to be 1.5 times bigger -->
        <h3>Segmentation of T1-weighted Image</h3>
        <img src="{anat_segmentation}" alt="T1 Segmentation">
        <h3>Comparison of Segmented T1 slices in GIF format</h3>
        <img src="{gif_path_seg}" alt="GIF of Segmented T1 Slices" style="width: 675px;"> <!-- Adjusted width to be 1.5 times bigger -->
        <h2>Functional Processing</h2>
        <h3>Functional BET Mask</h3>
        <img src="{func_bet_mask}" alt="Functional BET Mask">
        <h3>Functional Data Registered to T1-weighted Image</h3>
        <img src="{func_to_T1}" alt="Functional Data to T1">
        <h3>Cleaned Functional Data to Template</h3>
        <img src="{func_final_cleaned_to_template}" alt="Cleaned Functional Data to Template">
        <h2>Motion Correction Summary</h2>
        <img src="{mc_summary}" alt="Motion Correction Summary">
        <h2>Motion Correction Metrics</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Unit</th>
            </tr>
            {"".join([f"<tr><td>{mc_names[i]}</td><td>{max_all[i]}</td><td>{mean_all[i]}</td><td>{std_all[i]}</td><td>{y_names[i]}</td></tr>" for i in range(len(mc_names)-1)])}
            <tr><th>{mc_names[-1]}</th><th>{max_all[-1]}</th><th>{mean_all[-1]}</th><th>{std_all[-1]}</th><th>{y_names[-1]}</th></tr>
        </table>
        {'<h2 class="warning-text">Motion Correction Issues Detected!</h2>' if mc_metrics_warning else ''}
        <pre class="warning-text">{mc_metrics_warning}</pre>
        <h2>T1 Segmentation Metrics</h2>
        <table>
            <tr>
                <th>Region</th>
                <th>Number of Voxels</th>
            </tr>
            {"".join([f"<tr><td>{seg_names[i]}</td><td>{seg_voxels[i]}</td><tr>" for i in range(len(seg_names))])}
        </table>
        {'<h2 class="warning-text">T1 Segmentation Issues Detected!</h2>' if t1seg_metrics_warning else ''}
        <pre class="warning-text">{t1seg_metrics_warning}</pre>
    </body>
    </html>
    """


    # Writing the HTML file
    html_file_path = os.path.join(report_dir, f'report_{subject}.html')
    # If the file already exists, overwrite it
    with open(html_file_path, 'w') as html_file:
        html_file.write(html_content)



def create_gif_registration(subject, report_dir, path_to_anat, num_slices=10, axis=2):

    ''' Create a GIF showing the registration of the T1-weighted image to the MNI template. The two images are shown side by side. '''

    mni_path = '/pool/home/AD_Multimodal/Estudio_A4/MNI152_T1_3.3mm_brain.nii.gz'
    t1_path = path_to_anat + '/' + subject + '/ses-01/' + subject + '_ses-01_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'

    if not os.path.exists(t1_path):
        t1_path = path_to_anat + '/' + subject + '/ses-01/' + subject + '_ses-01_run-02_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'

    # Get the data
    mni_data = load_first_volume(mni_path)
    t1_data = load_first_volume(t1_path)

    # Preparing the GIF generation
    slice_indices = np.linspace(0, mni_data.shape[axis]-1, num_slices, dtype=int)
    frames = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for slice_index in slice_indices:
        if axis == 0:  # sagittal
            mni_slice = mni_data[slice_index, :, :]
            t1_slice = t1_data[slice_index, :, :]
        elif axis == 1:  # coronal
            mni_slice = mni_data[:, slice_index, :]
            t1_slice = t1_data[:, slice_index, :]
        else:  # axial
            mni_slice = mni_data[:, :, slice_index]
            t1_slice = t1_data[:, :, slice_index]

        axes[0].clear()
        axes[1].clear()
        axes[0].imshow(mni_slice.T, cmap='gray', origin='lower')
        axes[0].set_title('MNI')
        axes[0].axis('off')
        axes[1].imshow(t1_slice.T, cmap='gray', origin='lower')
        axes[1].set_title('T1')
        axes[1].axis('off')
        fig.suptitle(f'Slice {slice_index}', fontsize=35)
        fig.canvas.draw()

        # Save the current figure state to a PIL image
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
        frames.append(image.convert('RGB'))

    plt.close(fig)

    # Save the images as a gif file
    gif_path = report_dir + '/T1_template_slices.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)

def create_colored_overlay(mni_slice, t1_slice, alpha=0.5):

    ''' Create a colored overlay image with the MNI template in red and the T1-weighted image in green '''

    # Normalize the slices to range [0, 1]
    mni_slice_norm = mni_slice / mni_slice.max() if mni_slice.max() != 0 else mni_slice
    t1_slice_norm = t1_slice / t1_slice.max() if t1_slice.max() != 0 else t1_slice

    # Create RGB images: MNI in red and T1 in green
    mni_rgb = np.zeros((mni_slice.shape[0], mni_slice.shape[1], 3), dtype=np.float32)
    t1_rgb = np.zeros((t1_slice.shape[0], t1_slice.shape[1], 3), dtype=np.float32)

    mni_rgb[..., 0] = mni_slice_norm  # Red channel
    t1_rgb[..., 1] = t1_slice_norm    # Green channel

    # Blend the images
    blended = (1 - alpha) * mni_rgb + alpha * t1_rgb
    blended = (blended * 255).astype(np.uint8)

    return blended

def create_gif_registration2(subject, report_dir, path_to_anat, num_slices=10, axis=2):

    ''' Create a GIF showing the registration of the T1-weighted image to the MNI template. The two images are shown one on top of the other. '''

    mni_path = '/pool/home/AD_Multimodal/Estudio_A4/MNI152_T1_3.3mm_brain.nii.gz'
    t1_path = path_to_anat + '/' + subject + '/ses-01/' + subject + '_ses-01_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'

    if not os.path.exists(t1_path):
        t1_path = path_to_anat + '/' + subject + '/ses-01/' + subject + '_ses-01_run-02_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'

    # Get the data
    mni_data = load_first_volume(mni_path)
    t1_data = load_first_volume(t1_path)

    # Preparing the GIF generation
    slice_indices = np.linspace(0, mni_data.shape[axis]-1, num_slices, dtype=int)
    frames = []

    for slice_index in slice_indices:
        if axis == 0:  # sagittal
            mni_slice = mni_data[slice_index, :, :]
            t1_slice = t1_data[slice_index, :, :]
        elif axis == 1:  # coronal
            mni_slice = mni_data[:, slice_index, :]
            t1_slice = t1_data[:, slice_index, :]
        else:  # axial
            mni_slice = mni_data[:, :, slice_index]
            t1_slice = t1_data[:, :, slice_index]

        # Create the blended image with transparencies
        blended_image = create_colored_overlay(mni_slice.T, t1_slice.T, alpha=0.5)

        # Rotate the image 180 degrees to match the orientation of the MNI template
        blended_image = np.rot90(blended_image, k=2)

        # Increase the size of the output image
        pil_image = Image.fromarray(blended_image).resize((blended_image.shape[1] * 6, blended_image.shape[0] * 6), Image.NEAREST)
        frames.append(pil_image)

    # Save the images as a gif file
    gif_path = report_dir + '/T1_template_slices_overlay.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)

def create_custom_cmap():

    ''' Create a custom colormap with 4 levels for the segmentation: black, orange, green, and vivid green '''

    # Define a colormap with 4 levels: black, orange, green, and vivid green
    vivid_green = '#00FF00'
    colors = ["black", "orange", 'green', vivid_green]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=4)
    return cmap

def create_gif_segmentation(subject, report_dir, num_slices=15):

    ''' Create a GIF showing the segmented T1-weighted image in the axial, sagittal, and coronal planes '''

    path_to_anat = '/pool/home/AD_Multimodal/Estudio_A4/A4_BIDS/fmri_slicetiming_outputderivatives/anat'
    seg_path = f'{path_to_anat}/{subject}/ses-01/segmentation/{subject}_ses-01_T1w_fast_seg_pveseg.nii.gz'

    if not os.path.exists(seg_path):
        seg_path = f'{path_to_anat}/{subject}/ses-01/segmentation/{subject}_ses-01_run-02_T1w_fast_seg_pveseg.nii.gz'

    seg_data = load_first_volume(seg_path)

    # Create colormap
    cmap = create_custom_cmap()

    # Create a list to store images for the GIF
    images = []

    # Determine starting slice dynamically based on the shape of the segmented image
    start_slice = min(min(seg_data.shape), 20)

    # Iterate over slices of each plane
    slices = np.linspace(start_slice, min(seg_data.shape), num_slices, endpoint=False).astype(int)

    for slice_index in slices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot segmented image for each plane
        axial_img = np.rot90(seg_data[:, :, slice_index])
        sagittal_img = np.rot90(seg_data[:, slice_index, :])
        coronal_img = np.rot90(seg_data[slice_index, :, :])

        axes[0].imshow(coronal_img, cmap=cmap, vmin=0, vmax=3, aspect='auto')
        axes[0].set_title('Coronal plane')
        axes[0].axis('off')

        axes[1].imshow(sagittal_img, cmap=cmap, vmin=0, vmax=3, aspect='auto')
        axes[1].set_title('Sagittal plane')
        axes[1].axis('off')

        axes[2].imshow(axial_img, cmap=cmap, vmin=0, vmax=3, aspect='auto')
        axes[2].set_title('Axial plane')
        axes[2].axis('off')

        fig.suptitle(f'Slice {slice_index}', fontsize=35)

        # Save the current figure as an image in memory
        fig.canvas.draw()
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
        images.append(image.convert('RGB'))

        plt.close(fig)

    # Save images as a GIF
    gif_path = report_dir + '/T1_segmented_slices.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)

def main():

    ''' Main function to generate the report for all subjects '''

    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        run2 = ['sub-B12974065', 'sub-B14131196']
        for row in csvreader:
            subject = row[0].strip()
            if subject not in problematic_sub:
                if os.path.exists(path_to_func + '/' + subject + '/ses-01/' + subject + '_ses-01_task-rest_bold_desc-ts-mc-cleaned-final-reg-template.nii.gz'):
                    report_dir = path_to_func + '/' + subject + '/ses-01' + '/report'
                    create_directory(report_dir)
                    QCmetrics(subject, report_dir, path_to_anat)
                    get_slices(subject, report_dir)
                    generate_html(subject, report_dir, path_to_func)
                else:
                    print(f"Error: Preprocessing for subject '{subject}' not found. Please, check.")

if __name__ == "__main__":
    main()


    
