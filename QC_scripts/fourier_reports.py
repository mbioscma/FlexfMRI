# Author: Marc Biosca on 24.10.2014

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import scipy.stats as stats
from scipy.signal import hilbert, find_peaks, peak_widths
from scipy.integrate import simps
import json

def create_graphs(path_to_func, subject, save_nifti):
    path = os.path.join(path_to_func, subject, 'ses-01', subject + '_ses-01_task-rest_bold_desc-ts-mc-cleaned-final-reg-template.nii.gz')
    img = nib.load(path)
    data = img.get_fdata()

    time_points = np.arange(data.shape[-1])
    magnitude_imgs = []
    slices = []

    # ------------ 3D FOURIER TRANSFORM ------------

    for time_point in time_points:
        slice_3d = data[:, :, :, time_point]

        # Perform the Fourier Transform on the 3D slice
        ft_3d = np.fft.fftshift(np.fft.fftn(slice_3d))

        magnitude_ft = np.abs(ft_3d)

        slices.append(magnitude_ft.flatten())

        if save_nifti:
            magnitude_img = nib.Nifti1Image(magnitude_ft, img.affine, dtype=np.float32)
            magnitude_imgs.append(magnitude_img)
        
    #Take all of the 3d images in the list and create a 4d image
    path_save = os.path.join(path_to_func, subject, 'ses-01', 'report', 'fourier_analysis')
    os.makedirs(path_save, exist_ok=True)
    
    if save_nifti:
        final_magnitude_img = nib.concat_images(magnitude_imgs)
        nib.save(final_magnitude_img, os.path.join(path_save, subject + '_ses-01_task-rest_bold_desc-ts-mc-cleaned-final-reg-template_fourier.nii.gz'))
        print(subject + ' - Fourier Transform has been performed and saved as a nifti file')



    # ------------ PLOTTING ------------

    # Convert the list of slices to a 2D array
    slices = np.array(slices)

    slices_mean = np.mean(slices, axis=0)
    center = np.argmax(slices_mean)

    # Plot the 2D array
    plt.figure(figsize=(15, 10))
    plt.imshow(slices, cmap='gray', aspect='auto', vmin=0, vmax=1000)
    plt.colorbar()
    plt.title('Magnitude of Fourier Transform over Time')
    plt.xlabel('Voxel Index')
    plt.ylabel('Time Point')
    plt.savefig(os.path.join(path_save, subject + '_FT_Time.png'))

    plt.figure(figsize=(15, 10))
    plt.imshow(slices, cmap='gray', aspect='auto', vmin=0, vmax=1000)
    plt.colorbar()
    plt.title('Magnitude of Fourier Transform over Time')
    plt.xlabel('Voxel Index')
    plt.ylabel('Time Point')
    plt.xlim([center - 60*60, center + 60*60])
    plt.savefig(os.path.join(path_save, subject + '_FT_Time_Zoom.png'))


    slices_mean = np.mean(slices, axis=1)

    plt.figure(figsize=(15, 10))
    plt.plot(slices_mean)
    plt.title('Mean Magnitude of Fourier Transform over Time')
    plt.xlabel('Time Point')
    plt.ylabel('Mean Magnitude')
    plt.savefig(os.path.join(path_save, subject + '_FT_Mean_Time.png'))


    slices_mean = np.mean(slices, axis=0)
    center = np.argmax(slices_mean)

    # Calculate the Hilbert transform and amplitude envelope
    amplitude_envelope1 = high_envelope(slices_mean, 50)
    amplitude_envelope2 = high_envelope(slices_mean, 3500)

    amplitude_envelope11 = amplitude_envelope1[center-56*67//2:center+56*67//2]
    min_peak_distance = 400
    min_height = 0.1*np.max(amplitude_envelope1)
    prominence = 0.05*np.max(amplitude_envelope1)
    peaks, properties = find_peaks(amplitude_envelope11, distance=min_peak_distance, height=min_height, prominence=prominence)
    numer_peaks = len(peaks)
    peaks, properties = find_peaks(amplitude_envelope1, distance=min_peak_distance, height=min_height, prominence=prominence)

    plt.figure(figsize=(15, 10))
    plt.plot(slices_mean)
    plt.plot(amplitude_envelope2, label='Envelope 2')
    plt.title('Mean Voxel Magnitude of Fourier Transform over Time')
    plt.xlabel('Voxel Index')
    plt.ylabel('Mean Magnitude')
    plt.legend()
    plt.savefig(os.path.join(path_save, subject + '_FT_Mean_Voxel.png'))

    analysis_data = analyze_signal(slices_mean, subject, path_to_func,save_signal=True)

    # Save the analysis data dictionary as a csv file
    analysis_data.to_csv(os.path.join(path_save, subject + '_FT_Mean_Voxel_Analysis.csv'), index=False)


    plt.figure(figsize=(15, 10))
    plt.plot(slices_mean)
    plt.plot(amplitude_envelope1, label='Envelope 1')
    plt.title('Mean Voxel Magnitude of Fourier Transform over Time')
    plt.xlabel('Voxel Index')
    plt.ylabel('Mean Magnitude')
    plt.xlim([center - 60*60, center + 60*60])
    # Add text to show the number of peaks
    plt.text(0.4, 0.9, f'Number of Peaks: {numer_peaks}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=20)# Vertical lines at the ends
    plt.axvline(center - 56*67//2, color='red', linestyle='--')
    plt.axvline(center + 56*67//2, color='red', linestyle='--')
    #Mark the peaks with red circles size 5
    plt.plot(peaks, amplitude_envelope1[peaks], "o", color='red', markersize=5)
    # Mark the horizontal line at the height of 10
    plt.axhline(min_height, color='green', linestyle='--')

    plt.legend()
    plt.savefig(os.path.join(path_save, subject + '_FT_Mean_Voxel_Zoom.png'))

    slice_center = slices[:, center]

    plt.figure(figsize=(15, 10))
    plt.plot(slice_center)
    plt.title('Magnitude of Fourier Transform at Center Voxel over Time')
    plt.xlabel('Time Point')
    plt.ylabel('Magnitude')
    plt.savefig(os.path.join(path_save, subject + '_FT_Center_Voxel.png'))

    plt.close('all')

    print(subject + ' - Fourier Transform Data has been plotted and saved as png files')


def create_html_report(subject, base_path):
    report_dir = os.path.join(base_path, subject, 'ses-01', 'report', 'fourier_analysis')
    png_files = glob.glob(os.path.join(report_dir, '*.png'))
    png_theoretical_files = [os.path.join(report_dir, subject + '_FT_Mean_Voxel_Zoom.png'), os.path.join(report_dir, subject + '_FT_Mean_Voxel.png'), os.path.join(report_dir, subject + '_FT_Time_Zoom.png'), os.path.join(report_dir, subject + '_FT_Time.png'), os.path.join(report_dir, subject + '_FT_Mean_Time.png'), os.path.join(report_dir, subject + '_FT_Center_Voxel.png')]

    html_content = ['<html>', '<head>', '<title>Fourier Analysis Report</title>', '</head>', '<body>']
    html_content.append(f'<h1>Fourier Analysis Report for {subject}</h1>')

    # Invert the order of png_files
    png_final_files = []
    for theory in png_theoretical_files:
        if theory in png_files:
            png_final_files.append(theory)
            png_files.remove(theory)

    for png in png_final_files:
        file_name = os.path.basename(png)
        html_content.append(f'<div><h2>{file_name.replace("_", " ").replace(".png", "")}</h2>')
        html_content.append(f'<img src="{file_name}" width="800px"></div>')

    html_content.append('</body></html>')

    html_report_path = os.path.join(report_dir, 'fourier_analysis_report.html')
    with open(html_report_path, 'w') as f:
        f.write('\n'.join(html_content))

    print(f'Report created successfully for {subject}')

def high_envelope(signal, min_dist):

    # Find peaks in the signal
    peaks, _ = find_peaks(signal, distance=min_dist)
    
    # Interpolate between peaks to create the high envelope
    t = np.arange(len(signal))
    high_envelope = np.interp(t, peaks, signal[peaks])
    
    return high_envelope

def analyze_signal(signal, subject, path_to_func, save_signal=False):

    if save_signal:
        path_save = os.path.join(path_to_func, subject, 'ses-01', 'report', 'fourier_analysis')
        np.save(os.path.join(path_save, subject + '_FT_Mean_Voxel.npy'), signal)

        amplitude_envelope1 = high_envelope(signal, 50)
        np.save(os.path.join(path_save, subject + '_FT_Envelope1.npy'), amplitude_envelope1)

        amplitude_envelope2 = high_envelope(signal, 3500)
        np.save(os.path.join(path_save, subject + '_FT_Envelope2.npy'), amplitude_envelope2)

    # Calculate 3D Peak Position
    '''
    shape_3d = (56, 67, 56)
    half_length_of_window = (56*67)//2
    start_index = center - half_length_of_window
    mapped_main_peak_position = main_peak_position - start_index
    main_peak_position_3d = extract_3d_position(mapped_main_peak_position, shape_3d)
    if len(secondary_peaks) == 2:
        # Find the 3D coordinates of the secondary peaks
        first_secondary_peak_position = secondary_peaks[0] + start_index
        second_secondary_peak_position = secondary_peaks[1] + start_index
        first_secondary_peak_position_3d = extract_3d_position(first_secondary_peak_position, shape_3d)
        second_secondary_peak_position_3d = extract_3d_position(second_secondary_peak_position, shape_3d)
    '''

    #Normalize the signal
    signal = signal/np.max(signal)


    # Reduce signal to 5000 representative samples to compute the p_value
    signal1 = signal[::len(signal)//4000]

    # Normality test for the original signal
    stat, p_value = stats.shapiro(signal1)

    # Skewness and Kurtosis for the original signal
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)

    # Calculate the Hilbert transform and amplitude envelope
    amplitude_envelope1 = high_envelope(signal, 50)
    amplitude_envelope2 = high_envelope(signal, 3500)

    # Normality test for the amplitude envelope
    amplitude_envelope11 = amplitude_envelope1[::len(amplitude_envelope1)//4000]
    stat_env1, p_value_env1 = stats.shapiro(amplitude_envelope11)
    skewness_env1 = stats.skew(amplitude_envelope1)
    kurtosis_env1 = stats.kurtosis(amplitude_envelope1)

    amplitude_envelope22 = amplitude_envelope2[::len(amplitude_envelope2)//4000]
    stat_env2, p_value_env2 = stats.shapiro(amplitude_envelope22)
    skewness_env2 = stats.skew(amplitude_envelope2)
    kurtosis_env2 = stats.kurtosis(amplitude_envelope2)

    # ENV 1

    # Find all peaks in the signal
    center = np.argmax(amplitude_envelope1)
    envelope1 = amplitude_envelope1[center-56*67//2:center+56*67//2]
    min_peak_distance = 400
    min_height = 0.1*np.max(amplitude_envelope1)
    prominence = 0.05*np.max(amplitude_envelope1)
    
    peaks, properties = find_peaks(envelope1, distance=min_peak_distance, height=min_height, prominence=prominence)
    peaks_noprominence, _ = find_peaks(envelope1, distance=min_peak_distance, height=min_height)


    num_peaks = len(peaks)
    num_peaks_noprominence = len(peaks_noprominence)
    
    # Find the main peak (the highest peak)
    main_peak_idx = np.argmax(envelope1[peaks])
    main_peak_position = peaks[main_peak_idx]
    main_peak_amplitude = envelope1[main_peak_position]
    
    # Find secondary peaks (excluding the main peak)
    secondary_peaks = np.delete(peaks, main_peak_idx)
    
    # Secondary peak amplitudes and their average
    secondary_peak_amplitudes = envelope1[secondary_peaks] if len(secondary_peaks) > 0 else [0]
    mean_secondary_peak_amplitude = np.mean(secondary_peak_amplitudes) if len(secondary_peaks) > 0 else 0

    
    # Peak ratios
    peak_ratio = mean_secondary_peak_amplitude / main_peak_amplitude if main_peak_amplitude > 0 else 0
    
    # Distance between main peak and secondary peaks
    distance_to_left_secondary_peak = main_peak_position - secondary_peaks[secondary_peaks < main_peak_position].max() if any(secondary_peaks < main_peak_position) else 0
    distance_to_right_secondary_peak = secondary_peaks[secondary_peaks > main_peak_position].min() - main_peak_position if any(secondary_peaks > main_peak_position) else 0

    average_peak_distance = np.mean([distance_to_left_secondary_peak, distance_to_right_secondary_peak])

    # Find the minimum amplitude between the main peak and the two secondary peaks
    rel_min_amplitude = np.min(envelope1[secondary_peaks.min():main_peak_position]) if any(secondary_peaks < main_peak_position) else 0

    secondary_peaks_ratio = mean_secondary_peak_amplitude / rel_min_amplitude if rel_min_amplitude > 0 else 0

    # Calculate peak widths
    widths = peak_widths(envelope1, peaks, rel_height=0.5)[0]
    
    if len(widths) == 0:
        main_peak_width = 0
        secondary_peaks_width = 0
    elif len(widths) == 1:
        main_peak_width = widths[0]
        secondary_peaks_width = 0
    elif len(widths) == 3:
        main_peak_width = widths[0]
        secondary_peaks_width = widths[1]
    else:
        main_peak_width = 0
        secondary_peaks_width = 0


    # Energy under the peaks (using Simpson's rule for numerical integration)
    main_peak_area = simps(envelope1[max(0, main_peak_position - 150):min(len(envelope1), main_peak_position + 150)])
    secondary_peaks_area = np.average([simps(envelope1[max(0, sp - 150):min(len(envelope1), sp + 150)]) for sp in secondary_peaks]) if len(secondary_peaks) > 0 else 0

    # Energy ratios
    energy_ratio = secondary_peaks_area / main_peak_area if main_peak_area > 0 else 0

    # Absolute value of gradient for main peak and secondary peaks
    main_peak_gradient = np.abs(np.gradient(envelope1[max(0, main_peak_position - 150):main_peak_position]))
    secondary_peaks_gradient = np.average([np.abs(np.gradient(envelope1[max(0, sp - 150):sp])) for sp in secondary_peaks]) if len(secondary_peaks) > 0 else 0

    # Area under the curve
    auc = simps(amplitude_envelope1)
    auc_slice = simps(envelope1)

    # Statistical summaries
    data = {
        'Subject': subject,
        'Shapiro-Wilk Test (p-value)': p_value,
        'Shapiro-Wilk Test Statistic': stat,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Mean': np.mean(signal),
        'Standard Deviation': np.std(signal),
        'Maximum': np.max(signal),
        'Minimum': np.min(signal),
        'Median': np.median(signal),
        'Shapiro-Wilk Test Envelope 1 (p-value)': p_value_env1,
        'Shapiro-Wilk Test Envelope 1 Statistic': stat_env1,
        'Shapiro-Wilk Test Envelope 2 (p-value)': p_value_env2,
        'Shapiro-Wilk Test Envelope 2 Statistic': stat_env2,
        'Skewness Envelope 1': skewness_env1,
        'Kurtosis Envelope 1': kurtosis_env1,
        'Skewness Envelope 2': skewness_env2,
        'Kurtosis Envelope 2': kurtosis_env2,
        'Mean Envelope 1': np.mean(amplitude_envelope1),
        'Standard Deviation Envelope 1': np.std(amplitude_envelope1),
        'Maximum Envelope 1': np.max(amplitude_envelope1),
        'Minimum Envelope 1': np.min(amplitude_envelope1),
        'Median Envelope 1': np.median(amplitude_envelope1),
        'Mean Envelope 2': np.mean(amplitude_envelope2),
        'Standard Deviation Envelope 2': np.std(amplitude_envelope2),
        'Maximum Envelope 2': np.max(amplitude_envelope2),
        'Minimum Envelope 2': np.min(amplitude_envelope2),
        'Median Envelope 2': np.median(amplitude_envelope2),
        'Main Peak Amplitude': main_peak_amplitude,
        'Mean Secondary Peak Amplitude': mean_secondary_peak_amplitude,
        'Peak Ratio (Secondary/Main)': peak_ratio,
        'Main Peak Position': main_peak_position,
        'Distance to Left Secondary Peak': distance_to_left_secondary_peak,
        'Distance to Right Secondary Peak': distance_to_right_secondary_peak,
        'Average Peak Distance': average_peak_distance,
        'Main Peak Area': main_peak_area,
        'Secondary Peaks Area': secondary_peaks_area,
        'Energy Ratio (Secondary/Main)': energy_ratio,
        'Number of Secondary Peaks': len(secondary_peaks),
        'Number of Peaks': num_peaks,
        'Number of Peaks (no prominence)': num_peaks_noprominence,
        'Main Peak Width': main_peak_width,
        'Secondary Peaks Width': secondary_peaks_width,
        'Area under the curve': auc,
        'Area under the curve (slice)': auc_slice,
        'Main Peak Gradient': np.mean(main_peak_gradient),
        'Secondary Peaks Gradient': np.mean(secondary_peaks_gradient),
        'Secondary Peaks Ratio': secondary_peaks_ratio

    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data, index=[0])

    return df





def main():

    save_nifti = False

    path_to_func = "/pool/home/AD_Multimodal/Estudio_A4/fmri_slicetiming_outputderivatives/rs_preproc"

    subjects = pd.read_csv('/pool/home/AD_Multimodal/Estudio_A4/Scripts Preproc Slicing/subject_list.csv')['subject'].tolist()
    
    #problematic_subjects = pd.read_csv('/mnt/B468D0C568D0878E/usuarios/MarcBiosca/problematic_subjects.csv')['subject'].tolist()
    #subjects = [subject for subject in subjects if subject not in problematic_subjects]

    count = 1
    
    for subject in subjects:
        if os.path.exists(os.path.join(path_to_func, subject, 'ses-01', subject + '_ses-01_task-rest_bold_desc-ts-mc-cleaned-final-reg-template.nii.gz')):  
            print('Subject: ' + subject + ' (' + str(count) + '/' + str(len(subjects)) + ') - ' + str(round(count/len(subjects)*100)) + '%')
            create_graphs(path_to_func, subject, save_nifti)
            create_html_report(subject, path_to_func)
            print('----------------------------------')
            count += 1


if __name__ == '__main__':
    main()


