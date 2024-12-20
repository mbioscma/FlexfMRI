#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:29:25 2024

@author: rtudela

Modified by Marc Biosca to adapt to A4 study parameters
"""
from logwrapper import log_execution

# test para aplicar corrección de slice timing en el pre-procesado de resting
# usamos la libreria de Afni
from nipype.interfaces import afni

import os
import pandas as pd

@log_execution
def afni_slice_timing(path_nii_in, tr, slice_timing_list, path_out):
    """
    https://nipype.readthedocs.io/en/1.1.8/interfaces/generated/interfaces.afni/preprocess.html
    https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTshift.html

    Parameters
    ----------
    nii_in : str
        DESCRIPTION. Ruta completa al nifti para hacer time slicing
    tr : float
        DESCRIPTION. reteirion time (in seconds)
    slice_timing_list : list of floats   
        DESCRIPTION. lista con los tiempos para cada slice
    path_out: str
        DESCRIPTION. Ruta del path donde grabar la imagen procesada con slice timing

    Returns
    -------
    name_out: str
        DESCRIPTION. Nombre del nifti de la imagen procesada por time slicing

    """
    # TR = 2.5
    tshift = afni.TShift()
    tshift.inputs.in_file = path_nii_in
    tshift.inputs.tzero = 0.0
    tshift.inputs.tr = '%.1fs' % tr # en segundos
    
    # ('Fourier' or 'linear' or 'cubic' or 'quintic' or 'heptic')
    # different interpolation methods (see 3dTshift for details) default = Fourier
    # tshift.inputs.interp = 'Fourier' 
    
    # slice_timing: (an existing file name or a list of items which are a float)
    #     time offsets from the volume acquisition onset for each slice
    #     argument: ``-tpattern @%s``
    #     mutually_exclusive: tpattern
    # tshift.inputs.slice_timing = list(np.arange(40) / TR)
    tshift.inputs.slice_timing = slice_timing_list
    
    # tpattern: ('alt+z' or 'altplus' or 'alt+z2' or 'alt-z' or 'altminus'
    #        or 'alt-z2' or 'seq+z' or 'seqplus' or 'seq-z' or 'seqminus' or a
    #        unicode string)
    #     use specified slice time pattern rather than one in header
    #     argument: ``-tpattern %s``
    #     mutually_exclusive: slice_timing
    # tshift.inputs.tpattern = 'alt+z'
    
    # slice_encoding_direction: ('k' or 'k-', nipype default value: k)
    #     Direction in which slice_timing is specified (default: k). If
    #     negative,slice_timing is defined in reverse order, that is, the
    #     first entry corresponds to the slice with the largest index, and the
    #     final entry corresponds to slice index zero. Only in effect when
    #     slice_timing is passed as list, not when it is passed as file.
    # tshift.inputs.slice_encoding_direction = 'k'
    
    path_out_temp = path_nii_in.split('.')[0]+'_slice-timing.nii.gz'
    name_out = path_out_temp.split('/')[-1]
    tshift.inputs.out_file = path_out + name_out
    tshift.inputs.outputtype = 'NIFTI_GZ'
    
    tshift.cmdline
    
    res = tshift.run()
    
   
    tshift.cmdline
    
    return name_out

import json
import numpy as np
import nibabel as nib

# json_file = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/ICAS_ADNI/tests/sub-ADNI002S0295_ses-m072/func/sub-ADNI002S0295_ses-m072_task-rest_bold.json'
# json_file = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/ICAS_ADNI/tests/sub-ADNI002S4213_ses-m072/func/sub-ADNI002S4213_ses-m072_task-rest_bold.json'
@log_execution
def return_tr_from_json(json_filename):
    """
    Function to return the repetion time fron the json files from the resting state acquisitions from ADNI   

    Parameters
    ----------
    json_filename : TYPE str
        DESCRIPTION. path for loading the json file of the resting state acquisition to process

    Returns
    -------
    tr : TYPE float
        DESCRIPTION. repetition time from the json file, units should be seconds.

    """    
    with open(json_file, 'r') as f_in:
        json_in = json.load(f_in)
    
    if 'RepetitionTime' in json_in.keys():
        tr = json_in['RepetitionTime']
    else:
        tr = np.NaN
    
    return tr

@log_execution
def return_parameters_from_json(json_file, path_nii_in):
    """
    

    Parameters
    ----------
    json_file : str
        DESCRIPTION. Ruta y nombre al archivo json de donde se lee el tiempo de repetición y el slice timing

    Returns
    -------
    tr : TYPE float
        DESCRIPTION. repetition time from json file 
    slicetiming : list
        DESCRIPTION. list of floats with time time for each slice from the json file (estaran??)

    """

    nii_in = nib.load(path_nii_in)

    nvoxels_in = nii_in.header.get_data_shape()

    num_slices = nvoxels_in[2]

    path_to_slice_timing = '/pool/home/AD_Multimodal/Estudio_A4/A4_FMRI_PARAMS_PRV2_02Jul2024.csv'

    data = pd.read_csv(path_to_slice_timing)

    subject = json_file.split('/')[-1].split('_')[0]
    
    with open(json_file, 'r') as f_in:
        json_in = json.load(f_in)
    
    if 'RepetitionTime' in json_in.keys():
        tr = json_in['RepetitionTime']
    else:
        tr = np.NaN


    subject_data = data[data['Subject'] == subject]

    if not subject_data.empty:
        # Check if there is any non-NaN and non-empty SliceTiming value
        if pd.notna(subject_data['SliceTiming']).any() and (subject_data['SliceTiming'] != '').all():
            slice_timing_str = subject_data['SliceTiming'].values[0]
            if slice_timing_str:  # Check if slice_timing_str is not empty
                slicetiming = [float(x) for x in slice_timing_str.split(';') if x != '']
                lista = slicetiming.copy()
                saltos = [lista[i+2]-lista[i] for i,el in enumerate(lista[:-2])]
                med = np.median(saltos)
                slice_timing_list = []
                for i in range(num_slices):
                    if i%2 != 0:
                        slice_timing_list.append((i-1)/2*med)
                    else:
                        slice_timing_list.append((i)/2*med + tr/2)
                    slicetiming = slice_timing_list
            else:
                slicetiming = np.NaN
        else:
            slicetiming = np.NaN
    else:
        slicetiming = np.NaN
    
    return tr, slicetiming    

## prueba
@log_execution
def aplica_time_slicing(name_path, name_nii_in, name_json_in, path_out):
    """
    

    Parameters
    ----------
    name_path : str
        DESCRIPTION. Path whith the nifti and json files, add / at the end
    name_nii_in : str
        DESCRIPTION. Name with the in nifti to make time slicing
    name_json_in : str
        DESCRIPTION. Name of the json file with the TR and slicing_time info
    path_out: str
        DESCRIPTION. Path to save the time slicing processing

    Returns
    -------
    path_out : str
        DESCRIPTION. path and name with the time slicing image processed

    """
    # path_json_in = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/ICAS_ADNI/tests/sub-ADNI002S4213_ses-m072/func/'
    # name_json_in = 'sub-ADNI002S4213_ses-m072_task-rest_bold.json'
    # path_json_in = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/ICAS_ADNI/tests/sub-ADNI002S0295_ses-m072/func/'
    # name_json_in = 'sub-ADNI002S0295_ses-m072_task-rest_bold.json'
    # path_json_in = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/ICAS_ADNI/tests/sub-ADNI027S5288_ses-m096/func/'
    # name_json_in = 'sub-ADNI027S5288_ses-m096_task-rest_bold.json'
    
    path_json_in = name_path

    if os.path.isfile(path_json_in+name_json_in):    
        tr, slice_timing_list  = return_parameters_from_json(path_json_in+name_json_in, path_json_in+name_nii_in)
        tr_label = 'from json'
        slicetime_label = 'from fmri_csv'
    else:
        tr = np.NaN
        slice_timing_list = np.NaN

    # path_nii_in = path_json_in+'sub-ADNI002S4213_ses-m072_task-rest_bold.nii.gz'
    # path_nii_in = path_json_in+'sub-ADNI002S0295_ses-m072_task-rest_bold.nii.gz'
    # path_nii_in = path_json_in+'sub-ADNI027S5288_ses-m096_task-rest_bold.nii.gz'
    path_nii_in = path_json_in+name_nii_in #'sub-ADNI002S4213_ses-m072_task-rest_bold.nii.gz'

    nii_in = nib.load(path_nii_in)

    pixdim_in = nii_in.header.get_zooms()

    nvoxels_in = nii_in.header.get_data_shape()

    num_slices = nvoxels_in[2]



    if np.isnan(tr): # si del json no hay tr lo cogemos de la cabecera del nifti
        tr = pixdim_in[3]
        tr_label = 'from header'
    
    #si no hay lista de time en el slicing lo generamos a partir del nº de slices y el tr
    if np.isnan(slice_timing_list).any():
        difference = tr/num_slices
        slice_timing_list = []
        for i in range(num_slices):
            if i%2 != 0:
                slice_timing_list.append((i-1)/2*difference)
            else:
                slice_timing_list.append((i)/2*difference + tr/2)
            
        slicetime_label = 'from header'
        

    name_out = afni_slice_timing(path_nii_in, tr, slice_timing_list, path_out)
    
    with open(path_out+'log_slice_timing.txt', 'w') as text_file:
        print('tr: {}'.format(tr), file=text_file)
        print('tr_label: {}'.format(tr_label), file=text_file)
        print('slice_timing_list: {}'.format(slice_timing_list), file=text_file)
        print('slice_time_label: {}'.format(slicetime_label), file=text_file)

        
    return name_out

#####################################################################
### fin bloque funciones ############################################
#####################################################################

# %%

if __name__ == "__main__":
    pass

