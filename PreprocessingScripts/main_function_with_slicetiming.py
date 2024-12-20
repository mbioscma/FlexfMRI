"""
Main executable file for the homebrew pipeline. 
Can be modified to change some params that are not in the config, but should not need to be touched as is.

Authors: Raul, raphael.wurm@gmail.com, Marc Biosca
Date: 2023

Modificacion 25.09.2024 para añadir el time slicing al proceado del rs

Modified 10.09.2024 to adapt to A4 study data - Marc Biosca
"""



# %% import from definition and logwrapper
from logwrapper import log_execution
from fun_defs import *
from fun_defs_slice_timing import * 


import json
import yaml
import csv
import os
import glob

def load_config(config_file):  # loads the config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('/mnt/B468D0C568D0878E/usuarios/MarcBiosca/Scripts Raul Preproc Slice Timing/config_slicetiming.yaml')

@log_execution
def resting_preprocessing_pipeline(sub_subject, ses_session):
    # %%
    # Initial variables #####################################################
    # A. ANATOMICAL IMAGE

    # path_base = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/Preprocessing_resting/test_subjects/'
    path_base = config['path_base']

    path_in_anat = path_base + sub_subject + '/' + ses_session + '/anat/'  # path_base + subject + session + anat

    # Initialize name_anat_nii_in
    name_anat_nii_in = None

    # Modified pattern to match both with and without 'run-nn' suffix
    # The pattern now looks for anything that ends with '_T1w.nii.gz' after the session part
    file_pattern = f'{path_in_anat}*T1w.nii.gz'
    print("Searching for files with pattern:", file_pattern)
    matching_files = glob.glob(f'{path_in_anat}*T1w.nii.gz')

    # Check if any files were found
    if matching_files:
        # Prefer a file with 'run-02' if available, otherwise take the first matching file
        selected_file = next((f for f in matching_files if 'run-02' in f), matching_files[0])
        print(selected_file)
        name_anat_nii_in = os.path.basename(selected_file)
    else:
        # Handle the case where no files are found
        # You can set name_anat_nii_in to a default value, raise an error, or keep it as None
        pass

    if name_anat_nii_in:
        name_t1_base = name_anat_nii_in.split('.')[0]
    else:
        # Handle the case where name_anat_nii_in is None
        # For example, you might set name_t1_base to a default value or raise an error
        name_t1_base = None

    # B. RESTING STATE fMRI ACQUISITION

    path_in_rs = path_base + sub_subject + '/' + ses_session + '/func/'  # path_base + subject + session + func

    name_rs_nii_in = sub_subject + '_' + ses_session + '_task-rest_bold.nii.gz'

    # 0.-1. Slice timing process label (to or not to be applied) (r.25.09.2024)
    label_slice_timing = config['label_slice_timing']

    # 0. Remove First TRs (N=5)
    n_vols_to_delete = config['n_vols_to_delete']
        
    # 1.d Spatial smoothing
    # Suavizado espacial de los volúmenes de la serie temporal. Spatial smoothing of the volumes from the temporal series.
    # se activa con un label si se quiere hacer. It is activated with the label if choosen to carry out.
    label_spatial_smoothing = config['label_spatial_smoothing']

    # name_template_T1 = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/Preprocessing_resting/Resting_HCB/scripts/templates/MNI152_T1_2mm_brain.nii.gz'
    name_template_T1 = config['name_template_t1']

    # %%
    # 2. TEMPORAL PROCESSING CLEAN

    # procesado de las series temporales
    # utilizando la función nilearn.signal.clean
    # https://nilearn.github.io/dev/modules/generated/nilearn.signal.clean.html
    # Permite aplicar detreanding, cofounds de movimiento, hacer el filtrado de frecuencias y escalar los valores de la serie
    # lo ponemos todo en la linea de procesado pero se puede hacer funciones si se quieren usar otros metodos para el filtrado temporal

    # datos para el procesado
    # lp: Low cutoff frequency in Hertz. If specified, signals above this frequency will be filtered out. If None, no low-pass filtering will be performed. Default=None.
    # hp: High cutoff frequency in Hertz. If specified, signals below this frequency will be filtered out. Default=None.
    tr = config['repetition_time']
    hp = config['high_pass_filter']
    lp = config['low_pass_filter']

    # 4. Nuissance - confounds
    label_confounds = config['use_confounds']
    label_use_global_signal_confound = config['use_global_signal_for_confounds']
    label_use_csf_and_wm_signal_confound = config['use_csf_and_wm_for_confounds']

    # Selección de algortimos para el procesado. Processing algorithms selection

    label_skull_stripping_t1 = config['skull_strip_method']  # ['hd-bet, 'bet-nipype', 'bet-fslpy']
    label_fsl = config['set_fls_module']  # ['fslpy', 'nipype']
    label_registro = config['set_registration_method'] # ['ants', 'flirt']

    # para hacer ICA-Aroma

    label_fnirt_for_aroma = config['use_fnirt_for_aroma']  # hace le registro del T1 al template con FNIRT
    label_aroma = config['run_aroma']

    # <-  # Initial variables #####################################################

    # %%
    # Pipeline ###############################################################
    # A. ANATOMICAL IMAGE

    # ponemos un directorio base2 para poner las nuevos derivatives en otro sitio
    path_out_derivatives = config['output_folder']
    # creamos directorio para las derivatives de la imagen anatómica del sujeto

    path_out_anat = path_base + path_out_derivatives + 'derivatives/anat/' + sub_subject + '/' + ses_session + '/'  # derivatives/anat/ + subject + session # es este el mejor orden?
    if not os.path.isdir(path_out_anat):
        os.makedirs(path_out_anat)

    if label_skull_stripping_t1 == 'hd-bet':
        # skull stripping hd-bet   (hay alternativa con BET de FSL desde fslpy o nipype)
        name_brain_seg_img_anat_in = skull_stripping_hd_bet(path_in_anat, name_anat_nii_in, path_out_anat)
    elif label_skull_stripping_t1 == 'bet-nipype':
        # skull stripping nipype fsl bet
        name_brain_seg_img_anat_in = skull_stripping_nipype_bet(path_in_anat, name_anat_nii_in, path_out_anat)
    elif label_skull_stripping_t1 == 'bet-fslpy':
        # skull stripping fslpy
        skull_stripping_fsl_bet(path_in_anat, name_anat_nii_in, path_out_anat, 0.7)
    else:
        # skull stripping hd-bet   (hay alternativa con BET de FSL desde fslpy o nipype)
        name_brain_seg_img_anat_in = skull_stripping_hd_bet(path_in_anat, name_anat_nii_in, path_out_anat)
        # %%
    # ===========================================
    # 4. NUISSANCE REGRESSION
    # ===========================================

    # 4.a SEGMENTATION OF ANATOMICAL SCAN

    path_deriv_anat = path_out_anat

    # 4.a.1 directory for the anatomical segmentation of the T1 acquisition

    path_seg_anat = path_deriv_anat + 'segmentation/'
    if not os.path.isdir(path_seg_anat):
        os.makedirs(path_seg_anat)

    os.chdir(path_seg_anat)

    # 4.a.2 Aplicamos la segmentación con FAST de FSL sobre la imagen anatómica solo cerebro

    seg_img_in = name_brain_seg_img_anat_in  # path_deriv_anat + 'sub-0022_ses-01_T1w_seg-brain-hdbet.nii.gz'

    out_fast = name_t1_base + '_fast_seg'  # este nombre se puede complicar para añadir el prefijo del sujeto con más detalle

    if label_fsl == 'fslpy':
        segmentation_fast_fslpy(seg_img_in, out_fast)
    elif label_fsl == 'nipype':
        segmentation_fast_nipype(seg_img_in, out_fast)  # no testeada
    else:
        segmentation_fast_fslpy(seg_img_in, out_fast)

    # %%
    # B. RESTING STATE fMRI ACQUISITION

    # Definir directorios para el preprocesado de resting

    path_rs_preproc = path_base + path_out_derivatives + 'derivatives/rs_preproc/' + sub_subject + '/' + ses_session + '/'  # derivatives/rs_preproc/ + subject + session # es este el mejor orden?
    if not os.path.isdir(path_rs_preproc):
        os.makedirs(path_rs_preproc)

    os.chdir(path_rs_preproc)
    
    ## ->
    ## 0.-1. Apply time slicing correction (using 3dTshift from Afni) (r. 25.09.2024)
    name_rs_base = name_rs_nii_in.split('.')[0]
    
    if label_slice_timing == True:
        name_slice_timing = name_rs_base + '_slice-timing.nii.gz'
        
        path_out = path_rs_preproc
        path_in_rs = path_in_rs
        name_nii_in = name_rs_nii_in
        name_json_in = name_rs_nii_in.split('.')[0]+'.json'
        name_out = aplica_time_slicing(path_in_rs, name_nii_in, name_json_in, path_out)
        
        # con el slice timing hecho hay que empezar el preprocesado desde este punto
        name_nii_in_rs = path_out + name_out
        name_rs_nii_in = name_out
        
    else:
        
        name_nii_in_rs = path_in_rs + name_rs_nii_in

        
    
    ## <-
    
    ## ->
    # 0. Remove First TRs (N=5)

    # name_rs_base = name_rs_nii_in.split('.')[0] #(r. 25.09.2024)

    name_out_rs = name_rs_base + '_desc-roi-5out.nii.gz'

    # name_nii_in_rs = path_in_rs + name_rs_nii_in #(r. 25.09.2024)
    name_nii_out_out5vol = path_rs_preproc + name_out_rs

    if label_fsl == 'fslpy':
        # delete 5 vols fslroi with fslpy
        remove_first_volumes_fslpy(name_nii_in_rs, name_nii_out_out5vol, n_vols_to_delete)
    elif label_fsl == 'nipype':
        remove_first_volumes_nipype(name_nii_in_rs, name_nii_out_out5vol, n_vols_to_delete)
    else:
        remove_first_volumes_fslpy(name_nii_in_rs, name_nii_out_out5vol, n_vols_to_delete)

    ## <-

    # ->
    # 1. MOTION CORRECTION

    # 1.a. Reference volume for the resting adquisition

    name_in_rs = name_rs_nii_in
    # name_in_rs = name_nii_out_out5vol # hay que usar para sacar la referencia el que ya no tiene los volúmenes o no?
    # name_nii_in_rs = path_in_rs + name_in_rs

    name_ref = name_rs_base + '_desc-reference-rs.nii.gz'
    name_ref_rs_out = path_rs_preproc + name_ref

    nii_ref_anat_rs = create_resting_anatomical_reference(name_nii_in_rs, name_ref_rs_out)

    # 1.b generamos mascara para los volumenes de resting utilizando fslpy Bet con 0.5 y la imagen de ref de rs
    skull_stripping_fsl_bet(path_rs_preproc, name_ref, path_rs_preproc, 0.5)

    # 1.c Motion correction usando mcflirt (usamos de referencia el promedio creado en 1.a)
    # Utilizamos el interface de nipype para mcflirt de fsl
    # se podria hacer con el wrapper de fslpy e incluso mirar como estos wrapper permiten trabajar con arrays
    name_rs_in_mc = path_rs_preproc + name_out_rs  # el resting sin los volúmenes iniciales
    name_ref_mc = path_rs_preproc + name_ref  # la referencia del resitng
    name_mc_out = path_rs_preproc + name_rs_base + '_desc-mc.nii.gz'

    res = rs_motion_correction_nipype_mcflirt(name_rs_in_mc, name_ref_mc, name_mc_out)

    # 1.d Spatial smoothing
    # Suavizado espacial de los volúmenes de la serie temporal
    # se activa con un label si se quiere hacer
    # Utilizamos la función SUSAN de FSL a través Nipype
    # Hace falta FWHM para los datos y un threshold de intensidad que hay que setear y probar para su funcionamiento en la función

    if label_spatial_smoothing == True:
        name_rs_in = name_mc_out
        name_rs_smoothed = path_rs_preproc + name_rs_base + '_desc-mc-smoothed.nii.gz'
        rs_smoothing_nipype_susan(name_rs_in, name_rs_smoothed)

    # %%
    # 2. TEMPORAL PROCESSING CLEAN

    # serie con el motion correction hecho anteriormente (hay que tener en cuenta si se ha hecho el suavizado espacial o no)
    if label_spatial_smoothing == True:
        nii_series_in = nib.load(name_rs_smoothed)
    else:
        nii_series_in = nib.load(name_mc_out)

    # el archivo con los confounds del registro (los angulos y desplazamiento de cada volumen)
    rp = np.loadtxt(name_mc_out + '.par')

    # cargamos la mascara del cerebro calculada anteriormente (1.b)
    name_mask = path_rs_preproc + name_rs_base + '_desc-reference-rs_seg-brain-bet_mask.nii.gz'
    nii_mask = nib.load(name_mask)
    # aplicamos la mascara a toda la serie multiplicando la mascara a la serie a la que se ha corregido el movimiento
    data_mask = nii_mask.get_fdata()
    # nii_series_masked = apply_mask(nii_series_in, nii_mask)
    data_tseries = nii_series_in.get_fdata()
    data_masked = np.zeros(nii_series_in.shape)
    for n in range(nii_series_in.get_fdata().shape[3]):
        data_masked[:, :, :, n] = data_tseries[:, :, :, n] * data_mask

    # si se graba la serie motion corrected enmascarada, se puede comentar
    nii_series_masked = nib.Nifti1Pair(data_masked, nii_series_in.affine)

    name_nii_series_masked = path_rs_preproc + name_rs_base + '_desc-ts-mc-masked.nii.gz'
    nib.save(nii_series_masked, name_nii_series_masked)

    # liberamos memoria borrando arrays que no se vuelven a usar
    data_masked = 0
    del data_masked

    data_tseries = 0
    del data_tseries

    data_mask = 0
    del data_mask

    del nii_mask
    del nii_series_in

    ############### Antes de aplicar la corrección temporal hay que calcular registros
    # %%
    # 3. REGISTRATION

    # los procesos necesarios para llevar las adquisiciones al espacio standard

    # 3.a. registro de la referencia del rs (a la que se han registrado todos los volumenes del rs) al T1 del sujeto

    path_t1_brain = path_out_anat
    # name_ref_t1 = path_t1_brain + 'sub-0022_ses-01_T1w_seg-brain-hdbet.nii.gz'
    name_ref_t1 = name_brain_seg_img_anat_in

    name_ref_rs_out_masked = name_rs_base + '_desc-reference-rs_seg-brain-bet.nii.gz'
    name_mov_rs_ref = path_rs_preproc + name_ref_rs_out_masked

    name_rs_reg = path_rs_preproc + name_rs_base + '_desc-rs-ref-reg-2-t1.nii.gz'

    # name_transf_rs_2_t1 =name_rs_reg.split('.')[0]+'.mat'
    if label_registro == 'ants':
        rsref_reg, transform_rsref_2_t1 = ants_registration_rs_reference_2_subject_T1(name_ref_t1, name_mov_rs_ref,
                                                                                      name_rs_reg)  # , name_transf_rs_2_t1)
    elif label_registro == 'flirt':
        name_transf_rs_2_t1 = path_rs_preproc + name_rs_base + '_desc-rs-ref-reg-2-t1.mat'

        flirt_registration_rs_reference_2_subject_T1(name_ref_t1, name_mov_rs_ref, name_rs_reg, name_transf_rs_2_t1)
    else:
        rsref_reg, transform_rsref_2_t1 = ants_registration_rs_reference_2_subject_T1(name_ref_t1, name_mov_rs_ref,
                                                                                      name_rs_reg)  # , name_transf_rs_2_t1)

    # 3.b registro del T1 del sujeto al template

    # name_template = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/Preprocessing_resting/Resting_HCB/scripts/templates/MNI152_T1_2mm_brain.nii.gz'
    name_template = name_template_T1

    name_mov_t1 = name_ref_t1

    # name_t1_reg = path_t1_brain + 'sub-0022_ses-01_T1w_seg-brain-hdbet_desc-reg2-template.nii.gz'
    name_t1_reg = path_t1_brain + name_t1_base + '_seg-brain-hdbet_desc-reg2-template.nii.gz'

    # name_transf_t1_2_template =name_t1_reg.split('.')[0]+'.mat'
    if label_registro == 'ants':
        t1_reg, transform_t1_2_template, transform_list = ants_registration_subject_T1_2_template(name_template,
                                                                                                  name_mov_t1,
                                                                                                  name_t1_reg)  # , name_transf_t1_2_template)
    elif label_registro == 'flirt':
        name_transf_t1_2_template = path_t1_brain + name_t1_base + '_seg-brain-hdbet_desc-reg2-template.mat'

        flirt_registration_subject_T1_2_template(name_template, name_mov_t1, name_t1_reg, name_transf_t1_2_template)
    else:
        t1_reg, transform_t1_2_template, transform_list = ants_registration_subject_T1_2_template(name_template,
                                                                                                  name_mov_t1,
                                                                                                  name_t1_reg)  # , name_transf_t1_2_template)

    # 3.b.2 Registro de la imagen estructural T1 enmascarda al template usando FNIRT. Necesario para aplicar ICA-AROMA
    if label_fnirt_for_aroma == True:
        name_mov_t1 = name_brain_seg_img_anat_in  # imagen T1 del sujeto solo cerebro (after bet)
        if label_registro == 'flirt':
            name_aff_t1_2_template = name_transf_t1_2_template.split('.')[0] + '_flirt.mat'
        else:
            name_transf_t1_2_template = path_t1_brain + name_t1_base + '_seg-brain-hdbet_desc-reg2-template.mat'

            flirt_registration_subject_T1_2_template(name_template, name_mov_t1, name_t1_reg, name_transf_t1_2_template)
            name_aff_t1_2_template = name_transf_t1_2_template.split('.')[0] + '_flirt.mat'

        name_t1_reg_fnirt = path_t1_brain + name_t1_base + '_desc_t1-bet-fnirt-2-template'
        fnirt_registration_subject_T1_2_template(name_template, name_mov_t1, name_aff_t1_2_template, name_t1_reg_fnirt)

        # 3.c. Registro del resting al template con las transformaciones anteriores (este es opcional, ya que no es la serie filtrada final)

    # name_moving_rs = path_rs_preproc + 'sub-0022_ses-01_task-rest_bold_desc-ts-mc-masked.nii.gz'
    name_moving_rs = path_rs_preproc + name_rs_base + '_desc-ts-mc-masked.nii.gz'

    name_ref_template = name_template

    if label_registro == 'ants':
        transform_rs_2_t1 = transform_rsref_2_t1
        transform_t1_2_template2 = transform_t1_2_template

        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ts-mc-cleaned-masked-reg-template.nii.gz'

        rs_reg = ants_transform_rs_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1,
                                              transform_t1_2_template2, name_rs_reg_out)
    elif label_registro == 'flirt':
        name_transform_rs_2_t1 = name_transf_rs_2_t1.split('.')[0] + '_prueba_flirt.mat'
        name_transform_t1_2_template = name_transf_t1_2_template.split('.')[0] + '_prueba_flirt.mat'
        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ts-mc-cleaned-masked-reg-template.nii.gz'

        flirt_transform_rs_2_template(name_moving_rs, name_ref_template, name_transform_rs_2_t1,
                                      name_transform_t1_2_template)  # , name_rs_reg_out)
    else:
        transform_rs_2_t1 = transform_rsref_2_t1
        transform_t1_2_template2 = transform_t1_2_template

        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ts-mc-cleaned-masked-reg-template.nii.gz'

        rs_reg = ants_transform_rs_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1,
                                              transform_t1_2_template2, name_rs_reg_out)

        # 3.d Aplicamos el registro a la mascara del cerebro para llevarlo al espacio del template. Hay que usar otra funcion por la interpolación

    if label_registro == 'ants':
        # name_moving_rs = path_rs_preproc + name_rs_base+'_desc-reference-reg-rs_seg-brain-bet_mask.nii.gz'
        name_moving_rs = path_rs_preproc + name_rs_base + '_desc-reference-rs_seg-brain-bet_mask.nii.gz'

        name_ref_template = name_template_T1

        transform_rs_2_t1 = transform_rsref_2_t1
        transform_t1_2_template2 = transform_t1_2_template

        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ref-reg-rs_seg-brain-bet_mask-reg-template.nii.gz'

        rs_reg = ants_transform_mask_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1,
                                                transform_t1_2_template2, name_rs_reg_out)
    elif label_registro == 'flirt':
        # name_moving_mask = path_rs_preproc + name_rs_base+'_desc-reference-reg-rs_seg-brain-bet_mask.nii.gz'
        name_moving_mask = path_rs_preproc + name_rs_base + '_desc-reference-rs_seg-brain-bet_mask.nii.gz'
        name_ref_template = name_template

        name_transform_rs_to_template = name_moving_rs.split('.')[0] + '_reg_rs_to_template_flirt.mat'

        name_mask_reg_out = path_rs_preproc + name_rs_base + '_desc-ref-reg-rs_seg-brain-bet_mask-reg-template_prueba_flirt.nii.gz'

        flirt_transform_mask_2_template(name_moving_mask, name_ref_template, name_transform_rs_to_template,
                                        name_mask_reg_out)
    else:
        # name_moving_rs = path_rs_preproc + name_rs_base+'_desc-reference-reg-rs_seg-brain-bet_mask.nii.gz'
        name_moving_rs = path_rs_preproc + name_rs_base + '_desc-reference-rs_seg-brain-bet_mask.nii.gz'
        name_ref_template = name_template_T1

        transform_rs_2_t1 = transform_rsref_2_t1
        transform_t1_2_template2 = transform_t1_2_template

        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ref-reg-rs_seg-brain-bet_mask-reg-template.nii.gz'

        rs_reg = ants_transform_mask_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1,
                                                transform_t1_2_template2, name_rs_reg_out)

    ###############################################################################

    # ===========================================
    # 4. NUISSANCE REGRESSION
    # ===========================================

    # prueba confounds señales rs

    # 1. registro de las segmentaciones csf y wm al espacio de la rs

    name_img_ref = name_ref_rs_out  # la imagen de referencia para quitar el movimiento al rs

    # la transformación inversa de rs a T1. transformación de T1 a rs
    name_transform_T1_2_rs = path_rs_preproc + name_rs_base + '_desc-rs-ref-reg-2-t1_inv-transform.mat'  # matriz generada ya?

    name_img_in_seg_csf = path_seg_anat + name_t1_base + '_fast_seg_seg_0.nii.gz'

    name_mask_reg_csf = path_seg_anat + name_t1_base + '_fast_seg_seg_0_reg_2_rs.nii.gz'

    # name_img_in = name_img_in_seg_csf
    # name_img_ref = name_img_ref
    # name_transform_T1_2_rs = name_transform_T1_2_rs
    # name_mask_reg = name_mask_reg_csf
    reg_seg_csf = ants_transform_T1_segmentations_2_rs_space(name_img_in_seg_csf, name_img_ref, name_transform_T1_2_rs,
                                                             name_mask_reg_csf)

    name_img_in_seg_wm = path_seg_anat + name_t1_base + '_fast_seg_seg_2.nii.gz'

    name_mask_reg_wm = path_seg_anat + name_t1_base + '_fast_seg_seg_2_reg_2_rs.nii.gz'

    reg_seg_wm = ants_transform_T1_segmentations_2_rs_space(name_img_in_seg_wm, name_img_ref, name_transform_T1_2_rs,
                                                            name_mask_reg_wm)

    # 2 sacamos las señales del resting
    # name_rs_in = path_rs_preproc + 'sub-0022_ses-01_task-rest_bold_desc-roi-5out.nii.gz' # quizas no sea la imagen de donde sacar los promedios
    # name_rs_in = name_rs_reg_out #usamos la imagen con los volumenes registrados al template y aplicada la mascara del cerebro (todavía no seria)
    name_rs_in = name_nii_series_masked  # usamos la imagen con los volumenes registrados a la referencia del rs y aplicada la mascara del cerebro
    # name_mask_brain = path_rs_preproc + name_rs_base + '_desc-reference-reg-rs_seg-brain-bet_mask.nii.gz'
    name_mask_brain = path_rs_preproc + name_rs_base + '_desc-reference-rs_seg-brain-bet_mask.nii.gz'
    name_mask_csf = name_mask_reg_csf
    name_mask_wm = name_mask_reg_wm
    rs_in = nib.load(name_rs_in).get_fdata()
    mask_brain = nib.load(name_mask_brain).get_fdata()
    mask_csf = nib.load(name_mask_csf).get_fdata()
    mask_wm = nib.load(name_mask_wm).get_fdata()
    signal_global, signal_csf, signal_wm = extract_average_signals_from_rs(rs_in, mask_brain, mask_csf, mask_wm)

    transf_rp = np.loadtxt(name_mc_out + '.par')
    confounds_df = create_confounds_for_nilearn_clean_img(signal_global, signal_csf, signal_wm, transf_rp, path_rs_preproc)

    if label_use_global_signal_confound == False:
        confounds_df_without_global = confounds_df.drop(['global_signal'], axis=1).copy()
        confounds_matrix = confounds_df_without_global.values
    elif label_use_csf_and_wm_signal_confound == False:
        confounds_df_without_signals = confounds_df.drop(['global_signal', 'csf_signal', 'wm_signal'], axis=1).copy()
        confounds_matrix = confounds_df_without_signals.values
    else:
        confounds_matrix = confounds_df.values

    # TEMPORAL PROCESSING CLEAN

    # try to get tr from the nifti header
    try:
        tr = nib.load(name_nii_in_rs).header.get_zooms()[3]
        print(f"the TR used for this was {tr}")
    except Exception as e:
        print(f"Error occurred: {e}. Using default TR value.")

    # aplicamos la función de nilearn que procesa las series temporales
    if label_confounds == True:
        data_nl_clean = nlimg.clean_img(nii_series_masked,
                                        detrend=True,
                                        standardize='zscore',
                                        confounds=confounds_matrix,
                                        t_r=tr,
                                        # filter = 'butterworth',
                                        low_pass=lp,
                                        high_pass=hp,
                                        ensure_finite=True)
    else:
        data_nl_clean = nlimg.clean_img(nii_series_masked,
                                        detrend=True,
                                        standardize='zscore',
                                        # confounds = confounds_matrix,
                                        t_r=tr,
                                        # filter = 'butterworth',
                                        low_pass=lp,
                                        high_pass=hp,
                                        ensure_finite=True)

    # nombre y grabamos el nifti de la serie filtrada y con la mascara aplicada
    name_ts_cleaned_out = path_rs_preproc + name_rs_base + '_desc-ts-cleaned-confounds-time.nii.gz'
    nib.save(data_nl_clean, name_ts_cleaned_out)

    # liberamos memoria
    data_nl_clean = 0
    del data_nl_clean
    del rs_in
    del mask_brain
    del mask_csf
    del mask_wm

    # %%
    ##################################################
    # ICA AROMA
    #################################################
    if label_aroma == True:

        name_rs_in_for_aroma = name_nii_series_masked

        if label_registro == 'flirt':
            # name_aff_rs_2_t1 = name_transf_rs_2_t1
            name_aff_rs_2_t1 = name_transf_rs_2_t1.split('.')[0] + '_flirt.mat'
        else:
            name_transf_rs_2_t1 = path_rs_preproc + name_rs_base + '_desc-rs-ref-reg-2-t.mat'

            flirt_registration_rs_reference_2_subject_T1(name_ref_t1, name_mov_rs_ref, name_rs_reg, name_transf_rs_2_t1)
            name_aff_rs_2_t1 = name_transf_rs_2_t1.split('.')[0] + '_flirt.mat'

        name_warp_t1_2_template = name_t1_reg_fnirt + '_warp_fnirt.nii.gz'
        name_motion_parameters_mcflirt = name_mc_out + '.par'

        dir_out_aroma = path_rs_preproc + 'melodic_aroma/'
        # if not os.path.isdir(dir_out_aroma):
        #     os.makedirs(dir_out_aroma)

        # os.chdir(dir_out_aroma)

        try:
            apply_ica_aroma_to_rs(name_rs_in_for_aroma, name_aff_rs_2_t1, name_warp_t1_2_template,
                                  name_motion_parameters_mcflirt, tr, dir_out_aroma)
            label_aroma_runned = True
        except Exception as e:
            print('Ica-Aroma fails. Could not be computed.')
            label_aroma_runned = False

        if label_aroma_runned == True:
            name_temporal_aroma_cleaned_out = path_rs_preproc + name_rs_base + '_desc-ts-aroma-temporal-cleaned.nii.gz'
            temporal_filtering_after_aroma(dir_out_aroma, tr, lp, hp, name_temporal_aroma_cleaned_out)

    # %%
    # registro a template de la serie de resting tras los confounds y el filtrado temporal del img_clean
    # 3.c.2 Registro del resting al template con las transformaciones anteriores

    # name_moving_rs = path_rs_preproc + 'sub-0022_ses-01_task-rest_bold_desc-ts-mc-masked.nii.gz'
    name_moving_rs = name_ts_cleaned_out

    name_ref_template = name_template

    if label_registro == 'ants':
        transform_rs_2_t1 = transform_rsref_2_t1
        transform_t1_2_template2 = transform_t1_2_template

        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ts-mc-cleaned-final-reg-template.nii.gz'

        rs_reg = ants_transform_rs_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1,
                                              transform_t1_2_template2, name_rs_reg_out)
    elif label_registro == 'flirt':
        name_transform_rs_2_t1 = name_transf_rs_2_t1.split('.')[0] + '_prueba_flirt.mat'
        name_transform_t1_2_template = name_transf_t1_2_template.split('.')[0] + '_prueba_flirt.mat'
        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ts-mc-cleaned-final-reg-template.nii.gz'

        flirt_transform_rs_2_template(name_moving_rs, name_ref_template, name_transform_rs_2_t1,
                                      name_transform_t1_2_template)  # , name_rs_reg_out)
    else:
        transform_rs_2_t1 = transform_rsref_2_t1
        transform_t1_2_template2 = transform_t1_2_template

        name_rs_reg_out = path_rs_preproc + name_rs_base + '_desc-ts-mc-cleaned-final-reg-template.nii.gz'

        rs_reg = ants_transform_rs_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1,
                                              transform_t1_2_template2, name_rs_reg_out)

        ##
    # %%
    # registro al template de la serie temporal corregida con aroma y filtrada temporalmente
    if (label_aroma == True) and (label_aroma_runned == True):

        # registro a template de la serie de resting trasel ica-aroma y el filtrado temporal del img_clean
        # 3.c.3 Registro del resting al template con las transformaciones anteriores

        # name_moving_rs = path_rs_preproc + 'sub-0022_ses-01_task-rest_bold_desc-ts-mc-masked.nii.gz'
        name_moving_rs_aroma = name_temporal_aroma_cleaned_out

        name_ref_template = name_template

        if label_registro == 'ants':
            transform_rs_2_t1 = transform_rsref_2_t1
            transform_t1_2_template2 = transform_t1_2_template

            name_rs_reg_out_aroma = path_rs_preproc + name_rs_base + '_desc-ts-aroma-temporal-cleaned-final-reg-template.nii.gz'

            rs_reg = ants_transform_rs_2_template(name_moving_rs_aroma, name_ref_template, transform_rs_2_t1,
                                                  transform_t1_2_template2, name_rs_reg_out_aroma)
        elif label_registro == 'flirt':
            name_transform_rs_2_t1 = name_transf_rs_2_t1.split('.')[0] + '_prueba_flirt.mat'
            name_transform_t1_2_template = name_transf_t1_2_template.split('.')[0] + '_prueba_flirt.mat'
            name_rs_reg_out_aroma = path_rs_preproc + name_rs_base + '_desc-ts-mc-aroma-temporal-cleaned-final-reg-template.nii.gz'

            flirt_transform_rs_2_template(name_moving_rs_aroma, name_ref_template, name_transform_rs_2_t1,
                                          name_transform_t1_2_template)  # , name_rs_reg_out)
        else:
            transform_rs_2_t1 = transform_rsref_2_t1
            transform_t1_2_template2 = transform_t1_2_template

            name_rs_reg_out_aroma = path_rs_preproc + name_rs_base + '_desc-ts-aroma-temporal-cleaned-final-reg-template.nii.gz'

            rs_reg = ants_transform_rs_2_template(name_moving_rs_aroma, name_ref_template, transform_rs_2_t1,
                                                  transform_t1_2_template2, name_rs_reg_out_aroma)

    return
    ###################################################################



@log_execution
def delete_files_except(bids_derivatives_folder, keep, run_bool):
    if not run_bool:
        return

    # Walk through the folder and subdirectories
    for root, dirs, files in os.walk(bids_derivatives_folder):
        for file in files:
            if not any(substring in file for substring in keep):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"File deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")


subject_list = config['subject_list']

# Read the CSV file
def main():
    with open(subject_list, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)

        # Iterate through the rows
        for row in reader:
            try:
                sub_subject = row['subject']
                ses_session = row['session']

                # Call the preprocessing function with the subject and session IDs
                resting_preprocessing_pipeline(sub_subject, ses_session)

                # Set parameters for delete_files_except function
                bids_master = config['bids_master']  # path to folder, absolute
                files_to_keep = config['files_to_keep']  # strings to be searched for
                delete_bool = config['delete']  # if False not run

                # Call the delete function with the specified parameters
                delete_files_except(bids_master, files_to_keep, delete_bool)

            except Exception as e:
                log_records.append({"subject": sub_subject, "session": ses_session, "error": str(e)})
                print(f"Error processing subject {sub_subject}, session {ses_session}: {str(e)}")
                continue



from logwrapper import log_execution, log_records

def save_log_records(config):
    output_path = os.path.join(config['path_base'], config['output_folder'], 'log_data.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create the directory if it does not exist
    with open(output_path, 'w') as json_file:
        json.dump(log_records, json_file)



if __name__ == "__main__":
    main()
    save_log_records(config)
