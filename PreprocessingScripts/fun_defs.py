# Resting pre-processing pipeline and initial variables
from logwrapper import log_execution
from nipype.interfaces import fsl

import os
import nilearn.image as nlimg
import nibabel as nib
import numpy as np
import nitime.fmri.io as fmri_io
import pandas as pd

# %%
# ===========================================
# PREPROCESSING OF ANATOMICAL SCAN
# ===========================================

# %% Skull Stripping HD-BET
## ->
# Skull stripping of the anatomical image using HD-BET
#
# name_anat_nii_in : nombre del archivo con la imagen anatómica de entrada (en formato nii.gz). Name of the etrance file with the anatomical image (in nii..gz format)
# pathh_in: ruta al directorio con el nii anatomico de entrada. Route to the directory with the anatomical image
# path_out: ruta al directorio de salida. Route to the exit directory
import subprocess


@log_execution
def skull_stripping_hd_bet(path_in, name_anat_nii_in, path_out):
    nii_in = path_in + name_anat_nii_in
    nii_out = path_out + name_anat_nii_in.split('.')[0] + '_seg-brain-hdbet.nii.gz'
    hdbet_run = subprocess.run(['hd-bet',
                                '-i', nii_in,
                                '-o', nii_out,
                                '-mode', 'fast',
                                '-device', 'cpu',  # se puede usar tarjeta gráfica para accelerar
                                '-tta', '0'])

    return nii_out


## <-
# %%

# %% Skull Stripping BET fslpy
## ->
# Skull stripping con el BET de fsl a traves de un wrapper de fslpy
#
# name_anat_nii_in : nombre del archivo con la imagen anatómica de entrada (en formato nii.gz). Name of the etrance file with the anatomical image (in nii..gz format)
# path_in: ruta al directorio con el nii anatomico de entrada. Route to the directory with the anatomical image
# path_out: ruta al directorio de salida. Route to the exit directory
from pathlib import Path
from fsl.wrappers import bet


@log_execution
def skull_stripping_fsl_bet(path_in, name_anat_nii_in, path_out, fracint):
    nii_in = path_in + name_anat_nii_in
    nii_out = path_out + name_anat_nii_in.split('.')[0] + '_seg-brain-bet.nii.gz'

    bet(Path(nii_in), Path(nii_out),
        mask=True,
        seg=True,
        fracintensity=fracint,  # 0.7 para T1w, 0.5 para ref rs Bold
        robust=True)  # pero puede que haya de ser False -n

    return nii_out


## <-
# %%

# %%  Skull stripping BET con Nipype
# Skull stripping con Nipype
#
# name_anat_nii_in : nombre del archivo con la imagen anatómica de entrada (en formato nii.gz). Name of the etrance file with the anatomical image (in nii..gz format)
# path_in: ruta al directorio con el nii anatomico de entrada. Route to the directory with the anatomical image
# path_out: ruta al directorio de salida. Route to the exit directory

# from nipype.interfaces import fsl

@log_execution
def skull_stripping_nipype_bet(path_in, name_anat_nii_in, path_out):
    nii_in = path_in + name_anat_nii_in
    nii_out = path_out + name_anat_nii_in.split('.')[0] + '_seg-brain-npbet.nii.gz'

    btr = fsl.BET()

    btr.inputs.in_file = nii_in
    btr.inputs.frac = 0.7
    btr.inputs.mask = True
    btr.inputs.robust = True
    btr.inputs.out_file = nii_out
    btr.cmdline

    res = btr.run()

    return nii_out


## <-
# %%

# ==========================================
# PREPROCESSING OF FUNCTIONAL SCAN
# ==========================================

# %%
## 0. Remove first TRs (N=5)

## quitamos los primeros N volumenes de la adquisición del resting. We take out the first N volumes from the resitng acquisition

## usando fslroi con Nipype. with fslroi using Nipype
## nii_in: ruta completa a la imagen de resting de entrada. Complete route to the entance resting image
## nii_out: ruta completa al nifti de salida. Complete route to the exit nifti
## n_vols_to_remove: número de volumenes a quitar en el inicio de la adquisición. Number of volumes to remove from the beginning of the acquisition
@log_execution
def remove_first_volumes_nipype(nii_in, nii_out, n_vols_to_delete):
    in_file_rs = nii_in
    roi_file_rs = nii_out

    # fslroi = fsl.ExtractROI(in_file = in_file_rs, t_min = (n_vols_to_delete-1), t_size=(n_temp_vols-n_vols_to_delete))
    fslroi = fsl.ExtractROI(in_file=in_file_rs, roi_file=roi_file_rs, t_min=(n_vols_to_delete), t_size=-1)
    fslroi.cmdline
    fslroi.run()

    return


# %%
## usando fslroi con fslpy wrappper
## nii_in: ruta completa a la imagen de resting de entrada. Complete route to the entance resting image
## nii_out: ruta completa al nifti de salida. Complete route to the exit nifti
## n_vols_to_remove: número de volumenes a quitar en el inicio de la adquisición. Number of volumes to remove from the beginning of the acquisition
from fsl.wrappers import fslroi


@log_execution
def remove_first_volumes_fslpy(nii_in, nii_out, n_vols_to_delete):
    in_file_rs = nii_in
    roi_file_rs = nii_out

    fslroi(in_file_rs, roi_file_rs, (n_vols_to_delete), -1)

    return


# %%
# Rs processing
# 1. Motion Correction

# Primer paso referencia para los registros de motion correction. First step. Generate the reference image for the motion correction registrations
# La construimos con los primeros cinco volumenes de la adquisición de resting. We generate it with the 5 first volumes of the resting
# en muchos casos estos volúmenes no han alcanzado la estabilidad en T1 y son buena referencia anatómica.

@log_execution
def create_resting_anatomical_reference(name_rs_in, name_ref_rs_out):
    nii_in = nib.load(name_rs_in)
    # the reference volume is compute with the average of the first 5 vols of the rs
    data_ref = np.mean(nii_in.get_fdata()[:, :, :, 0:5], axis=3)

    nii_ref_rs_out = nib.Nifti1Pair(data_ref, nii_in.affine)

    nib.save(nii_ref_rs_out, name_ref_rs_out)

    return nii_ref_rs_out


# %%
# from nipype.interfaces import fsl

@log_execution
def rs_motion_correction_nipype_mcflirt(name_rs_in_mc, name_ref_mc, name_mc_out):
    mcflt = fsl.MCFLIRT()

    mcflt.inputs.in_file = name_rs_in_mc
    mcflt.inputs.ref_file = name_ref_mc
    mcflt.inputs.cost = 'mutualinfo'

    mcflt.inputs.save_plots = True  # path_rs_preproc + 'mc_mean_file.nii.gz'
    mcflt.inputs.out_file = name_mc_out  # path_rs_preproc + 'mc_reg_file.nii.gz'
    # mcflt.inputs.save_rms = True #path_rs_preproc + 'mc_par_file.txt'

    res = mcflt.run()

    return res


# %%
# Suavizado espacial de los volúmenes de la serie temporal usando SUSAN. Spatial smoothing of the volume from the temporal series with SUSAN method
# Con Nipype fsl SUSAN

@log_execution
def rs_smoothing_nipype_susan(name_rs_in, name_rs_smoothed):
    sus = fsl.SUSAN()
    sus.inputs.in_file = name_rs_in
    sus.inputs.brightness_threshold = 5.0
    sus.inputs.fwhm = 7  # en (mm)
    sus.inputs.out_file = name_rs_smoothed

    result = sus.run()

    return result


# %%
# funciones para hacer los registros. Functions for the registration
#
# https://github.com/ANTsX/ANTsPy
import ants  # pip install antspyx


# registro de la referencia de resting al T1 del sujeto. registration of the resting reference to the T1 anatomical image of the subject

@log_execution
def ants_registration_rs_reference_2_subject_T1(name_ref_t1, name_mov_rs_ref, name_rs_reg):  # , name_transf_rs_2_t1):

    ref_t1 = ants.image_read(name_ref_t1)
    mov_rs_ref = ants.image_read(name_mov_rs_ref)

    name_transf_rs_2_t1 = name_rs_reg.split('.')[
                              0] + '_transform.mat'  # dependiendo del tipo de registro hay que grabar la transformación de una forma o otra
    # name_transf_rs_2_t1 =name_rs_reg.split('.')[0]+'_transform.nii.gz'

    transform_rsref_2_t1 = ants.registration(fixed=ref_t1,
                                             moving=mov_rs_ref,
                                             type_of_transform='Affine')

    rsref_reg = ants.apply_transforms(fixed=ref_t1,
                                      moving=mov_rs_ref,
                                      transformlist=transform_rsref_2_t1['fwdtransforms'])

    ants.image_write(rsref_reg, name_rs_reg)

    redTx = ants.read_transform(transform_rsref_2_t1['fwdtransforms'][0])
    ants.write_transform(redTx, name_transf_rs_2_t1)

    # grabamos también la transformación inversa
    name_transf_rs_2_t1_inv = name_rs_reg.split('.')[0] + '_inv-transform.mat'
    redTx_inv = ants.read_transform(transform_rsref_2_t1['invtransforms'][0])
    ants.write_transform(redTx_inv, name_transf_rs_2_t1_inv)

    return rsref_reg, transform_rsref_2_t1


# registro del T1 del sujeto al template. Registration of the subject's T1 to the template

@log_execution
def ants_registration_subject_T1_2_template(name_template, name_mov_t1, name_t1_reg):  # , name_transf_t1_2_template):

    ref_template = ants.image_read(name_template)
    mov_t1 = ants.image_read(name_mov_t1)

    # name_transf_t1_2_template =name_t1_reg.split('.')[0]+'_transform.mat' #dependiendo del tipo de registro hay que grabar la transformación de una forma o otra
    name_transf_t1_2_template = name_t1_reg.split('.')[0] + '_transform.nii.gz'

    transform_t1_2_template = ants.registration(fixed=ref_template,
                                                moving=mov_t1,
                                                type_of_transform='SyN')

    t1_reg = ants.apply_transforms(fixed=ref_template,
                                   moving=mov_t1,
                                   transformlist=transform_t1_2_template['fwdtransforms'])

    ants.image_write(t1_reg, name_t1_reg)

    tx_list = []
    for ntx, tx in enumerate(transform_t1_2_template['fwdtransforms']):
        if tx.split('.')[-1] == 'mat':
            redTx = ants.read_transform(tx)
            name_tx = name_t1_reg.split('.')[0] + '_transform' + str(ntx) + '.mat'
            ants.write_transform(redTx, name_tx)
            tx_list.append(name_tx)
        else:
            redTx = ants.image_read(tx)
            name_tx = name_t1_reg.split('.')[0] + '_transform' + str(ntx) + '.nii.gz'
            ants.image_write(redTx, name_tx)
            tx_list.append(name_tx)

    # también guardamos las transformadas inversas. To save the inverse tranforms also.
    tx_inv_list = []
    for ntx, tx in enumerate(transform_t1_2_template['invtransforms']):
        if tx.split('.')[-1] == 'mat':
            redTx_inv = ants.read_transform(tx)
            name_tx_inv = name_t1_reg.split('.')[0] + '_inv-transform' + str(ntx) + '.mat'
            ants.write_transform(redTx_inv, name_tx_inv)
            tx_inv_list.append(name_tx_inv)
        else:
            redTx_inv = ants.image_read(tx)
            name_tx_inv = name_t1_reg.split('.')[0] + '_inv-transform' + str(ntx) + '.nii.gz'
            ants.image_write(redTx_inv, name_tx_inv)
            tx_inv_list.append(name_tx_inv)

    # redTx = ants.read_transform( transform_t1_2_template['fwdtransforms'][0] )
    # ants.write_transform(redTx, name_transf_t1_2_template)

    return t1_reg, transform_t1_2_template, tx_list


# Aplicacion de las transformaciones anteriores para llevar el resting al espacio del template. To apply the transformations compute before to put the resting image in the template space.

@log_execution
def ants_transform_rs_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1, transform_t1_2_template,
                                 name_rs_reg_out):
    ref_template = ants.image_read(name_ref_template)
    # ref_template = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/Preprocessing_resting/test_subjects/derivatives/rs_preproc/sub-0022/ses-01/sub-0022_ses-01_task-rest_bold_desc-ref-reg-rs_seg-brain-bet.nii.gz'
    moving_rs = ants.image_read(name_moving_rs)

    # transforms_rs_2_template = transform_t1_2_template[ 'fwdtransforms'] + transform_rs_2_t1[ 'fwdtransforms']

    # transforms_rs_2_template = transform_rs_2_t1[ 'invtransforms']

    transforms_rs_2_template = transform_rs_2_t1['fwdtransforms'] + transform_t1_2_template['fwdtransforms']

    rs_reg = ants.apply_transforms(fixed=ref_template,
                                   moving=moving_rs,
                                   transformlist=transforms_rs_2_template,
                                   imagetype=3)
    # interpolator  = 'nearestNeighbor',
    # whichtoinvert = [True,True])

    ants.image_write(rs_reg, name_rs_reg_out)

    return rs_reg


# Aplicacion de las transformaciones anteriores para llevar el resting al espacio del template. To apply the transformations compute before to put the resting image in the template space. Interpolation for masks.

@log_execution
def ants_transform_mask_2_template(name_moving_rs, name_ref_template, transform_rs_2_t1, transform_t1_2_template,
                                   name_rs_reg_out):
    ref_template = ants.image_read(name_ref_template)
    # ref_template = '/mnt/D6E6B8EFE6B8D0CB/Raul_marte/Projects_2/Preprocessing_resting/test_subjects/derivatives/rs_preproc/sub-0022/ses-01/sub-0022_ses-01_task-rest_bold_desc-ref-reg-rs_seg-brain-bet.nii.gz'
    moving_rs = ants.image_read(name_moving_rs)

    # transforms_rs_2_template = transform_t1_2_template[ 'fwdtransforms'] + transform_rs_2_t1[ 'fwdtransforms']

    # transforms_rs_2_template = transform_rs_2_t1[ 'invtransforms']

    transforms_rs_2_template = transform_rs_2_t1['fwdtransforms'] + transform_t1_2_template['fwdtransforms']

    rs_reg = ants.apply_transforms(fixed=ref_template,
                                   moving=moving_rs,
                                   transformlist=transforms_rs_2_template,
                                   imagetype=0,
                                   interpolator='nearestNeighbor')
    # whichtoinvert = [True,True])

    ants.image_write(rs_reg, name_rs_reg_out)

    return rs_reg


# %%
# reproducimos registros usando flirt de fsl con fslpy. Registration but using FLIRT from FSL
# registro de la referencia del resting al T1 del sujeto. Registration of the resting reference to the subject's T1 space.
from fsl.wrappers import flirt, invxfm, concatxfm, applyxfm4D, applyxfm


@log_execution
def flirt_registration_rs_reference_2_subject_T1(name_ref_t1, name_mov_rs_ref, name_rs_reg, name_transf_rs_2_t1):
    mov = nib.load(name_mov_rs_ref)

    ref = nib.load(name_ref_t1)

    flirt(mov,
          ref,
          out=name_rs_reg.split('.')[0] + '_flirt.nii.gz',
          omat=name_transf_rs_2_t1.split('.')[0] + '_flirt.mat',
          cost='corratio',
          dof='6',
          interp='trilinear',
          verbose=True)

    # matriz invertida
    invxfm(name_transf_rs_2_t1.split('.')[0] + '_flirt.mat',
           name_transf_rs_2_t1.split('.')[0] + '_flirt_inverse.mat')

    return


# registro del T1 del sujeto al template. Registration of the T1 to the template.

@log_execution
def flirt_registration_subject_T1_2_template(name_template, name_mov_t1, name_t1_reg, name_transf_t1_2_template):
    mov = nib.load(name_mov_t1)

    ref = nib.load(name_template)

    flirt(mov,
          ref,
          out=name_t1_reg.split('.')[0] + '_flirt.nii.gz',
          omat=name_transf_t1_2_template.split('.')[0] + '_flirt.mat',
          cost='corratio',
          dof='12',
          interp='trilinear',
          verbose=True)

    # matriz invertida
    invxfm(name_transf_t1_2_template.split('.')[0] + '_flirt.mat',
           name_transf_t1_2_template.split('.')[0] + '_flirt_inverse.mat')

    return


# aplicamos las transformaciones al resting state. To apply the transformations to the resting image.

@log_execution
def flirt_transform_rs_2_template(name_moving_rs, name_ref_template, name_transform_rs_2_t1,
                                  name_transform_t1_2_template):  # , name_rs_reg_out):

    name_transform_rs_to_template = name_moving_rs.split('.')[0] + '_reg_rs_to_template_flirt.mat'

    path = '/'.join(name_transform_rs_to_template.split('/')[1:-1])
    path = '/' + path + '/'
    # primero concatenamos las transformaciones del rs a t1 y t1 a template
    atob = name_transform_rs_2_t1
    btoc = name_transform_t1_2_template
    # atoc = path+'MAT_0000.mat'
    atoc = name_transform_rs_to_template
    cmd = concatxfm(atob, btoc, atoc)
    # aplicamos la transformación al rs
    mov = nib.load(name_moving_rs)

    ref = nib.load(name_ref_template)

    cmd = applyxfm4D(mov,
                     ref,
                     out=name_moving_rs.split('.')[0] + '_desc-reg-rs-to-template_flirt.nii.gz',
                     mat=atoc,
                     singlematrix=True,  # importante el orden de parametros para que funcione
                     interp='trilinear')

    # matriz invertida
    invxfm(name_transform_rs_to_template,
           name_transform_rs_to_template('.')[0] + '_inverse.mat')

    return name_transform_rs_to_template


##
# aplicamos las transformaciones a la máscara. Apply tranformations to the mask

@log_execution
def flirt_transform_mask_2_template(name_moving_mask, name_ref_template, name_transform_rs_to_template,
                                    name_mask_reg_out):  # name_transform_rs_2_t1, name_transform_t1_2_template):#, name_rs_reg_out):

    # name_transform_rs_to_template = name_moving_rs.split('.')[0]+'_reg_rs_to_template_flirt.mat'

    # path = '/'.join(name_transform_rs_to_template.split('/')[1:-1])
    # path = '/'+path+'/'
    # # primero concatenamos las transformaciones del rs a t1 y t1 a template
    # atob = name_transform_rs_2_t1
    # btoc = name_transform_t1_2_template
    # # atoc = path+'MAT_0000.mat'
    # atoc = name_transform_rs_to_template
    # cmd = concatxfm(atob, btoc, atoc)
    # aplicamos la transformación al rs
    mov = nib.load(name_moving_mask)

    ref = nib.load(name_ref_template)

    cmd = applyxfm(mov,
                   ref,
                   out=name_mask_reg_out,
                   mat=name_transform_rs_to_template,
                   # singlematrix = True, #importante el orden de parametros para que funcione
                   interp='nearestneighbour')

    # matriz invertida
    # invxfm(name_transform_rs_to_template,
    #        name_transform_rs_to_template('.')[0]+'_inverse.mat')

    return


#################### fin funciones registro #########################

## funciones for nuissance #######
# %%
# ===========================================
# 4. NUISSANCE REGRESSION
# ===========================================

# 4.a SEGMENTATION OF ANATOMICAL SCAN

from fsl.wrappers import fast


@log_execution
def segmentation_fast_fslpy(seg_img_in, out):
    cmd = fast(seg_img_in, out,
               p=True,
               g=True,
               t=1)

    return


@log_execution
def segmentation_fast_nipype(seg_img_in, out):
    fast = fsl.FAST()

    fast.inputs.in_files = seg_img_in
    fast.inputs.img_type = 1
    fast.inputs.out_basename = out + '_nipype'
    fast.inputs.probability_maps = True
    fast.inputs.segments = True

    res = fast.run()

    return res


@log_execution
def ants_transform_T1_segmentations_2_rs_space(name_img_in, name_img_ref, name_transform_T1_2_rs, name_mask_reg):
    tx_t1_2_rs = ants.read_transform(name_transform_T1_2_rs)

    img_ref = ants.image_read(name_img_ref)
    img_in = ants.image_read(name_img_in)

    reg_im = ants.apply_transforms(fixed=img_ref,
                                   moving=img_in,
                                   # transformlist = tx_t1_2_rs, #falta poner [0]? ver como carga y trabaja con las transformaciones ants
                                   transformlist=name_transform_T1_2_rs,
                                   interpolator='nearestNeighbor')

    ants.image_write(reg_im, name_mask_reg)

    return reg_im


import scipy.ndimage as scpim


#@log_execution
def extract_average_signals_from_rs(rs_in, mask_brain, mask_csf, mask_wm):
    mask_brain_eroded = scpim.binary_erosion(mask_brain, iterations=1)
    mask_csf_eroded = scpim.binary_erosion(mask_csf, iterations=1)
    mask_wm_eroded = scpim.binary_erosion(mask_wm, iterations=1)

    n_time_points = rs_in.shape[3]
    signal_global = np.zeros((n_time_points))
    signal_csf = np.zeros((n_time_points))
    signal_wm = np.zeros((n_time_points))
    for ntp in range(n_time_points):
        rs_tp = rs_in[:, :, :, ntp]
        signal_global[ntp] = rs_tp[mask_brain_eroded == 1].mean()
        signal_csf[ntp] = rs_tp[mask_csf_eroded == 1].mean()
        signal_wm[ntp] = rs_tp[mask_wm_eroded == 1].mean()

    return signal_global, signal_csf, signal_wm

#@log_execution
def create_confounds_for_nilearn_clean_img(signal_global, signal_csf, signal_wm, transf_rp, path_rs_preproc):
    confound_vars = ['trans_x', 'trans_y', 'trans_z',
                     'rot_x', 'rot_y', 'rot_z',
                     'global_signal',
                     'csf', 'white_matter']

    confounds_df = pd.DataFrame()

    confounds_df[confound_vars[0]] = transf_rp[:, 3]
    confounds_df[confound_vars[1]] = transf_rp[:, 4]
    confounds_df[confound_vars[2]] = transf_rp[:, 5]

    confounds_df[confound_vars[3]] = transf_rp[:, 0]
    confounds_df[confound_vars[4]] = transf_rp[:, 1]
    confounds_df[confound_vars[5]] = transf_rp[:, 2]

    confounds_df[confound_vars[6]] = signal_global
    confounds_df[confound_vars[7]] = signal_csf
    confounds_df[confound_vars[8]] = signal_wm

    return confounds_df


# %%
# ===========================================
# 5. FUNCTIONS FOR ICA-AROMA
# ===========================================

# registro del T1 del sujeto al template usando FNIRT. Subject's T1 registration to the template.
from fsl.wrappers import fnirt


@log_execution
def fnirt_registration_subject_T1_2_template(name_template, name_mov_t1, name_aff_t1_2_template, name_t1_reg):
    # src = nib.load(name_mov_t1)

    # ref = nib.load(name_template)

    name_out = name_t1_reg.split('.')[0]

    fnirt(name_mov_t1,
          ref=name_template,
          aff=name_aff_t1_2_template,
          cout=name_out + '_coefficients_fnirt',
          iout=name_out + '_image_reg_fnirt.nii.gz',
          fout=name_out + '_warp_fnirt.nii.gz')

    return


from nipype.interfaces.fsl import ICA_AROMA


@log_execution
def apply_ica_aroma_to_rs(name_rs_in, name_aff_rs_2_t1, name_warp_t1_2_template, name_motion_parameters_mcflirt, tr,
                          dir_out):
    AROMA_obj = ICA_AROMA()
    AROMA_obj.inputs.in_file = name_rs_in  # 'functional.nii'
    AROMA_obj.inputs.mat_file = name_aff_rs_2_t1  # 'func_to_struct.mat'
    AROMA_obj.inputs.fnirt_warp_file = name_warp_t1_2_template
    AROMA_obj.inputs.motion_parameters = name_motion_parameters_mcflirt
    # AROMA_obj.inputs.mask = 'mask.nii.gz' # mascara pero consideramos que ya enviamos la serie rs enmascarada para solo cerebro
    AROMA_obj.inputs.denoise_type = 'nonaggr'  # 'both'
    # Type of denoising strategy:
    # -none: only classification, no denoising
    # -nonaggr (default): non-aggresssive denoising, i.e. partial component regression
    # -aggr: aggressive denoising, i.e. full component regression
    # -both: both aggressive and non-aggressive denoising (two outputs)
    AROMA_obj.inputs.out_dir = dir_out
    AROMA_obj.inputs.TR = tr
    AROMA_obj.cmdline

    # os.chdir('/home/biofisica/ICA-AROMA/ICA-AROMA')
    AROMA_obj.run()

    return


@log_execution
def temporal_filtering_after_aroma(dir_aroma, tr, lp, hp, name_temporal_aroma_cleaned_out):
    lista_archivos = os.listdir(dir_aroma)
    for file in lista_archivos:
        if file.endswith('nii.gz') and file.startswith('denoised'):
            aroma_file = file  # atencion si hay más de un denoised cogerá solo uno

    if aroma_file.endswith('nii.gz'):
        nii_aroma = nib.load(dir_aroma + aroma_file)
        data_nl_temporal_clean = nlimg.clean_img(nii_aroma,
                                                 detrend=True,
                                                 standardize='zscore',
                                                 t_r=tr,
                                                 # filter = 'butterworth',
                                                 low_pass=lp,
                                                 high_pass=hp,
                                                 ensure_finite=True)

        # name_temporal_aroma_cleaned_out = path_rs_preproc + name_rs_base+'_desc-ts-aroma-temporal-cleaned.nii.gz'
        nib.save(data_nl_temporal_clean, name_temporal_aroma_cleaned_out)

    else:
        print('There is not aroma denoised file')

    return

#####################################################################
### fin bloque funciones ############################################
#####################################################################

# %%
