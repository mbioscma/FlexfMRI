# FlexfMRI

## Overview
This repository contains all the scripts and tools used in the thesis project focusing on **functional connectivity (FC)** analysis in preclinical Alzheimer's Disease (AD) using data from the A4 study. The project involves structuring, preprocessing, reviewing, and analyzing large-scale neuroimaging data to uncover early alterations in brain connectivity associated with AD progression.

## Repository Structure
The repository is organized into the following directories:

- **`Data_structuring`**:  
  Contains scripts like `A4-Dataset_to_BIDS.ipynb` for converting raw neuroimaging data into BIDS format for standardized data structuring.
  
- **`FC_analysis`**:  
  Includes notebooks (`Exploratory_framework_FC.ipynb` and `GLM_auto_FCmatrix.ipynb`) for conducting exploratory and statistical analyses of functional connectivity data.

- **`ICA_analysis`**:  
  Contains scripts for ICA-based analysis of resting-state networks (RSNs).

- **`PreprocessingScripts`**:  
  Provides a main function and dependencies to apply a flexible preprocessing pipeline. Refer to the `config` file in the `example_inputs` folder for detailed descriptions of the configurations and parameters needed.

- **`QC_scripts`**:  
  Includes a comprehensive Quality Control (QC) framework for visual and quantitative assessments of fMRI data. Key scripts include:
  - `generate_QC_reports.py`: Generates QC reports for individual datasets.
  - `QC_reviewer_statistics.ipynb`: Provides statistical summaries of QC evaluations.
  - `extract_motion_voxel_data.py`: Extracts motion-related voxel data for additional QC metrics.
  - `qc_api_main.py`: Runs a Flask application to allow for QC revision of the previously created reports.
  The folder also contains templates for generating standardized QC reports and `example_outputs` showcasing report examples.

## Contact
For questions or feedback, contact Marc Biosca at mbioscma7@alumnes.ub.edu.

## Acknowledgment
All scripts and notebooks were developed at the **Biomedical Imaging Group (BIG)** of the University of Barcelona, under the supervision of **PI: Roser Sala-Llonch**.
