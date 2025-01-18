
# [FlexfMRI](#overview)

## [Overview](#overview)
This repository contains all the scripts and tools used in the thesis project focusing on **functional connectivity (FC)** analysis in preclinical Alzheimer's Disease (AD) using data from the A4 study. The project involves structuring, preprocessing, reviewing, and analyzing large-scale neuroimaging data to uncover early alterations in brain connectivity associated with AD progression.

## [Repository Structure](#repository-structure)
The repository is organized into the following directories:

### [`Data_structuring`](./Data_structuring)  
Contains scripts like [`A4-Dataset_to_BIDS.ipynb`](./Data_structuring/A4-Dataset_to_BIDS.ipynb) for converting raw neuroimaging data into BIDS format for standardized data structuring.

### [`FC_analysis`](./FC_analysis)  
Includes notebooks ([`Exploratory_framework_FC.ipynb`](./FC_analysis/Exploratory_framework_FC.ipynb) and [`GLM_auto_FCmatrix.ipynb`](./FC_analysis/GLM_auto_FCmatrix.ipynb)) for conducting exploratory and statistical analyses of functional connectivity data.

### [`ICA_analysis`](./ICA_analysis)  
Contains scripts for ICA-based analysis of resting-state networks (RSNs). Example outputs can be found in the [`example_outputs`](./ICA_analysis/example_outputs) folder.

### [`PreprocessingScripts`](./PreprocessingScripts)  
Provides a main function and dependencies to apply a flexible preprocessing pipeline. Refer to the `config` file in the [`example_inputs`](./PreprocessingScripts/example_inputs) folder for detailed descriptions of the configurations and parameters needed.

### [`QC_scripts`](./QC_scripts)  
Includes a comprehensive Quality Control (QC) framework for visual and quantitative assessments of fMRI data. Key scripts include:
- [`generate_QC_reports.py`](./QC_scripts/generate_QC_reports.py): Generates QC reports for individual datasets.
- [`QC_reviewer_statistics.ipynb`](./QC_scripts/QC_reviewer_statistics.ipynb): Provides statistical summaries of QC evaluations.
- [`extract_motion_voxel_data.py`](./QC_scripts/extract_motion_voxel_data.py): Extracts motion-related voxel data for additional QC metrics.
- [`qc_api_main.py`](./QC_scripts/qc_api_main.py): Runs a Flask application to allow for QC revision of the previously created reports.

The folder also contains [`templates`](./QC_scripts/templates) for generating standardized QC reports and [`example_outputs`](./QC_scripts/example_outputs) showcasing report examples.

## [Contact](#contact)
For questions or feedback, contact Marc Biosca at [mbioscma7@alumnes.ub.edu](mailto:mbioscma7@alumnes.ub.edu).

## [Acknowledgment](#acknowledgment)
All scripts and notebooks were developed at the **Biomedical Imaging Group (BIG)** of the University of Barcelona, under the supervision of **PI: Roser Sala-Llonch**.

![BIG Logo](image.png)
