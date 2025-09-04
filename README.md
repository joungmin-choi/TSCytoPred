# TSCytoPred: A Deep Learning Framework for Inferring Cytokine Expression Trajectories from Irregular Longitudinal Gene Expression Data to Enhance Multi-Omics Analyses

## Requirements
* Python (>= 3.6)
* Pytorch (>= v1.6.0)
* Other python packages : numpy (>=1.19.1), pandas (>=1.1.1), os, sys, datetime

## Usage
Clone the repository or download source code files.

## Installation
To install TSCytoPred, run either:
```
pip install -r requirements.txt
```
or clone the repository and run:
```
pip install tscytopred
```

## Inputs
[Note!] All the example datasets can be found in './example/' directory.

#### Time-series/Longitudinal Gene expression profiles (Both training/testing)
* Contains gene expression profiles for each timepoint per patient
* Row : Timepoint (Sample), Column : Feature (Gene)
* The dataset should contain two columns named **"sample_id"** and **"timepoint"**, where, "sample_id" corresponds to the id of each patient, and "timepoint" should have timepoint information, where each timepoint needs to be denoted as the format of "%Y-%m-%d" (e.g., 2021-08-03).
* Dataset should be in sequential order of timepoint and the patients. For example, if per patient has three timepoints, then, it should be ordered in "patient01_timepoint_1,patient01_timepoint_2,patient01_timepoint_3,patient02_timepoint_1,patient02_timepoint_,...".
* File name should be "train_gene_expression.csv" and "test_gene_expression.csv"
* Example : ./example/train_gene_expression.csv

#### Time-series/Longitudinal Cytokine expression profiles (Training)
* Contains cytokine expression profiles for each timepoint per patient
* Row : Timepoint (Sample), Column : Feature (Cytokine)
* The dataset should contain two columns named **"sample_id"** and **"timepoint"**, same as gene expression file.
* Dataset should be in sequential order of timepoint and the patients, same as gene expression file.
* File name should be "train_cytokine_expression.csv"
* Example : ./example/train_cytokine_expression.csv

## How to run (Example)
1. Clone the respository, move to the cloned directory, and edit the **run_TSCytoPred.sh** to make sure each variable indicate the corresponding files.
2. Run the below command :
```
chmod +x run_TSCytoPred.sh
./run_TSCytoPred.sh
```
If you clone the directory and run the above command directly, you will get the result for the example dataset.

3. All the results will be saved in the newly created **results** directory.
   * pred_cytokine.csv : inferred cytokine expression values

## Contact
If you have any questions or problems, please contact to **joungmin AT vt.edu**.
