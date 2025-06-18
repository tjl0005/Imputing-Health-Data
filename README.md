# Improving Predictive Performance of Mortality through Imputation
## Summary

This code is used to pre-process the MIMIC-IV dataset to be used for evaluating different imputation techniques.  
NOTE: This README is not complete as the project is ongoing but details of what has been acheived are available in the docstrings of each file.  
The next steps are to implement optimisation of the imputation methods, final evaluation of the imputation using ground truth data and then comparison of performances when using different levels imputed data with a prediction model.

## File Structure

Currently the code is broken down into four segments: preprocessing, imputation, evaluation and scoring.

## The Data

The dataset is used is reffered to as MIMIC-IV, which is available from:  
(insert url)  
(insert references)  
This data is required for the code to run, specifically the admissions and patients data from the "HOSP" module and chart events, item directory and icu stays from the "ICU" module.
