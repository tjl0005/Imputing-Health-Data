# Improving Predictive Performance of Mortality through Imputation

## Summary

This code is used to pre-process the MIMIC-IV dataset (Halevy et al., 2009; Henry et al., 2016; Johnson et al., 2016) to
be used for evaluating different imputation techniques. After pre-processing the data can be imputed via mean, median,
MICE and kNN in the current code. For GAIN and MIWAE the code is executed as a notebook due to the long training times.

Each of the imputation methods can be optimised under two main categories of missing data, artificially missing and real
missing. For artificially missing the imputations are evaluated through MAE and their downstream performance in the
prediction of mortality. For the real missing data only downstream performance is directly evaluated.

The findings of the evaluations are presented as visualisations.

## The Data

The dataset is referred to as MIMIC-IV, which is available from:  
https://physionet.org/content/mimiciv/1.0/ (Halevy et al., 2009; Henry et al., 2016; Johnson et al., 2016)

This data is required for the code to run, specifically the admissions and patients data from the "HOSP" module and
chart events, item directory and icu stays from the "ICU" module.

## File Structure

The code is broken down into four segments: preprocessing, create_missing, imputation, evaluation and scoring.

### Preprocessing

THe folder is used for initial exploration of the data such as the mortality rates, distribution of features and the
missing rates for APACHE II variables. It limits the patient cohorts to valid stays only and links them with their
administrative data, making a dataset reading for imputation testing.

### Create_missing

#### Enforce Missingness

This refers to creating different versions of the dataset with different levels of missingness. Those implemented by
default are 2 values, 5 values and 10 values, missing per patient, but these can be changed. This results in 3 different
versions of the data with increasingly incomplete data. A missingness level of 0 is also used, acting as "ground truth"
data, and required for artificial injection of missing data.

#### Randomise Missing

This refers to artificially injecting missing data into the "ground truth" dataset created by enforce_missingness.py.
Missingness is injected under MCAR and MNAR, with MNAR being achieved through central data missing and a mixture of
extreme and central data missing. This is done to match assumptions of missing data in healthcare
(Kazijevs et al., 2023; Steif et al., 2024). Again this is done at different levels of missingness under each
mechanism, specifically 20%, 50% and 70% missing per feature overall.

### Imputation

This contains the simple (Mean and Median) and Machine Learning (knn and MICE) imputation methods. Using the
sklearn implementations. The functions can be called with the missing data and relevant hyperparameters and then the
imputations saved.

### Evaluation

This contains the evaluation methods for the imputations, specifically through optimisation of each of the imputation
methods. Ground truth optimisation is used for optimising the different imputers through MAE on how well they recreate
the
missing data when compared to the ground truth.

Raw optimisation is used for optimising the different imputers through ROC-AUC and F1 through their optimised XGBoost
models when predicting mortality. The imputers are optimised to maximise the ROC-AUC and F1 through their
hyperparameters and those of XGBoost.

### Scoring

Measurement processing is used to convert the chart times data into a usable format with the patient stays in the ICU.
It is set up to retain only the worst readings within a given timeframe and to extract the hourly data within given
intervals and timeframe.

The apache.py file is used to convert the measurements into their relevant APACHE scores (Knaus et al., 1985;
Knaus et al., 1991; Zimmerman et al., 2006). Although it is not used in the final version of the project as the raw
measurements were found to be more insightful. Feature selection is used to identify further measurements which may be
of use. Although it is not used in the final version either due to time constraints.

## Notebooks

These are written for the GAIN (Dong et al., 2018) and MIWAE (Mattei et al., 2018) deep learning implementations,
specifically to work with the data produced
here. They contain the full code for setting up the models, evaluation through ground truth and downstream performance.
The data from them is also included in this repository and used when visualising results by default.

## Results

Not yet final

# References

Dong, W., Fong, D. Y. T., Yoon, J., Wan, E. Y. F., Bedford, L. E., Tang, E. H. M., & Lo Kuen Lam, C. (2021). Generative
adversarial networks for imputing missing data for big data clinical research. BMC Medical Research Methodology, 21(
1). https://doi.org/10.1186/s12874-021-01272-3
Henry, J., Pylypchuk, Y., Searcy T. & Patel V. (May 2016). Adoption of Electronic Health Record Systems among U.S.
Non-Federal Acute Care Hospitals: 2008-2015. ONC Data Brief, no.35. Office of the National Coordinator for Health
Information Technology: Washington DC.+
Halevy, A., Norvig, P., & Pereira, F. (2009). The unreasonable effectiveness of data. IEEE Intelligent Systems, 24(2),
8-12.
Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L.H., Feng, M., Ghassemi, M., ... & Mark, R. G. (2016). MIMIC-III, a
freely accessible critical care database. Scientific data, 3(1), 1-9.
Kazijevs, M., & Samad, M. D. (2023). Deep imputation of missing values in time series health data: A review with
benchmarking. Journal of Biomedical Informatics, 144, 104440. https://doi.org/10.1016/j.jbi.2023.104440
Knaus, W. A., Draper, E. A., Wagner, D. P., & Zimmerman, J. E. (1985). APACHE II: a severity of disease classification
system. Critical care medicine, 13(10), 818–829.
Knaus, W. A., Wagner, D. P., Draper, E. A., Zimmerman, J. E., Bergner, M., Bastos, P. G., Sirio, C. A., Murphy, D. J.,
Lotring, T., Damiano, A., & Harrell, F. E. (1991). The APACHE III Prognostic System. CHEST Journal, 100(6),
1619–1636. https://doi.org/10.1378/chest.100.6.1619
Mattei, P., & Frellsen, J. (2018). MIWAE: Deep Generative Modelling and Imputation of Incomplete Data. arXiv (Cornell
University). https://doi.org/10.48550/arxiv.1812.02633
Steif, J., Brant, R., Sreepada, R. S., West, N., Murthy, S., & Görges, M. (2021). Prediction model performance with
different imputation strategies: A simulation study using a North American ICU registry. Pediatric Critical Care
Medicine, 23(1), e29–e44. https://doi.org/10.1097/pcc.0000000000002835 
