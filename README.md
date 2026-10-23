# Error-challenge testing

This repository contains the analysis code associated with the study on error testing of the secondary-radiation monitoring system in ocular proton therapy, within the context of passive-scattering proton therapy.

## Repository structure


```text
proton-dose-calibration/
├── README.md
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── LICENSE
├── data/
│   └── input/
│       ├── all_data_train.csv
│       ├── all_data_test.csv
│       ├── input_complet_test.csv
│       ├── input_partial_test.csv
│       └── campaign_deliveries.csv
├── code/
│   ├── models.ipynb
│   └── figure1_and_tables.py
└── results/
    └── figures_and_tables/
        ├── Figure_S_1.pdf
        ├── Figure_1_a.pdf
        ├── Figure_1_a.png
        ├── Figure_1_b.pdf
        ├── Figure_1_b.png
        ├── Figure_1_c.pdf
        ├── Figure_1_c.png
	├── test_metrics.csv
        └── campaign_detection.csv
```

## Manuscript and archived release


This repository contains the analysis codes, computational resources, and clinical input dataset associated with the manuscript currently under peer review.

The version corresponding to the current archived release is:

- GitHub release: `v1.0.0`
- Zenodo DOI: `10.5281/zenodo.21947824`

The archived version can be cited using the DOI above.

## Analysis workflow

The repository contains two complementary analysis codes:

- `code/models.ipynb` implements the complete machine-learning evaluation and model-selection procedure.
- `code/figure1_and_tables.py` reproduces the final manuscript figures and supplementary tables using the retained model configurations and the analytical and RBF reference models.

### 1. Model evaluation and selection — `models.ipynb`

The selection of the models and their parameters, model training, evaluation, and testing on new data are described and implemented in:

```text
code/models.ipynb
```

The notebook performs the complete machine-learning evaluation and selection procedure. The workflow includes:

1. Loading and preprocessing of the training and independent test datasets.
2. Construction of derived variables, including the charge-per-MU quantity.
3. Definition and fitting of the analytical inverse power-law model.
4. Definition and fitting of the radial basis function (RBF) interpolator.
5. Definition of the machine-learning feature combinations.
6. Definition of data transformations and preprocessing operations.
7. Definition of Random Forest, Extra Trees, and XGBoost regression models.
8. Definition of the hyperparameter grids.
9. Five-fold cross-validation.
10. Grid-search-based hyperparameter optimization.
11. Evaluation using relative-error metrics.
12. Comparison of the different feature, transformation, and model configurations.
13. Selection of the machine-learning configurations retained for the final comparative analysis.
14. Final retraining of the selected machine-learning models.
15. Evaluation on independent test data.
16. Comparative analysis of the analytical, RBF, and selected machine-learning approaches.

The relative-error metrics include:

- relative MAE;
- relative RMSE;
- relative standard deviation.

Bootstrap procedures are also implemented for the estimation of 95% confidence intervals for selected relative-error metrics.

### 2. Figure and supplementary tables — `figure1_and_tables.py`

The Python script:

```text
code/figure1_and_tables.py
```

provides a standalone reproduction of the main figure and supplementary tables using the model configurations retained by the notebook.

The script:

- trains the seven retained machine-learning configurations on the 6960 training sessions;
- fits the analytical and RBF reference models;
- evaluates all models on the three independent test datasets;
- generates Supplementary Tables S5 and S12;
- generates Figure 1, including its three panels;
- evaluates the models on the two error-challenge campaigns and generates Supplementary Table S9;
- saves the corresponding numerical outputs and Figure 1 panels.

The seven machine-learning configurations and their hyperparameters are explicitly defined in the script and correspond to the configurations retained by the grid-search procedure implemented in `models.ipynb`. The script is therefore intended for reproducibility of the reported results rather than for repeating the model-selection grid search.

Figure 1 is generated as three separate files:

```text
results/figures_and_tables/Figure_1_a.pdf
results/figures_and_tables/Figure_1_a.png
results/figures_and_tables/Figure_1_b.pdf
results/figures_and_tables/Figure_1_b.png
results/figures_and_tables/Figure_1_c.pdf
results/figures_and_tables/Figure_1_c.png
```

The first two panels show the analytical relationship and its inverse representation. The third panel provides the comparative distribution of relative prediction errors for the analytical, RBF, and selected machine-learning approaches.

The script also produces:

```text
results/figures_and_tables/test_metrics.csv
results/figures_and_tables/campaign_detection.csv
```

`test_metrics.csv` contains the model-performance metrics for the independent test datasets and corresponds to the values reported in Supplementary Tables S5 and S12.

`campaign_detection.csv` contains the results of the error-challenge campaigns used to generate Supplementary Table S9.

To run the script:

```bash
python ./code/figure1_and_tables.py DATA_DIR CAMPAIGN_CSV OUT_DIR
```

where:

- `DATA_DIR` is the directory containing the training and independent test datasets (`data/input/`);
- `CAMPAIGN_CSV` is the input CSV file containing the error-challenge campaign deliveries (`data/input/campaign_deliveries.csv`);
- `OUT_DIR` is the directory where the generated figures and tables are saved (`results/figures_and_tables/campaign_deliveries.csv`).


The script uses the same analytical model, RBF approach, retained machine-learning configurations, test datasets, and relative-error definitions as the analysis described in the notebook, so that the manuscript figures and tables can be reproduced from the archived analysis code.

## Input data

The analysis uses separate training and independent test datasets.

The expected input directory is:

```text
data/input/
```

The input files are:

```text
all_data_train.csv
all_data_test.csv
input_complet_test.csv
input_partial_test.csv
campaign_deliveries.csv
```

### Training dataset

```text
all_data_train.csv
```

This dataset is used for:

- model fitting;
- cross-validation;
- hyperparameter optimization;
- selection of the machine-learning configurations;
- fitting of the analytical model;
- fitting of the RBF interpolator.

### Independent test datasets

Three independent test datasets are defined:

```text
all_data_test.csv
input_complet_test.csv
input_partial_test.csv
```

They correspond to the different test subsets used in the analysis.

The final comparative workflow explicitly defines these datasets through the `TEST_FILES` configuration:

```python
TEST_FILES = {
    "all": "all_data_test.csv",
    "complete": "input_complet_test.csv",
    "interrupted": "input_partial_test.csv"
}
```

The input datasets are not generated by the analysis codes. An authorized copy of the required datasets must therefore be available in `data/input/` before running the analysis.

## Machine-learning models

Three supervised machine-learning regression algorithms are considered:

- Random Forest (`RF`);
- Extra Trees (`ET`);
- XGBoost (`XGB`).

Several feature combinations are evaluated, including combinations based on:

- charge (`Qγ`);
- delivered monitor units (`MU`);
- charge per monitor unit (`Qγ/MU`).

The machine-learning configurations combine these feature sets with different preprocessing operations.

## Data transformations

The notebook evaluates several preprocessing and feature-transformation strategies, including:

- raw features;
- standardization using `StandardScaler`;
- min-max scaling using `MinMaxScaler`;
- normalization using `Normalizer`;
- polynomial feature expansion;
- polynomial features combined with standardization;
- polynomial features combined with min-max scaling;
- polynomial features combined with normalization.

Polynomial transformations use degree 4 in the current analysis.

## Hyperparameter optimization

For each evaluated machine-learning configuration, hyperparameters are optimized using `GridSearchCV`.

The optimization is performed using five-fold cross-validation with a fixed random seed.

The final selected machine-learning configurations are then retrained using the complete training dataset before evaluation on the independent test data.

## Analytical model

An inverse power-law model is used as an analytical reference model.

The relationship is based on the inverse form of:

```text
Qg/D = a (D/MU)^b
```

and is fitted by minimizing the mean relative error on the predicted dose per monitor unit.

The fitted analytical model is subsequently included in the final comparative analysis.

## RBF model

A radial basis function (RBF) interpolator is also used as a non-parametric reference approach.

The RBF model is fitted using:

- charge (`Qγ`);
- delivered monitor units (`MU`);
- delivered dose (`D`).

The resulting predictions are included in the comparative analysis with the analytical and machine-learning approaches.

## Evaluation procedure

The notebook evaluates model performance using relative prediction errors.

For a predicted value `y_pred` and a reference value `y_true`, the relative error is expressed as:

```text
100 × (y_pred - y_true) / y_true
```

The main evaluation metrics are:

- relative MAE;
- relative RMSE;
- relative standard deviation.

The model-selection stage uses cross-validation on the training dataset, while the final comparison is performed on independent test data.

## Final comparative analysis

After the machine-learning evaluation and model-selection procedure, the retained machine-learning configurations are compared with the analytical and RBF approaches.

The final comparison includes the relative prediction-error distributions for:

- analytical model;
- RBF interpolator;
- selected Random Forest configurations;
- selected Extra Trees configurations;
- selected XGBoost configurations.

The final boxplot provides a common visual comparison of the prediction-error distributions obtained with the different approaches.

## Outputs

The analysis produces graphical and tabular results in:

```text
results/figures/
```

The main Figure 1 outputs are:

```text
Figure_1_a.pdf
Figure_1_a.png
Figure_1_b.pdf
Figure_1_b.png
Figure_1_c.pdf
Figure_1_c.png
```

The first two panels illustrate the analytical relationship and its inverse representation, while the final panel provides the comparative distribution of relative prediction errors for the analytical, RBF, and selected machine-learning approaches.

When using `figure1_and_tables.py`, the following additional outputs are generated:

```text
test_metrics.csv
campaign_detection.csv
```

The figures and tables can be regenerated by running `figure1_and_tables.py` with the required input datasets and error-challenge campaign file.

## Software requirements

The Python dependencies required to run the analysis are listed in:

```text
requirements.txt
```

Install them using:

```bash
python -m pip install -r requirements.txt
```

The analysis relies on, among others:

- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- XGBoost;
- Matplotlib;
- Plotly.

## Reproducing the analysis

1. Clone or download this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Make the required input datasets available in the input data directory.
4. Open:

   ```text
   code/models.ipynb
   ```

5. Run the notebook cells in order to reproduce the model evaluation, hyperparameter selection, final model training, and independent-test evaluation.

The first part of the notebook performs the complete machine-learning evaluation and selection procedure. The final part combines the retained machine-learning configurations with the analytical and RBF models for the final comparative analysis.

For a direct reproduction of Figure 1 and Supplementary Tables S5, S9, and S12 using the retained configurations, run:

```bash
python code/figure1_and_tables.py DATA_DIR CAMPAIGN_CSV OUT_DIR
```

This script does not repeat the grid-search model-selection procedure. Instead, it uses the configurations and hyperparameters retained by `models.ipynb` and reproduces the reported final results.

## Data availability

The clinical input datasets used in this study are not publicly available. Access to the clinical datasets requires a request to the authors.

The datasets required to reproduce the analysis, including `campaign_deliveries.csv`, are not publicly available and must be requested from the authors. Once access has been granted, the datasets should be placed in the appropriate input directory and `campaign_deliveries.csv` should be provided as the `CAMPAIGN_CSV` input when running `figure1_and_tables.py`.

## Citation

For the archived software release, please cite:

Ndong, M., Leni, P.-E., Hofverberg, P., Schiappa, R.,  
Gschwind, R., and Maneval, D. (2026).  
*Prompt-gamma flux monitoring for independent in vivo verification  
of the dose calibration coefficient in passive-scattering proton  
therapy: a machine learning study on 1781 patients.*  
Zenodo. https://doi.org/10.5281/zenodo.21947824

The citation metadata are also provided in `CITATION.cff`.

The associated manuscript is currently under peer review.
