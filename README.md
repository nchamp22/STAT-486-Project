# STAT 486 Final Project

## House Prices: Comparing Reproducible Regression Pipelines

| Project detail | Information |
|---|---|
| **Course** | STAT 486, Fall 2026 |
| **Team** | Noah Champagne, Gunnar Griffith, and Chase Nielsen |
| **Proposal date** | October 27, 2026 |
| **Final repository due** | December 10, 2026, at 11:59 PM Mountain Time |
| **Repository** | [STAT-486-Project](https://github.com/nchamp22/STAT-486-Project) |
| **Competition** | [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) |
| **Selected route** | Route B: Kaggle competition |

## Executive summary

This project investigates whether a gradient-boosting pipeline with engineered housing features improves sale-price predictions over a random-forest pipeline in Kaggle's *House Prices: Advanced Regression Techniques* competition.

We are building two meaningfully different, reproducible machine-learning pipelines and evaluating them using the same local validation design and Kaggle's logged-price root mean squared error. We will compare local and Kaggle results, submit predictions from both pipelines, and examine the homes that produce the largest prediction errors.

The project emphasizes careful preprocessing, leakage prevention, reproducibility, and interpretation rather than leaderboard rank alone.

## Research question and purpose

### Primary research question

How much can a second, meaningfully different machine-learning pipeline improve house-price predictions over a random-forest pipeline in the Kaggle House Prices competition, and which types of houses produce the largest prediction errors?

### Motivation

Residential sale prices depend on a complex combination of numerical and categorical characteristics. This project allows our team to examine how missing-data treatment, categorical encoding, feature engineering, and model choice affect predictive performance on realistic tabular data.

Beyond comparing scores, we want to identify where each model succeeds or fails and determine whether improvements are consistent across different kinds of homes.

### Expected result

We expect gradient boosting with engineered housing features to outperform the random-forest pipeline because boosting can model complicated nonlinear relationships while correcting errors across successive trees.

### Unit of analysis

Each observation represents one residential property sale in Ames, Iowa.

## Project route

We selected **Route B: Kaggle competition**.

Our work includes:

- A simple reference prediction
- A random-forest pipeline
- A gradient-boosting pipeline with engineered features
- Consistent local validation
- At least two valid Kaggle submissions
- A comparison of local and public leaderboard results
- An investigation of important errors and limitations

The two main pipelines differ substantively in their model type and feature representation. The second pipeline is not simply a minor hyperparameter adjustment to the first.

## Data

### Source

The data come from Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) competition.

The competition data describe residential properties in Ames, Iowa. The training data contain the target variable `SalePrice`, while the test data omit the target and are used to create Kaggle submissions.

### Available files

The competition provides the following main files:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

The training dataset contains 1,460 observations and 81 columns, including the target variable. The test dataset contains 1,459 observations and 80 columns.

### Data access

Each team member can obtain the data by joining the Kaggle competition and downloading the competition files. The files should then be placed in:

```text
data/raw/
```

The raw competition files are excluded from Git because teammates can obtain identical copies directly from Kaggle.

### Data considerations

The dataset contains:

- Numerical and categorical predictors
- Missing values
- Variables where missingness represents the absence of a feature
- Skewed numerical variables
- Rare categorical levels
- Variables measured on different scales
- A positively skewed target variable

Missing values must be interpreted carefully. For example, a missing garage-related value may mean that the property has no garage rather than that the value was recorded incorrectly.

### Competition metric

Kaggle evaluates submissions using root mean squared error between the logarithms of predicted and observed sale prices:

$$
\operatorname{LogRMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\log(\widehat{y}_i)-\log(y_i)
\right]^2
}.
$$

Lower values indicate better predictive performance.

## Modeling plan

### Reference model

Our simplest reference predicts the mean training-set log sale price for every property.

This is not intended to be competitive. It provides a basic reference score that confirms the data, metric, prediction transformation, and submission format are working correctly.

### Pipeline 1: Random forest

The first full machine-learning pipeline uses a `RandomForestRegressor`.

Planned preprocessing includes:

- Median imputation for numerical predictors
- Most-frequent imputation for categorical predictors
- One-hot encoding for categorical predictors
- Log transformation of `SalePrice`
- A reproducible random seed

The random forest serves as a credible tree-based baseline that can capture nonlinear relationships and interactions without extensive feature engineering.

### Pipeline 2: Gradient boosting with engineered features

The second pipeline uses gradient boosting and additional housing features.

The substantive design changes include:

- Replacing random forest with gradient boosting
- Adding engineered aggregate features
- Preserving preprocessing within a reproducible pipeline
- Modeling the logarithm of `SalePrice`

Candidate engineered features include:

- Total finished square footage
- Total bathrooms
- Total porch area
- Total outdoor area
- Property age at sale
- Years since remodeling
- Total quality-related measures
- Indicators for garages, basements, fireplaces, and other amenities

This pipeline is meaningfully different because both the model-training procedure and feature representation change.

## Evaluation plan

### Local validation

All model comparisons should use the same local validation design whenever possible.

Our primary local evaluation will use a fixed training-validation split with `random_state=42`. The model will be trained only on the training portion, and performance will be measured on the held-out validation portion using log-RMSE.

Cross-validation may also be used during model development, but the test data will not be used for model selection.

### Leakage prevention

To reduce the risk of data leakage:

- The Kaggle test data will never be used for local model selection.
- Imputation and categorical encoding will be fitted only on training data.
- Feature engineering will use only information available for each property.
- Preprocessing steps will be included inside fitted pipelines when practical.
- `SalePrice` will be removed from the predictor matrix before training.
- The same validation observations will be used for comparable model evaluations.

### Kaggle comparison

Each main pipeline will generate its own submission file. We will compare:

- Local validation log-RMSE
- Kaggle public leaderboard score
- The direction and size of improvement between pipelines
- Any discrepancy between local and Kaggle performance

Leaderboard rank will be treated as context rather than the sole measure of project quality.

## Error analysis

We will examine the largest absolute log-price errors from the validation data.

Potential error patterns include:

- Very expensive homes
- Very inexpensive homes
- Older homes
- Recently remodeled homes
- Homes with unusual living areas
- Homes with missing amenities
- Rare neighborhood or zoning categories
- Properties with unusual combinations of quality and size

This analysis will help us determine whether a better overall score reflects consistent improvement or improvement concentrated among particular properties.

## Reproducibility

### Current repository structure

```text
STAT-486-Project/
├── README.md
├── .gitignore
├── data/
│   └── raw/
│       └── .gitkeep
├── outputs/
│   └── .gitkeep
├── data_checkpoint.ipynb
└── working_investigation.ipynb
```

Raw Kaggle data and generated submission files are excluded from Git when appropriate.

### General setup

1. Clone the repository.
2. Join the Kaggle competition.
3. Download the competition files.
4. Place the files in `data/raw/`.
5. Install the required Python packages.
6. Run the notebooks from top to bottom.

Example:

```bash
git clone https://github.com/nchamp22/STAT-486-Project.git
cd STAT-486-Project
```

The notebooks currently use common Python packages including:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Team workflow

The team will share responsibility for:

- Data preparation
- Pipeline development
- Validation
- Kaggle submissions
- Error analysis
- Documentation
- Presentation preparation

Before pushing changes, team members should pull the most recent version of the repository. Code and notebook changes should be accompanied by clear commit messages.

## Risks and limitations

Potential challenges include:

- Missing-value handling
- High-cardinality categorical features
- Overfitting during feature engineering or tuning
- Differences between local and Kaggle scores
- Inconsistent software environments
- Limited time for extensive hyperparameter searches
- Correlated housing features
- Difficulty interpreting complex ensemble models

To manage these risks, we will use fixed seeds, consistent validation, simple reproducible preprocessing, and documented run procedures.

## Project milestones

| Date | Milestone | Status |
|---|---|---|
| October 27, 2026 | Project proposal | Complete |
| November 5, 2026 | Data and reference result | Complete |
| November 12, 2026 | First full pipeline and Kaggle submission | Complete |
| November 19, 2026 | Evidence and interpretation | Complete |
| Late November 2026 | Error analysis and model refinement | In progress |
| December 8 or 10, 2026 | Class presentation | Upcoming |
| December 10, 2026 | Final repository submission | Upcoming |

## Sources

- [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)
- [House Prices competition data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- [House Prices evaluation metric](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation)
- [Scikit-learn documentation](https://scikit-learn.org/stable/)

---

# Checkpoint: Data and Reference Result

**Date:** November 5, 2026  
**Team:** Noah Champagne, Gunnar Griffith, and Chase Nielsen  
**Route:** B — Kaggle competition

## Data

The data come from Kaggle's *House Prices: Advanced Regression Techniques* competition. Each team member can obtain the same files directly from the competition page.

The available files are:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

The training data contain 1,460 observations and 81 columns, including `SalePrice`. The test data contain 1,459 observations and 80 columns.

The raw data are stored locally in `data/raw/` and excluded from Git. The tracked `.gitkeep` file preserves the directory structure for teammates.

No additional data need to be collected. Future cleaning will include missing-value treatment, categorical encoding, and selected feature engineering.

## Evaluation plan

We use log-RMSE because it matches the Kaggle competition metric and reduces the disproportionate influence of very expensive homes.

For the initial result, we used a fixed 80/20 training-validation split with `random_state=42`.

The split produced:

- 1,168 training observations
- 292 validation observations

The mean logged sale price was calculated using the training portion only. This avoids using validation outcomes when constructing the reference prediction.

## First runnable result

The first runnable result is available in [`data_checkpoint.ipynb`](data_checkpoint.ipynb).

The reference model predicts the same value—the mean training-set log sale price—for every property.

| Model | Validation log-RMSE | Training rows | Validation rows |
|---|---:|---:|---:|
| Mean log-price reference | 0.410480 | 1,168 | 292 |

The notebook also generated a valid Kaggle submission file with the required columns:

- `Id`
- `SalePrice`

The first predictions were approximately `$166,716.73` for every test observation.

### Interpretation

The local validation score of `0.410480` is intentionally weak because the reference model ignores all housing characteristics. However, it confirms that:

- The data load correctly
- The target transformation is correct
- The metric works
- The submission has the required format
- Predictions can be converted back to dollar values

### Next step

The next step is to fit the first full machine-learning pipeline using preprocessing and a random forest. Its performance will be compared against the reference model using the same validation design.

---

# Checkpoint: First Pipeline and Kaggle Submission

**Date:** November 12, 2026  
**Team:** Noah Champagne, Gunnar Griffith, and Chase Nielsen  
**Route:** B — Kaggle competition

## Pipeline 1: Random forest

The first full pipeline uses a random-forest regressor with preprocessing for both numerical and categorical variables.

The pipeline includes:

- Median imputation for numerical predictors
- Most-frequent imputation for categorical predictors
- One-hot encoding for categorical predictors
- A random-forest regression model
- Training on the logarithm of `SalePrice`
- Conversion of predictions back to dollar values for submission

The code and output are available in [`working_investigation.ipynb`](working_investigation.ipynb).

## First Kaggle result

The first valid random-forest submission received a Kaggle score of:

```text
0.14381
```

A later random-forest submission improved the score to:

```text
0.13960
```

Both scores substantially improve on the constant reference submission score of:

```text
0.41637
```

### Interpretation

The large improvement over the reference model shows that housing characteristics contain substantial information about sale prices.

The random forest captures nonlinear relationships and interactions that the constant reference cannot represent. The improved random-forest submission also demonstrates that preprocessing and model configuration affect prediction quality.

### Next step

The next step is to build a meaningfully different second pipeline using gradient boosting and engineered housing features. We will compare it with the random forest using both local validation and Kaggle results.

---

# Checkpoint: Evidence and Interpretation

**Date:** November 19, 2026  
**Team:** Noah Champagne, Gunnar Griffith, and Chase Nielsen  
**Route:** B — Kaggle competition

## Evidence so far

We compared the reference submission, the random-forest pipeline, and the gradient-boosting pipeline using Kaggle's log-RMSE metric. Lower scores are better.

| Pipeline | Substantive design | Kaggle log-RMSE |
|---|---|---:|
| Mean log-price reference | Constant prediction for every property | 0.41637 |
| Random forest | Imputation, one-hot encoding, and random-forest regression | 0.13960 |
| Gradient boosting | Gradient boosting with engineered housing features | **0.13263** |

The gradient-boosting pipeline improved the Kaggle score from `0.13960` to `0.13263`.

The absolute improvement was:

$$
0.13960-0.13263=0.00697.
$$

The relative reduction in Kaggle error was approximately:

$$
\frac{0.13960-0.13263}{0.13960}\times100
\approx 5.0\%.
$$

The code, model construction, output, and submission generation are available in [`working_investigation.ipynb`](working_investigation.ipynb).

The Kaggle competition and submission results are available from the [House Prices competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

## Pipeline comparison

The second pipeline made two substantive changes:

1. It replaced the random-forest model with gradient boosting.
2. It added engineered housing features that summarize property size, age, bathrooms, outdoor space, and amenities.

The gradient-boosting submission received a Kaggle score of `0.13263`, compared with the best random-forest score of `0.13960`. This is evidence that the second pipeline improved predictive performance.

However, the model type and feature representation changed at the same time. Therefore, the current results do not show how much of the improvement came from gradient boosting and how much came from feature engineering.

## Interpretation

The results changed our thinking because the random forest was already much stronger than the constant reference, but the second pipeline still produced a meaningful improvement.

This suggests that model choice and feature representation both matter for these data. It also shows that improvements beyond a reasonable tree-based baseline are possible without relying only on extensive hyperparameter tuning.

The gradient-boosting score is currently the team's strongest Kaggle result.

## Informative limitation

The main limitation is that the current comparison changes both the model and the features simultaneously.

Because of this, we cannot yet make a clean causal claim that gradient boosting alone produced the entire improvement. The engineered features may account for part of the score reduction.

Another limitation is that the public Kaggle score represents performance on only part of the hidden test data. A lower public score does not guarantee the same improvement on the private leaderboard.

## Next steps

Our next steps are to:

1. Compare random forest and gradient boosting using the same feature set.
2. Compare models with and without engineered features.
3. Record local validation results beside the Kaggle scores.
4. Investigate the largest validation errors.
5. Check whether expensive, unusual, or rare homes produce larger errors.
6. Confirm that both pipelines can be rerun from a clean environment.
7. Improve notebook labels, comments, and run instructions.
8. Prepare final tables and figures for the report and presentation.

These comparisons will help separate the effect of model choice from the effect of feature engineering and strengthen the conclusions we can draw from the project.