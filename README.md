# STAT 486 Final Project Proposal

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

This project will investigate whether a gradient-boosting pipeline with engineered housing features improves sale-price predictions over a random-forest pipeline in Kaggle's *House Prices: Advanced Regression Techniques* competition. We will construct two meaningfully different, reproducible machine-learning pipelines; evaluate them using the same local validation design and Kaggle's logged-price root mean squared error; submit predictions from both pipelines; and analyze the types of homes that produce the largest errors. The project emphasizes careful preprocessing, leakage prevention, reproducibility, and interpretation rather than leaderboard rank alone.

## Research question and purpose

### Primary research question

How much can a second, meaningfully different machine-learning pipeline improve house-price predictions over a random-forest pipeline in the Kaggle House Prices competition, and which types of houses produce the largest prediction errors?

### Motivation

Residential sale prices reflect a complex combination of numerical and categorical characteristics. This project gives our team an opportunity to examine how missing-data treatment, categorical encoding, feature engineering, and model choice affect predictive performance on realistic tabular data. Beyond comparing scores, we want to identify where each model succeeds or fails and determine whether improvements are consistent across different price ranges and property types.

### Initial expectation

We expect a gradient-boosting pipeline with engineered features to achieve lower local validation error and a lower Kaggle score than the random-forest pipeline. The evidence may instead show that the improvement is small, unstable across validation folds, or concentrated among particular price ranges or types of homes. A weaker second pipeline would still provide useful evidence about the value of feature engineering and model choice.

### Unit of observation

Each observation represents one residential property sold in Ames, Iowa.

## Chosen project route

- [ ] **Route A:** Develop and test something new in an ML method or system
- [x] **Route B:** Enter a Kaggle competition with two meaningfully different pipelines
- [ ] **Route C:** Build a dataset and use it in a reproducible ML pipeline

Route B fits our question because it provides a clearly defined regression task, a fixed evaluation metric, official training and test data, and an external leaderboard score. The dataset is also small enough to process on our personal computers before the December 10 deadline. Instructor approval of the competition and the two proposed pipeline designs is pending.

## Data source and access

### Competition data

We will use the data from Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) competition. The dataset contains 79 explanatory variables describing residential properties in Ames, Iowa. Relevant fields include lot size, neighborhood, overall quality, year built, above-ground living area, basement characteristics, garage characteristics, and room counts. The response variable is `SalePrice`, and `Id` identifies each property.

The official files are:

- `train.csv`: labeled training observations
- `test.csv`: unlabeled observations for Kaggle inference
- `data_description.txt`: variable definitions
- `sample_submission.csv`: required submission structure

Each team member will join the competition, accept its rules, and download the official files. The data and computational requirements are feasible on personal computers. Raw competition data will not be committed to Git; instead, the README will explain how authorized users can obtain and place the files locally.

### Rules and responsible use

We will follow the Kaggle competition rules and the course requirements. Private collaboration will remain within the registered team. We will not redistribute restricted competition data or use the data for commercial purposes. External notebooks, tutorials, code, or modeling ideas used in the project will be cited clearly. We will remain within Kaggle's submission limit.

### Anticipated data-quality issues

The dataset includes:

- Missing values
- Numerical and categorical predictors
- Skewed distributions
- Rare categorical levels
- Potential outliers
- Highly correlated predictors
- Missing values that indicate the absence of a feature rather than an unknown measurement

For example, a missing garage or basement field may mean that the property does not have that feature. Our preprocessing will distinguish structural absence from ordinary missingness where the data documentation supports that interpretation. Every learned preprocessing step will be fitted only on the appropriate training data to prevent leakage.

## Modeling plan

### Naive reference model

Our simplest credible reference will predict the median training-set log sale price for every observation and then transform that value back to dollars. This creates a reproducible benchmark that uses no housing characteristics. Both machine-learning pipelines should substantially outperform this reference.

### Pipeline 1: Random-forest pipeline

Pipeline 1 will serve as the primary machine-learning baseline:

1. Load the official training and test files.
2. Separate `SalePrice` and `Id` from the predictors.
3. Impute numerical variables using medians learned from the current training fold.
4. Impute categorical variables using the most frequent training-fold category or a documented explicit missing category.
5. One-hot encode categorical predictors while handling previously unseen categories safely.
6. Fit a `RandomForestRegressor` to predict log-transformed `SalePrice`.
7. Evaluate the pipeline using logged-price RMSE.
8. Refit the selected configuration on all available labeled data.
9. Transform predictions back to dollars and create a valid `Id,SalePrice` submission file.

### Pipeline 2: Gradient boosting with engineered features

Pipeline 2 will differ substantively in both its model and feature representation. It will use gradient boosting and engineered predictors intended to summarize related housing characteristics. Candidate features include:

- Total finished square footage
- Total bathroom count
- Total porch area
- Property age at sale
- Years since remodeling
- Indicators for the presence of a garage, basement, fireplace, or pool

The second pipeline will use the same leakage-safe preprocessing structure as Pipeline 1, but it will fit a gradient-boosting regressor to the revised feature set. This design will test whether sequentially correcting residual errors and supplying domain-informed summary variables reduces logged-price prediction error.

## Validation and evaluation

### Local validation design

We will reserve 20% of the labeled observations as an untouched local holdout set. The remaining 80% will serve as the development set. Within that development set, we will use the same fixed five-fold cross-validation splits for both pipelines to guide model selection and assess stability.

After choosing each pipeline's configuration, we will fit it on the full 80% development set and evaluate it once on the untouched 20% holdout set. Using identical partitions for both pipelines will support a direct comparison.

All imputation, encoding, feature engineering, and model fitting will occur inside each training fold. Validation-fold and holdout observations will never be used to learn preprocessing rules or tune model settings.

### Primary metric

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

Lower values indicate better performance. This metric penalizes large errors while making proportional errors on lower- and higher-priced homes more comparable. We may also report mean absolute error as a secondary, dollar-scale diagnostic, but logged-price RMSE will determine the main comparison.

### Comparison criteria

We will compare:

1. The median-log-price reference
2. Pipeline 1: random forest
3. Pipeline 2: gradient boosting with engineered features

For each machine-learning pipeline, we will report:

- Mean five-fold cross-validation logged-price RMSE
- Standard deviation of fold scores
- Untouched holdout logged-price RMSE
- Kaggle public leaderboard score
- Difference from the naive reference
- Difference between local and Kaggle performance

Evidence against our initial expectation would include Pipeline 2 producing similar or higher error than Pipeline 1, unstable results across folds, or an apparent local improvement that does not transfer to the Kaggle test set.

## Error analysis and communication

We will inspect the largest underpredictions and overpredictions and compare error distributions across sale-price ranges. In particular, we will examine whether unusually inexpensive or expensive homes are harder to predict and whether either pipeline produces systematic residual patterns.

Planned outputs include:

- A table comparing all three models
- Cross-validation score distributions
- Predicted-versus-observed plots
- Residual plots
- Error summaries by sale-price range
- Feature-importance plots for the fitted tree ensembles
- A table of the homes with the largest prediction errors and their relevant characteristics
- A comparison of local validation results with Kaggle leaderboard scores

Feature importance will be interpreted as a model-specific measure of predictive contribution, not as evidence of causation.

## Reproducibility

Each machine-learning pipeline will have its own configuration and reproducible run command. We will record the following for every experiment and submission:

- Pipeline name and version
- Git commit
- Configuration file
- Random seed
- Run command
- Cross-validation results
- Holdout result
- Submission filename and date
- Kaggle public score

The planned repository structure is:

```text
STAT-486-Project/
├── README.md
├── .gitignore
├── configs/
├── data/
├── notebooks/
├── reports/
├── src/
└── submissions/
```

Source code will live in `src/`, configurations in `configs/`, exploratory work in `notebooks/`, figures and result tables in `reports/`, and generated Kaggle files in `submissions/`. The `data/` directory will contain instructions or placeholders only; raw Kaggle files will remain ignored by Git.

Our first runnable result will be a script that:

1. Loads the locally downloaded competition data.
2. Runs the random-forest pipeline with five-fold cross-validation.
3. Prints the mean and standard deviation of logged-price RMSE.
4. Creates a correctly formatted Kaggle submission file.

## Team workflow

Before each milestone, we will divide the work so all three members make substantive contributions to the pipelines, analysis, documentation, and presentation. Initial responsibilities will include data validation and preprocessing, Pipeline 1 implementation, Pipeline 2 implementation and feature engineering, and shared evaluation and error analysis.

We will use GitHub branches and pull requests when practical. At least one teammate will review each major change before it is merged. We will also meet in person as needed to test the complete workflow and confirm that preprocessing, validation, training, inference, and submission generation work as intended.

Early instructor feedback would be especially helpful on whether the competition is approved and whether the proposed random-forest and gradient-boosting designs satisfy the requirement for two meaningfully different pipelines.

## Risks and backup scope

### Main risks

- Missing-value and categorical-encoding errors
- Inconsistent transformations between training and test data
- Data leakage during preprocessing or feature engineering
- Overfitting caused by model tuning on a small dataset
- Long training or tuning times
- Kaggle access or submission-format problems
- A mismatch between local validation and public leaderboard results

### Backup plan

If time or computational resources become limited, we will reduce the number of trees and the size of the tuning grid while retaining the same preprocessing and validation design. We will still complete two reproducible pipelines and at least one valid Kaggle submission from each. If necessary, feature engineering will be limited to a smaller, prespecified group of variables such as total square footage, total bathrooms, and property age.

We will reconsider the plan if a pipeline consistently fails to outperform the naive reference, shows a large training-validation gap, requires more computation than is available, or cannot generate valid Kaggle submissions. Any pivot will be documented in the README with the date, supporting evidence, and reason for the change.

## Milestones

| Milestone | Target outcome |
|---|---|
| **Competition approval and repository setup** | Confirm competition and pipeline approval; establish repository structure and data instructions. |
| **Reference and Pipeline 1** | Produce median-log-price reference, random-forest cross-validation results, and first valid Kaggle submission. |
| **Pipeline 2** | Implement engineered features and gradient boosting; produce second valid Kaggle submission. |
| **Evaluation and error analysis** | Compare validation, holdout, and Kaggle results; analyze residuals and difficult observations. |
| **Final communication** | Complete README, report, figures, reproducibility checks, and presentation. |

## Sources

- [Kaggle competition overview and evaluation](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)
- [Kaggle competition data and variable descriptions](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- [Kaggle competition rules](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules)
- [Team GitHub repository](https://github.com/nchamp22/STAT-486-Project)
- [scikit-learn `RandomForestRegressor` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn `GradientBoostingRegressor` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
