# Data and Reference Result

**Checkpoint:** November 5, 2026, by 11:59 PM Mountain Time  

**Team:** Chase Nielsen, Noah Champagne, and Gunnar Griffith

**Date:**  November 5, 2026

Use any prompts that help you describe your progress. Add a dated entry to the root `README.md` of your team's repository and link the relevant code and output there; this file is optional supporting material.

## Data

- **Where did the data come from, and how did you obtain or prepare them?** 

The data came from the House Prices - Advanced Regression Techniques Kaggle competition. The data will be stored in our data/raw folder and the data that we downloaded from the Kaggle page includes four files: train.csv, test.csv, sample_submission.csv, and data_description.txt.

- **What data are available to your team now? What remains to be collected or cleaned?** 

As mentioned above, we have four files available to us through download. The data of primary interest to us is contained in train.csv, which has 1,460 rows and 81 columns (80 predictor variables and the target), and test.csv, which has 1,459 rows and 80 columns. No external data collection is required. Some features have a lot of missing data (e.g., PoolQC is missing 1,453 values, MiscFeature is missing 1,406, Alley is missing 1,369, Fence is  missing 1,179, and FireplaceQu is missing 690).

## Evaluation plan

- **What comparison, split, metric, or diagnostic will help answer your question? Why?**

We are using an 80/20 split with a fixed random state, log-RMSE, and a mean baseline. We will plot residuals against predicted prices to identify systematic errors or patterns the Random Forest misses and use cross-validation to evaluate performance across multiple folds to determine whether results are stable and not dependent on one particular split. 
- **How will you avoid data leakage or another misleading comparison?**

We will prevent data leakage by separating the training and validation data before fitting models or performing preprocessing. Any preprocessing steps, including missing-value imputation, categorical encoding, and feature selection, will be fitted only on the training data and then applied to the validation data. We will also use the same 80/20 split, random state, and log-RMSE metric when comparing different models to ensure a fair comparison. The Kaggle test dataset will remain separate and will only be used to generate final predictions. This will help ensure that our evaluation reflects how well the models perform on genuinely unseen data.  
- **How will you validate locally before making a Kaggle submission?**

We will validate our models locally using the existing 80/20 training-validation split with random_state=###. Each model will be trained on the training portion and evaluated on the validation portion using log-RMSE. We will compare the Random Forest's performance against our baseline model and any future models to determine which performs best. Once we select the best-performing model, we will retrain it using the entire labeled training dataset, generate predictions for the Kaggle test dataset, convert predictions back to dollar values, and submit them using the required CSV format. 

## First runnable result

- **What baseline or other reference did you run? What did it produce?** 

The baseline we used in our first runnable result to compare to our Random Forest and Gradient Boosting pipelines is the Scikit-learn DummyRegressor, which is a baseline regression model that makes predictions using simple rules rather than learning relationships from input features. If our pipelines do worse than this baseline, then that is a bad sign. The DummyRegressor was fit on fit on $$\log(1 + \text{SalePrice})$$ and had a log-RMSE of 0.410480.

- **Where are the code and output? Link them, and include the command needed to rerun the result if available.** 

The code we used to run our baseline result can be found in this [this interactive Python notebook](https://github.com/nchamp22/STAT-486-Project/blob/main/data_checkpoint.ipynb). 
- **What did this first result suggest, and what will you try next?** 

The result had a log-RMSE of 0.410480. We did this as our first runnable result to set a minimum reference score that any model we make should be able to beat. Our next steps are to make our Random Forest and Gradient Boosting pipelines.
