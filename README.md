# Spotify Audio Feature-Based Music Genre Prediction

A machine learning classification project that predicts Spotify song genres from audio features using a leakage-safe preprocessing pipeline, PCA-based dimensionality reduction, KMeans-derived cluster features, and multiclass classification.

## Overview

This project uses Spotify audio feature data to classify songs into one of 10 genres. The notebook was built as a capstone-style machine learning project with a strong emphasis on:

- reproducibility
- no data leakage
- interpretable preprocessing decisions
- model comparison
- multiclass ROC/AUC evaluation

The final selected model was a **multinomial logistic regression** classifier.

## Dataset

The dataset contains about **50,000 songs** and includes audio-related variables such as:

- popularity
- acousticness
- danceability
- duration
- energy
- instrumentalness
- key
- liveness
- loudness
- mode
- speechiness
- tempo
- valence

Target variable:

- `music_genre` (10 classes)

## Project goal

The goal is to predict the genre of a song from its Spotify audio features while following a strict no-leakage train/test workflow.

## Method

The notebook follows this pipeline:

1. **Load and inspect data**
   - Read the dataset
   - Check structure and missingness
   - Clean invalid values in `duration_ms` and `tempo`

2. **Drop non-predictive metadata**
   - Removed:
     - `instance_id`
     - `artist_name`
     - `track_name`
     - `obtained_date`

3. **Create a leakage-safe train/test split**
   - For each genre:
     - 500 songs in the test set
     - 4500 songs in the training set
   - Final split:
     - **45,000 training rows**
     - **5,000 test rows**

4. **Handle missing data**
   - Median imputation on numeric columns
   - Imputer fit on training data only
   - Applied to both train and test sets

5. **Encode categorical features**
   - One-hot encoded:
     - `key`
     - `mode`

6. **Scale numeric features**
   - Standardized numeric columns only
   - One-hot categorical columns were **not** scaled

7. **Dimensionality reduction with PCA**
   - PCA fit on standardized numeric training data only
   - Selected **8 principal components** at a 90% explained-variance threshold
   - PCA was also analyzed across multiple variance thresholds

8. **Add clustering structure**
   - Fit **KMeans (k = 10)** in PCA space on the training set
   - One-hot encoded cluster assignments
   - Added cluster indicators as extra classifier features

9. **Train classification models**
   - Multinomial Logistic Regression
   - Calibrated Linear SVM

10. **Evaluate performance**
    - One-vs-Rest ROC curves
    - Macro-averaged AUC
    - Per-genre AUC
    - Confusion matrix and error analysis

## Final results

### Selected model
**Multinomial Logistic Regression**

### Test performance
- **Macro OvR AUC: 0.901**
- Calibrated Linear SVM achieved a lower AUC than logistic regression in the final comparison

### PCA sensitivity analysis
The notebook also tested multiple PCA explained-variance thresholds:

| Variance Threshold | Components | Test Macro AUC |
|---|---:|---:|
| 0.75 | 6 | 0.853 |
| 0.80 | 6 | 0.853 |
| 0.85 | 7 | 0.883 |
| 0.90 | 8 | 0.895 |
| 0.95 | 9 | 0.903 |

Although 0.95 produced a slightly higher analysis AUC in the sensitivity check, the notebook’s reported final model takeaway is the logistic-regression pipeline with a **macro OvR AUC of 0.898**.

## Per-genre AUC highlights

Best-performing genres in the final logistic regression evaluation included:

- Classical: 0.967
- Anime: 0.932
- Rock: 0.927
- Hip-Hop: 0.921
- Rap: 0.913

More difficult genres included:

- Alternative: 0.830
- Country: 0.873
- Jazz: 0.876

## Why the pipeline worked

The strongest parts of the project were:

- careful prevention of train/test leakage
- cleaning invalid numeric values before modeling
- handling missing values using train-only statistics
- keeping categorical one-hot variables unscaled
- reducing numeric noise with PCA
- adding KMeans cluster features to capture broader structure in the lower-dimensional space

Overall, the project suggests that **clean preprocessing and leakage-safe feature engineering mattered as much as model choice**.
