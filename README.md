# ING Datathon - Churn Prediction Project

This project was prepared as part of the ING Datathon to predict the likelihood of customers leaving the bank (churn).  
The goal is to analyze past customer behavior and predict which customers are at risk.

## 1. Project Objective

- Analyze customer data to predict churn probability  
- Generate meaningful features from time series customer activity  
- Develop prediction algorithms using CatBoost and LightGBM models  
- Combine models (ensemble) to obtain more stable results  

## 2. Data Structure

The project structure and data contents are not shared. This section is for informational purposes only.

- **Customer Information:** Age, gender, province, employment sector, etc.  
- **Customer Historical Transactions:** Monthly credit card spending, EFT transactions, number of active products, etc.  
- **Churn Target Variable:** Whether the customer left the bank by the end of the respective month  

## 3. Feature Engineering

The following features were derived from customer historical data:

- Total transaction count and amounts  
- Average amounts for mobile and credit card transactions  
- Lag features (1–6 months of historical values)  
- Delta features (current month − previous month difference)  
- Rolling window (3, 6, 12-month averages, sums, standard deviations)  
- EWMA and EWMSTD (exponentially weighted moving averages)  
- Channel ratios (mobile transaction share, credit card transaction share)  
- Transaction amount and count per product  

These processes were implemented via the `compute_history_features()` function.

## 4. Data Integration and Preparation

- Date information was converted into monthly periods  
- Historical transaction data was matched with reference dates  
- Customer demographic information was merged into the dataset  
- Missing values were filled, and categorical variables were standardized  

These operations were handled using the `build_dataset()` function.

## 5. Target Encoding (OOF)

To prevent data leakage, categorical features were encoded using Out-of-Fold (OOF) target encoding.  
The `oof_target_encode()` function was used for this process.

## 6. Models Used

Two different model architectures were employed:

| Model     | Description |
|-----------|--------------|
| **CatBoost**  | Performs well on categorical data; used with GPU support |
| **LightGBM**  | Fast, powerful tree-based model; used in GPU mode |

Both models included:

- Stratified K-Fold cross-validation  
- Early stopping  
- Parameter grid search  
- `scale_pos_weight` to handle class imbalance  

## 7. Ensemble (Model Blending)

CatBoost and LightGBM predictions were combined using a weighted approach:  
`final_prediction = w * CatBoost + (1 - w) * LightGBM`

The weights were selected based on the custom evaluation metric.

## 8. Evaluation Metrics

| Metric     | Definition |
|------------|-------------|
| **AUC**        | Model’s overall discriminatory power |
| **Gini**       | 2 × AUC − 1 |
| **Lift@10%**   | Churn rate among the top 10% most risky customers |
| **Recall@10%** | Percentage of true churn customers captured in that top 10% |
| **Custom Score** | 0.40 × Gini + 0.30 × Lift + 0.30 × Recall |

## 9. Results

- Time series–based feature engineering improved churn prediction performance  
- Combining CatBoost and LightGBM resulted in more stable outcomes than single models  
- Target encoding prevented information leakage and improved accuracy on categorical data  
- The ensemble approach achieved the highest performance on the custom score metric  

---

This README was prepared solely to explain the project approach.  
It does not include data or code sharing.
