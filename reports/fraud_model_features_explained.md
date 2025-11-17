# Fraud Detection Model Features Explained for Actuaries

This document provides clear definitions of the key features used in our fraud detection model, ranked by their importance according to SHAP (SHapley Additive exPlanations) analysis. These features help the model identify potentially fraudulent transactions.

## Understanding SHAP Importance Scores

**SHAP Score**: This measures how much each feature contributes to the model's fraud predictions, on average across all transactions. Think of it as the "influence weight" of each feature:
- **Higher scores** = More influential in determining fraud risk
- **Range**: Typically 0.0 to 1.0+ (higher is more important)
- **Interpretation**: A score of 0.5 means this feature has about half the influence of a feature scoring 1.0

## Top 20 Most Important Features (by SHAP importance)

| Rank | Feature Name | SHAP Score | Category | Definition | Business Interpretation |
|------|--------------|------------|----------|------------|------------------------|
| 1 | `unique_merchants_30d` | 0.661 | **Merchant Behavior** | Number of distinct merchants a card has transacted with in the last 30 days | Higher variety of merchants may indicate normal shopping behavior OR potential account compromise |
| 2 | `amount` | 0.517 | **Transaction Amount** | Raw transaction amount (can be positive or negative) | Large amounts are often scrutinized; negative amounts indicate refunds/reversals |
| 3 | `amt_sum_24h` | 0.448 | **Velocity** | Total sum of transaction amounts for the card in the last 24 hours | High daily spending totals may indicate fraud or legitimate high-value periods |
| 4 | `hour_cos` | 0.354 | **Temporal Pattern** | Cosine transformation of transaction hour (captures cyclical time patterns) | Transactions at unusual hours (e.g., 3 AM) may be suspicious |
| 5 | `dow` | 0.212 | **Temporal Pattern** | Day of week (0=Monday, 6=Sunday) | Certain fraud patterns may be more common on specific days |
| 6 | `merchant_freq_global` | 0.174 | **Merchant Risk** | How frequently this merchant appears across all transactions (normalized) | Very rare merchants may pose higher risk due to less transaction history |
| 7 | `operation_type_shopping_pos` | 0.114 | **Transaction Type** | Binary indicator for Point-of-Sale shopping transactions | Certain transaction types have different fraud risk profiles |
| 8 | `unique_merchants_7d` | 0.106 | **Merchant Behavior** | Number of distinct merchants a card has used in the last 7 days | Similar to 30-day metric but captures more recent behavioral changes |
| 9 | `prop_new_merchants_30d` | 0.101 | **Novelty Risk** | Proportion of transactions with new merchants in the last 30 days | High proportion of new merchants may indicate account compromise |
| 10 | `amount_z_iqr_cat` | 0.095 | **Amount Anomaly** | Z-score of transaction amount within its category, using robust IQR-based scaling | Measures how unusual the amount is for this type of transaction |
| 11 | `hour` | 0.067 | **Temporal Pattern** | Raw hour of day (0-23) when transaction occurred | Complements hour_cos; certain hours are riskier than others |
| 12 | `if_anomaly_score` | 0.062 | **Anomaly Detection** | Score from Isolation Forest model indicating transaction unusualness | Higher scores suggest the transaction pattern is unusual |
| 13 | `operation_type_shopping_net` | 0.050 | **Transaction Type** | Binary indicator for online shopping transactions | Online transactions may have different risk profiles than in-person |
| 14 | `hour_sin` | 0.043 | **Temporal Pattern** | Sine transformation of transaction hour (captures cyclical patterns) | Works with hour_cos to model time-of-day risk patterns |
| 15 | `operation_type_misc_pos` | 0.040 | **Transaction Type** | Binary indicator for miscellaneous point-of-sale transactions | Captures risk associated with less common POS transaction types |
| 16 | `time_since_last_sec` | 0.039 | **Velocity** | Seconds elapsed since the card's previous transaction | Very short intervals may indicate rapid-fire fraud attempts |
| 17 | `operation_type_food_dining` | 0.036 | **Transaction Type** | Binary indicator for food and dining transactions | Food purchases typically have lower fraud risk |
| 18 | `operation_type_home` | 0.034 | **Transaction Type** | Binary indicator for home improvement/household transactions | Different transaction categories have varying risk profiles |
| 19 | `operation_type_entertainment` | 0.031 | **Transaction Type** | Binary indicator for entertainment-related transactions | Entertainment purchases may have specific fraud patterns |
| 20 | `prop_new_merchants_7d` | 0.023 | **Novelty Risk** | Proportion of new merchants in the last 7 days | Short-term indicator of merchant novelty behavior |

## Feature Categories Explained

### **Transaction Amount Features**
- **Purpose**: Identify unusually large, small, or contextually inappropriate amounts
- **Risk Indicators**: Transactions significantly above/below typical amounts for the merchant type or cardholder

### **Temporal Pattern Features**
- **Purpose**: Detect transactions occurring at unusual times
- **Risk Indicators**: Purchases at 3 AM, unusual day-of-week patterns, seasonal anomalies
- **Technical Note**: Sin/cos transformations preserve the cyclical nature of time (hour 23 is close to hour 0)

### **Merchant Behavior Features**
- **Purpose**: Track how cardholders interact with merchants over time
- **Risk Indicators**: Sudden changes in merchant diversity, concentration of spending at unknown merchants

### **Velocity Features**
- **Purpose**: Detect rapid-fire transactions that may indicate automated fraud
- **Risk Indicators**: Multiple transactions in seconds/minutes, unusually high daily spending totals

### **Novelty Risk Features**
- **Purpose**: Identify when cardholders deviate from established patterns
- **Risk Indicators**: First-time merchants, new transaction types, geographical changes

### **Transaction Type Features**
- **Purpose**: Apply category-specific risk rules
- **Risk Indicators**: High-risk categories (e.g., cash advances) or unusual category combinations

### **Anomaly Detection Features**
- **Purpose**: Capture complex patterns that rule-based systems might miss
- **Risk Indicators**: Transactions that don't fit typical cardholder behavior patterns

## Model Interpretation Guidelines for Actuaries

1. **Feature Interactions**: The model considers combinations of features, not just individual values. For example, a large amount at an unusual hour at a new merchant creates higher risk than any single factor alone.

2. **Threshold Considerations**: Features are calibrated per transaction category to account for different spending patterns (e.g., grocery vs. luxury goods).

3. **Seasonal Adjustments**: The model includes cyclical features to handle legitimate seasonal spending variations.

4. **False Positive Management**: Features like merchant frequency help distinguish between legitimate new spending patterns and potential fraud.

5. **Privacy Protection**: All merchant and user identifiers are hashed or aggregated to protect customer privacy while preserving risk signals.

## Model Performance Context

- **Primary Dataset**: ULB Credit Card Fraud (European cardholders)
- **Current Performance**: 92% Average Precision, 98% Precision at 0.5% alert rate
- **Calibration**: Model outputs are probability-calibrated using isotonic regression
- **Validation**: Time-aware cross-validation ensures model works on future unseen data

This feature set represents the enhanced model version implemented in September 2025, incorporating seasonality and balance velocity improvements that increased precision by 1-2 percentage points over the baseline model.