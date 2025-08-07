# Fraud Detection

This document outlines the strategy for building a fraud detection system. It is an extraction from a broader document, focusing exclusively on the fraud detection problem.

## 1 Problem formulation

### Fraud Detection: Anomaly Detection

**The Business Problem:** We need to build a system that can analyze incoming payment transactions and accurately identify which ones are fraudulent. The primary objective is to identify and flag these fraudulent payments, thereby preventing financial loss. A crucial constraint is to achieve this with minimal disruption to legitimate customers, meaning we must keep the rate of incorrectly flagged transactions (false positives) as low as possible.

#### Key Performance Indicators (KPIs)

* **Area Under the Precision-Recall Curve (AUPRC):** Given the severe class imbalance (very few fraudulent transactions), AUPRC is a more informative metric than simple accuracy or AUROC. It measures the trade-off between precision and recall for the positive (fraud) class.
* **Cost-weighted F₁ Score:** A custom metric that assigns different costs to false positives (blocking a real customer) and false negatives (letting fraud through). This directly aligns the model's performance with business impact, often expressed as "net dollars saved per 1,000 transactions."
* **Mean Time to Detect (MTTD):** Measures how quickly the system can identify and adapt to new types of fraudulent activity.

#### Core Challenges

* **Extreme Class Imbalance:** Fraudulent events are rare, often accounting for less than 0.1% of all transactions. This makes it difficult for a model to learn their patterns without being overwhelmed by the majority class.
* **Concept Drift:** Fraudsters constantly change their tactics (Methods of Operation or "MOs"). A model trained on historical data may quickly become outdated. The system must be able to adapt to these evolving patterns.
* **Latency Constraints:** To be effective, the fraud check must happen efficiently within the payment processing flow. This imposes a strict latency budget, typically under 150 milliseconds for the entire feature lookup and scoring process.

---

## 2 Data‑preparation plan

### 2.1 Raw Data "Mock Sheet"

For the fraud detection problem, we would source data from the following key tables:

#### Fraud Detection Data

##### `payments_raw`

* **Volume / Cadence:** 2 M rows / day, via stream
* **Source:** Internal payment processing logs, generated as customers transact via payment gateways (e.g., Stripe, Adyen).
* `txn_id` (string): Unique transaction identifier.
* `cust_id` (string): Unique customer identifier.
* `amount` (float): Transaction value in a standard currency.
* `timestamp` (datetime): Time of the transaction.
* `geo_ip` (string): IP address of the device used for the transaction.
* `device_fingerprint` (string): Hash representing the user's device and browser combination.
* `mcc_code` (integer): Merchant Category Code, indicating the type of business (e.g., restaurant, airline).
* `label_is_fraud` (boolean): The ground truth label, available post-facto after investigation.

##### `customer_profile` (Fraud-related fields)

* **Volume / Cadence:** 6 M active customers, daily snapshot
* **Source:** A combination of the internal CRM database and data enriched from external credit bureaus (e.g., Experian, Equifax).
* `cust_id` (string): Unique customer identifier, links to `payments_raw`.
* `account_age_days` (integer): How long the customer has been with the company. A very new account can be a risk signal.
* `credit_score` (integer): A numerical score representing the customer's creditworthiness.
* `avg_txn_amount` (float): The customer's average transaction amount over their lifetime.
* `country_of_residence` (string): The customer's registered country.

### 2.2 ETL / Feature Store Pipeline

1. **Ingestion & Staging:**

   * **Multi-source Ingestion:** Data arrives from diverse sources and formats. We'll need connectors for Kafka streams (`payments_raw`), SFTP drops for batch files (CSVs, Excel files from partners), and REST APIs (for fetching data from sources which might use JSON or XML).
   * **Raw Data Lake:** All incoming data is first landed in its raw format into a staging area in a data lake (e.g., AWS S3, Google Cloud Storage, or Azure Blob Storage). This provides a durable, auditable source of truth.
   * **Initial Structuring:** From the data lake, initial processing jobs parse the various formats and load the data into a more structured, but still raw, format (e.g., Parquet tables in the lake). This is where we decide on the best storage: non-relational/columnar stores (like Mongo or Elasticsearch) are ideal for high-volume event logs, while relational databases (like PostgreSQL) are well-suited for customer profile data.
2. **Cleaning & Normalisation:**

   * **Schema Enforcement & Type Consistency:** Apply strict schemas to all datasets. This includes casting fields to their correct data types (e.g., ensuring all monetary values are `float`, IDs are `string`). A critical step is standardizing inconsistent formats, such as converting all date strings (`MM/DD/YYYY`, `YYYY-DD-MM`) to the ISO 8601 standard (`YYYY-MM-DDTHH:MM:SSZ`).
   * **Contextual Data Validation:** Instead of blindly removing rows, we apply business-rule-based validation. For example:
     * A negative `amount` in `payments_raw` could be a valid refund, so it should be flagged but not necessarily dropped.
     * A `timestamp` in the near future might be valid for a scheduled payment. We would define a reasonable lookahead window (e.g., 90 days) and flag anything beyond that as a potential error.
     * Similarly, a `timestamp` too far in the past (e.g., before the customer's account existed or older than a reasonable transaction history, say 2 years) is likely an error and should also be flagged.
   * **Value Unification:** Standardise categorical features (e.g., mapping `"USA"`, `"U.S."`, and `"United States"` to a single representation). This also includes unifying geo-resolution from IP addresses.
3. **Enrichment / Feature Engineering:**

   This step is where we create the predictive signals for our models by combining and transforming the cleaned data.

   #### Fraud Detection Features


   * **Transactional Aggregates (Velocity Features):** These capture the pace and pattern of spending. We'll compute these over various rolling time windows (e.g., last 1h, 6h, 24h, 7d).
     * `txn_count_window`: Number of transactions.
     * `txn_amount_sum_window`: Total amount spent.
     * `unique_merchants_window`: Number of distinct merchants visited.
     * `avg_time_between_txns_window`: Average time between transactions.
   * **Behavioral Profile Comparison:** We'll compare the current transaction to the customer's historical behavior.
     * `amount_zscore`: How many standard deviations is the current transaction amount from the customer's historical average?
     * `is_new_merchant`: Has the customer ever used this merchant before?
     * `is_new_country`: Is this the first transaction from this country for this customer?
     * `balance_depletion_speed`: A feature that tracks how quickly a series of transactions is depleting the account's available balance, flagging rapid "cash-out" patterns common in account takeovers.
   * **Graph-based Network Features:** We will model relationships between entities to uncover sophisticated fraud rings and patterns, drawing inspiration from anti-money laundering (AML) techniques. This requires a batch process using a graph database (e.g., Neo4j) or library (e.g., NetworkX) to pre-compute scores that are then served from the feature store.
     * We will construct a graph where customers, devices, IP addresses, and merchants are nodes, and transactions are edges.
     * `device_fraud_ring_score`: A score based on how many other accounts have used the same `device_fingerprint` and if any of them have confirmed fraud.
     * **Transaction Cycle Detection:** We will run algorithms to detect circular money movements (e.g., Account A pays B, B pays C, and C pays A back). Such cycles are highly anomalous and a strong indicator of synthetic activity.
     * **"Smurfing" Pattern Detection:** We will identify patterns of "structuring," where a single entity uses multiple accounts to make many small transactions. This includes flagging repeated suspicious amounts (e.g., values like $1,999 or amounts just under a legal reporting threshold) to stay below detection thresholds. A feature could be `fan_in_ratio` or `fan_out_ratio` for an account over a specific period.
   * **Geo-Spatial Enrichment:**
     * By joining the `geo_ip` with a database like MaxMind, we can derive `ip_risk_score` or check if it's from an anonymous proxy.
     * Joining `postcode` with public census data can provide `postcode_deprivation_index` or `regional_risk_score`.
4. **Missing Values & Outliers:**

   * **Continuous:** Winsorise numerical features at the 1st and 99th percentiles to handle extreme outliers. Impute missing values using the median, or create a distinct *missing* category if absence is a signal.
   * **Categorical:** Introduce a dedicated "UNK" (Unknown) category for unseen values during inference.
   * The outlier detection process itself can generate features for the unsupervised fraud model.
5. **Feature Store: The Bridge Between Data and Models**
   A crucial question is whether we need a dedicated Feature Store or if storing engineered features in a standard database is sufficient. For our use cases, especially fraud detection, a feature store is highly recommended.

   #### The Core Problem: Train-Serve Skew

   The primary challenge a feature store solves is preventing **train-serve skew**. This occurs when the features used to train a model are different from the features used to make predictions in production. This can happen for several reasons:


   * **Dual Logic:** Without a feature store, teams often write one pipeline for batch-generating features for training (e.g., a Spark job) and a separate, independent implementation for the serving path (e.g., a microservice). It is extremely difficult to keep the logic of these two systems perfectly identical. A subtle difference in how nulls are handled or a floating-point calculation can degrade model performance.
   * **Data Availability:** The serving system needs to generate features with very low latency (<150ms for fraud). A standard analytical database is not built for such rapid key-value lookups, whereas a feature store has a specialized low-latency online component.
   * **Point-in-Time Correctness:** For training, it's critical to use the feature values as they existed at the exact moment of the event. Querying a standard database to get this "point-in-time" view for millions of historical events is complex and slow.

   #### How a Feature Store Solves This

   A feature store (like Feast, Tecton, or Hopsworks) provides a unified framework to solve these issues:

   * **Define Once, Use Anywhere:** You define the feature transformation logic a single time. The feature store then manages both the batch generation of historical features for training and the low-latency serving of online features for inference. This guarantees consistency.
   * **Online/Offline Architecture:** It automatically manages two storage layers:
     * An **Offline Store** (e.g., data warehouse tables) for large-scale training data.
     * An **Online Store** (e.g., Redis, DynamoDB, or Azure Cosmos DB) for fast, key-value lookups at serving time.
   * **Discovery and Governance:** It acts as a central registry for features, allowing teams to discover, share, and reuse features, which improves efficiency and governance.

   For our project, the high-stakes nature of **Fraud Detection** makes a feature store almost essential.

---

## 3 Modelling strategy

### 3.1 Fraud Detection

Our fraud detection strategy is a multi-layered approach, combining a powerful supervised model with techniques to handle the specific challenges of this domain.

* **Baseline Model: Gradient-Boosted Trees**

  * **Technique:** We will use a Gradient-Boosted Tree model, specifically XGBoost or LightGBM.
  * **Rationale:** These models are the industry standard for tabular data. They excel at handling mixed feature types (numerical and categorical), are highly performant for inference, and have built-in mechanisms like `scale_pos_weight` to handle class imbalance effectively.
* **Unsupervised Supplement for Novel Fraud**

  * **Technique:** We will deploy an unsupervised model, such as a Deep Autoencoder or an Isolation Forest, in parallel.
  * **Rationale:** This model is not trained on fraud labels but on learning the structure of "normal" transactions. It helps us catch fraud patterns our main model hasn't seen before. The anomaly score it generates becomes a powerful feature for the main GBDT model.
    * A **Deep Autoencoder** is a neural network trained to reconstruct its input. When trained on normal transactions, it fails to accurately reconstruct anomalous ones, resulting in a high "reconstruction error" which we use as an anomaly score.
    * An **Isolation Forest** builds a forest of random decision trees. It isolates observations by randomly selecting a feature and then randomly selecting a split value. Anomalies are "few and different" and are therefore easier to isolate, leading to shorter paths in the trees and a lower isolation score.
* **Combining the Models: A Hybrid Approach**

  The supervised GBDT model and the unsupervised anomaly detector are not used in isolation; they are combined in a hybrid structure to maximize both accuracy on known patterns and detection of novel threats. This is more sophisticated than a simple voting ensemble.

  * **Method 1: Stacking via Feature Engineering:** This is the primary method of combination. The unsupervised model (e.g., Isolation Forest) is run first. Its output—a numerical anomaly score for each transaction—is not used to make a final decision directly. Instead, this score is appended as a new column to the dataset and fed as a highly informative **feature** into the supervised GBDT model. This "stacks" the models, allowing the GBDT model to learn the relationship between the anomaly score and the likelihood of fraud, effectively combining the strengths of both approaches.
  * **Method 2: Rule-Based Safety Net:** The final decision logic provides a secondary layer of combination. A transaction is flagged if *either* of two conditions is met:

    1. The main GBDT model's predicted fraud probability is above its tuned threshold (`GBDT_proba > τ₁`).
    2. The unsupervised model's anomaly score is above a separate, much higher threshold (`AnomalyScore > τ₂`).

    This creates a crucial **safety net**. The GBDT model is the primary decision-maker, but if a completely new type of fraud appears that the GBDT model has never seen (and thus gives a low score), the unsupervised model can still catch it if it is sufficiently anomalous. This prevents catastrophic failures due to concept drift.
* **Advanced Imbalance Handling**

  * **Technique:** Beyond simple model weighting, we will implement dynamic class weighting where the weight is proportional to the financial cost of an error. We will also experiment with advanced sampling techniques like SMOTE-NC during training.
  * **Rationale:** This directly aligns the model's optimization function with the business impact of its errors. By creating synthetic fraud examples, we give the model more data to learn from, improving its ability to recognize fraudulent patterns.
    * **SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features):** This is a powerful technique for creating "synthetic" examples of the minority class (fraud). Unlike standard SMOTE, it can correctly handle datasets containing both numerical and categorical features, which is exactly our case. It generates new fraud samples by interpolating between existing fraud cases for numerical features and using the most frequent category for categorical features among nearest neighbors.
* **Explainability for Compliance**

  * **Technique:** We will use the SHAP (SHapley Additive exPlanations) framework to generate multiple types of visual explanations:
    * **Global Feature Importance:** SHAP summary plots will show the top 20 most influential features across all predictions.
    * **Feature Dependence Plots:** These plots will help us understand how the value of a single feature (e.g., transaction amount) affects the model's output, capturing non-linear relationships.
    * **Local Explanations:** For individual transaction reviews, we will use LIME or SHAP force plots for case-by-case analysis.
  * **Rationale:** This multi-faceted approach to explainability is crucial. Global importance plots provide a high-level overview for model validation. Dependence plots allow data scientists to debug feature behavior. And local explanations are required by compliance teams and fraud analysts to justify individual decisions to block a transaction.

#### Serving path

The inference pipeline will be simple and fast:

```text
(Payment JSON) → FeatureLookups → GBDT model → If AnomalyScore > τ₂ OR GBDT_proba > τ₁ → block & queue for manual review
```

---

## 4 Evaluation & testing

Our evaluation strategy is two-pronged, with distinct offline and online testing plans to ensure both statistical rigor and real-world business impact.

### Fraud Model Evaluation

#### Offline Evaluation

* **Method:** We will use a time-sensitive, stratified 5-fold cross-validation. This approach simulates a realistic production scenario by always training on older data and validating on more recent data (e.g., train on days `t-60` to `t-7`, validate on days `t-7` to `t`). Stratification ensures the rare fraud class is appropriately represented in each fold.
* **Key Metrics:** Our primary offline metric will be **AUPRC (Area Under the Precision-Recall Curve)**, which is ideal for imbalanced datasets. We will also plot a **net-dollar-saved curve** to directly translate model performance into business value.

#### Online Evaluation

* **Method:** The model will first be deployed in **shadow mode**, where it makes predictions on live traffic without taking any blocking action. This allows us to safely monitor its decisions and alert volumes. After validation, we will perform a **progressive rollout**, starting with 1% of traffic and gradually increasing to 100%, all managed via feature flags for quick rollback if needed.
* **Key Metrics:** In production, we will monitor core business KPIs: the **chargeback rate** (our primary measure of success), the **false-positive cost** (from reviewing incorrectly blocked transactions), and the model's average **prediction latency**.

### Automated Model Quality & Safety Gates

A crucial part of our MLOps strategy is building automated gates to ensure that only high-quality, safe models make it to production.

#### Offline Pre-deployment Gates

* **What it is:** This is an automated check within our CI/CD pipeline (e.g., using GitHub Actions) that runs before a model can be deployed.
* **Mechanism:** The pipeline automatically trains the new model candidate and evaluates it against our held-out test set. We then compare its key offline metrics (e.g., AUPRC for fraud) against the currently deployed production model's scores on the same test set.
* **The Gate:** If the new model shows a regression (namely performance decrease) greater than a pre-defined tolerance (e.g., 0.5 percentage points), the pipeline fails. This automatically blocks the model from being promoted, preventing a worse model from reaching users.

#### Online Post-deployment Gates

* **What it is:** This is a live monitoring and automated rollback system that operates during the progressive rollout phase.
* **Mechanism:** As the new model serves a small percentage of live traffic (e.g., the first 1% or 5%), an automated monitoring system will track its performance on key business KPIs.
* **The Gate:** We will set predefined safety thresholds for critical online metrics (e.g., a >10% increase in the false positive rate, a >20ms increase in median latency). If the new model variant breaches any of these thresholds for a sustained period, the system will automatically trigger an alert and can be configured to halt the rollout and revert traffic back to the stable production model, ensuring minimal negative impact.

---

## 5 Recommended Technology Stack

Choosing the right cloud platform is a critical decision that impacts the entire lifecycle of the MLOps project, from development and deployment to scalability and cost-management. This section provides a recommended technology stack, offering a primary recommendation based on stated preferences and team familiarity, alongside a comparative analysis of alternatives.

### Primary Recommendation: Microsoft Azure

Given the team's existing experience and familiarity with Azure, this platform is the recommended choice. This familiarity translates into faster development cycles, more efficient troubleshooting, and a smoother operational experience. Azure provides a mature, enterprise-grade ecosystem of services that comprehensively cover the needs of our fraud detection system.

**Subjective View:** My personal experience aligns with the team's; I have found Azure's services to be well-integrated and the developer experience to be slightly more streamlined, particularly for teams that are already part of the Microsoft ecosystem.

#### End-to-End Azure Stack

1. **Data Ingestion & Storage (Data Warehouse):**

   * **Azure Data Factory (ADF):** To orchestrate the ingestion of data from various sources (Kafka, SFTP, APIs).
   * **Azure Data Lake Storage (ADLS) Gen2:** The central data lake for storing raw data in formats like Parquet.
   * **Azure Synapse Analytics:** A unified analytics platform that combines data warehousing, data integration, and big data analytics. It will serve as our primary data warehouse for structured data and for running large-scale feature engineering jobs using its integrated Spark engine.
2. **Feature Engineering & Transformation:**

   * **Azure Databricks or Synapse Spark:** For large-scale, distributed data processing and feature engineering, leveraging PySpark as described in the "Production Scale" code snippets.
   * **Azure Functions:** For real-time, event-driven feature calculations on streaming data.
3. **Feature Store:**

   * **Feast (on Azure):** We can deploy the open-source feature store Feast, using ADLS for the offline store and **Azure Cosmos DB or Azure Cache for Redis** for the low-latency online store. This combination is ideal for solving the train-serve skew problem and meeting the <150ms latency requirement.
4. **Model Training & Tracking:**

   * **Azure Machine Learning (AzureML):** This will be the core of our MLOps workflow.
     * **Training:** Use AzureML's compute instances for training the XGBoost and Isolation Forest models.
     * **Experiment Tracking & Model Registry:** Use **MLflow**, which is natively integrated into AzureML. This will be used to log experiments, store model artifacts, track parameters, and version our models, providing crucial governance and reproducibility.
5. **Model Deployment & Serving:**

   * **Azure Kubernetes Service (AKS) with AzureML:** For deploying the final model as a high-availability, scalable REST endpoint. AzureML simplifies the process of containerizing the model and deploying it to AKS.
   * **Azure API Management:** To act as a gateway for the model endpoint, handling authentication, rate limiting, and monitoring.
6. **CI/CD & Automation:**

   * **GitHub Actions:** For orchestrating the entire CI/CD pipeline, from code commit to automated model training, evaluation, and deployment. The automated quality gates described in the "Evaluation & testing" section will be implemented as steps in this pipeline.
7. **Monitoring & Observability:**

   * **Azure Monitor & Application Insights:** To monitor the health of the deployed model, track latency, and set up alerts for performance degradation or infrastructure issues.
   * **Power BI or Grafana:** For creating dashboards to visualize model performance metrics (AUPRC, chargeback rates, etc.) and business KPIs.

### Comparative Analysis: GCP and AWS

While Azure is the primary recommendation, both GCP and AWS offer powerful and viable alternatives. The choice often depends on specific organizational strengths, existing infrastructure, and cost considerations.

| **Component**           | **Microsoft Azure (Recommended)**      | **Google Cloud Platform (GCP)**                         | **Amazon Web Services (AWS)**                                    |
| :---------------------------- | :------------------------------------------- | :------------------------------------------------------------ | :--------------------------------------------------------------------- |
| **Data Warehouse**      | **Azure Synapse Analytics, ADLS Gen2** | **Google BigQuery, Google Cloud Storage**               | **Amazon Redshift, Amazon S3**                                   |
| **Feature Engineering** | **Azure Databricks, Synapse Spark**    | **Dataproc, Vertex AI Dataflow**                        | **Amazon EMR, AWS Glue**                                         |
| **Feature Store**       | **Feast on Azure (Cosmos DB/Redis)**   | **Vertex AI Feature Store**                             | **Amazon SageMaker Feature Store**                               |
| **Model Training/Mgmt** | **Azure Machine Learning with MLflow** | **Vertex AI Training & Experiments**                    | **Amazon SageMaker Training & Experiments**                      |
| **Model Deployment**    | **Azure Kubernetes Service (AKS)**     | **Vertex AI Endpoints, Google Kubernetes Engine (GKE)** | **Amazon SageMaker Endpoints, Elastic Kubernetes Service (EKS)** |
| **CI/CD**               | **GitHub Actions**                     | **Google Cloud Build**                                  | **AWS CodePipeline**                                             |

#### Objective Criteria & Key Differentiators:

* **Integration:**

  * **Azure:** Excellent integration with enterprise tools (Office 365, Active Directory) and a very cohesive feel across its data and AI services. The native MLflow integration is a significant plus.
  * **GCP:** Often praised for its strengths in Kubernetes (GKE), big data (BigQuery is a serverless powerhouse), and AI/ML innovation (Vertex AI is a highly unified and developer-friendly platform).
  * **AWS:** The most mature and comprehensive platform with the widest array of services. SageMaker is an incredibly powerful and flexible suite, though it can sometimes feel less integrated than GCP's Vertex AI.
* **Cost:**

  * Cost is highly dependent on usage patterns and negotiated discounts.
  * **GCP's BigQuery** has a serverless, consumption-based pricing model that can be very cost-effective for sporadic, heavy queries.
  * **Azure Synapse** and **Amazon Redshift** often involve provisioned capacity, which can be better for predictable, steady workloads.
* **Ease of Use:**

  * **GCP's Vertex AI** is often cited as having a very clean, unified, and intuitive user experience for MLOps.
  * **Azure ML** has made significant strides in usability and its integration with MLflow is a major advantage for teams already familiar with that tool.
  * **AWS SageMaker**, while extremely powerful, has a steeper learning curve due to its vast number of components and options.

In summary, while all three platforms are leaders in the space, the recommendation for **Azure** is based on leveraging the team's existing expertise to maximize efficiency and speed of delivery.

---

## 6 GenAI / LLM Integration Ideas

Here are several ways we can leverage Large Language Models (LLMs) to enhance our project:

* **Synthetic Data Generation for Robustness Testing**
  LLMs can create high-quality synthetic data to augment our training sets and improve model robustness.

  * **For Fraud Detection:** By prompting a model like GPT-4.1 with scenarios like, *"Generate 20 realistic fraudulent transaction sequences conditioned on a stolen credit card,"* we can create diverse, hard-to-detect examples to augment our rare minority class and stress-test the model's resilience.
* **Advanced Feature Extraction from Text**
  LLMs are excellent at converting unstructured text into meaningful numerical representations (embeddings). We can use a powerful embedding model (e.g., OpenAI's `text-embedding-3-large`) to process text fields like merchant descriptors or insurance claim notes. These rich vector embeddings can then be fed as features into our downstream models.
* **Semantic Fraud Signal Detection**
  This is a more targeted application of text analysis for fraud. An LLM can be trained or fine-tuned to specifically look for signs of fraudulent intent within text fields. For example, it could analyze recipient names, payment descriptions, or account holder information to flag semantically suspicious patterns (e.g., a recipient named "Quick Cash Ltd.", variations of the same name to avoid limits, or descriptions that sound like money laundering). The LLM's output (e.g., a "suspiciousness score") would be a powerful feature for the main fraud model.
* **Interactive Analyst Tooling**
  We can build a chat-based interface that allows data analysts to query our data and model results using natural language. This would enable an analyst to ask questions like, *"Why has the fraud block rate increased for users in segment X over the last week?"* The LLM-powered tool would then translate this question into the appropriate queries against our feature store and SHAP explanation dashboards.
* **Automated Documentation and Lineage**
  An LLM can be configured to watch our project's Git repository. When a data scientist commits a change to a feature engineering pipeline, the LLM can automatically draft documentation for the new feature, describe its lineage, and even update the README for the corresponding Airflow DAG.
* **AI-assisted Development**
  The entire development lifecycle can be accelerated by using AI coding assistants (e.g., Cursor, GitHub Copilot). These tools can write boilerplate code for data processing, generate unit tests for feature transformations, help debug complex models, and explain unfamiliar parts of the codebase, freeing up data scientists to focus on higher-level problem-solving.
* **Multimodal Document Processing**
  For tasks like customer onboarding or claim processing, we can use multimodal models that understand both images and text. A user could upload a photo of their ID or a PDF of a medical bill. The model could then extract the structured information, check for signs of tampering, and automatically populate the user's profile, streamlining data entry and verification.
* **Reasoning for Complex Analysis**
  We can use the advanced reasoning capabilities of LLMs to automate complex analysis. For example, a "fraud investigation assistant" could be built. When a complex case is flagged, the model would receive all linked data and generate a step-by-step "chain-of-thought" hypothesis about the fraud method (e.g., "Hypothesis: Account Takeover. Step 1: Login from a new, high-risk IP. Step 2..."). This would drastically speed up the work of human analysts.

---

## 7 Bonus: Code from implementation notebooks

A selection of Python snippets from the accompanying notebooks that demonstrate key parts of the implementation.

### Fraud Detection: Feature Engineering (Pandas)

This snippet shows the creation of time-windowed "velocity" features, which are highly predictive for fraud.

```python
# Create time-windowed velocity features using pandas
for window_hours in [1, 6, 24, 168]:  # 1h, 6h, 1d, 1w
    window_str = f'{window_hours}H'
    data_indexed = data.set_index('timestamp')
  
    # Count of transactions in window
    data[f'txn_count_{window_hours}h'] = (
        data_indexed.groupby('cust_id')
        .rolling(window_str, closed='both')
        .size()
        .reset_index(level=0, drop=True)
    )
  
    # Sum of amounts in window
    data[f'amount_sum_{window_hours}h'] = (
        data_indexed.groupby('cust_id')['amount']
        .rolling(window_str, closed='both')
        .sum()
        .reset_index(level=0, drop=True)
    )
```

### Fraud Detection: Cost-Sensitive Model Training (XGBoost)

Here we configure an XGBoost classifier, paying special attention to the `scale_pos_weight` parameter. This tells the model to penalize misclassifying a rare "fraud" instance much more heavily than a "legitimate" one, aligning the model with business costs.

```python
# Calculate class weights and train a cost-sensitive XGBoost model
n_legitimate = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()

# Give more weight to catching fraud based on business cost
fraud_cost = 100
false_positive_cost = 1
scale_pos_weight = (n_legitimate * fraud_cost) / (n_fraud * false_positive_cost)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr', # Area under PR curve is better for imbalance
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    n_estimators=200,
    random_state=42
)
# Train using data balanced with SMOTE-NC for better performance
xgb_model.fit(X_train_balanced, y_train_balanced)
```

### Fraud Detection: Unsupervised Anomaly Detection (Isolation Forest)

To catch novel fraud patterns not seen in the training labels, we supplement our main model with an unsupervised anomaly detector like Isolation Forest. Its score can be a powerful feature.

```python
# Train an Isolation Forest on legitimate data to find novel anomalies
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    contamination=0.001,  # Expected proportion of anomalies
    random_state=42
)

# Train only on normal transactions
X_train_legitimate = X_train[y_train == 0]
iso_forest.fit(X_train_legitimate)

# Anomaly scores can now be generated for all data
anomaly_scores = iso_forest.decision_function(X_test)
```

### For Production Scale: Feature Engineering (PySpark)

While notebooks can use pandas for rapid prototyping, a production system handling millions of daily events would use a distributed framework like Spark. This snippet illustrates how similar velocity features would be generated at scale.

```python
# === Fraud features (PySpark) ===
from pyspark.sql import functions as F, Window

# Define window for last 6 hours of activity per card
window_6h = Window.partitionBy('cust_id').orderBy('timestamp').rangeBetween(-21600, 0) # seconds in 6h

txn_feats = (
    payments_raw
    .withColumn('amt_zscore', (F.col('amount') - mean_amt) / std_amt)
    .withColumn('txn_count_6h', F.count('*').over(window_6h))
    .withColumn('unique_mcc_24h',
                F.approx_count_distinct('mcc_code').over(window_6h.rangeBetween(-86400, 0)))
)
```
