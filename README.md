# CreditWise Loan System 💳

An intelligent loan approval prediction system built for **SecureTrust Bank**, using Machine Learning to predict whether a loan application should be **Approved** or **Rejected**, before final human verification.

## The Problem

SecureTrust Bank currently reviews loan applications manually — loan officers check income proofs, employment details, and credit history by hand. This process is slow, inconsistent, and prone to two costly mistakes:

- **Good customers get rejected** → the bank loses business
- **High-risk customers get approved** → the bank takes on financial losses

This project builds a Machine Learning model that learns patterns from 1,000 historical loan applications, and predicts loan outcomes automatically — faster and more consistently than manual review.

## How It Works (in plain terms)

The model is **not** a set of hardcoded rules (like "if credit score > 700, approve"). Instead, it's trained on 1,000 real historical loan decisions and learns on its own which combinations of factors — income, credit score, debt-to-income ratio, existing loans, etc. — tend to lead to approval or rejection. It then applies those learned patterns to new applicants it has never seen before.

**Pipeline:**
1. **Data Cleaning** — handled missing values in both numeric fields (mean imputation) and categorical fields (mode imputation)
2. **Exploratory Data Analysis** — visualized how income, credit score, and other factors relate to loan approval
3. **Encoding** — converted categorical fields (employment type, marital status, property area, etc.) into a numeric format models can use
4. **Model Training** — trained and compared three models: Logistic Regression, K-Nearest Neighbors, and Naive Bayes
5. **Evaluation** — measured performance using Precision, Recall, F1-score, and Accuracy (not just accuracy alone, since a bank cares more about *which kind* of mistake a model makes)

## Dataset

Each row represents one loan applicant, with 20 attributes covering their personal, financial, and credit profile — income, credit score, employment status, existing loans, collateral value, and more. Full column descriptions are in [`data/loan_approval_data.csv`](data/loan_approval_data.csv).

**Target variable:** `Loan_Approved` → `1` = Approved, `0` = Rejected

## Results

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | 0.79 | 0.80 | 0.80 | **0.875** |
| Naive Bayes | 0.78 | 0.77 | 0.78 | 0.865 |
| K-Nearest Neighbors | 0.62 | 0.51 | 0.56 | 0.755 |

**Logistic Regression performed best overall**, making it the most reliable choice for this use case — it correctly predicts loan outcomes about 87.5% of the time on data it has never seen before.

> Why these metrics matter for a bank: Accuracy alone can be misleading on imbalanced data. Precision and Recall specifically tell us how well the model avoids the two costly mistakes described above — wrongly rejecting good applicants, and wrongly approving risky ones.

## Project Structure

```
CreditWise_Loan_System/
├── data/
│   └── loan_approval_data.csv     # Historical loan application data
├── credit_wise.ipynb              # Full analysis: EDA, cleaning, modeling, evaluation
├── requirements.txt                # Python dependencies
└── README.md
```

## How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `credit_wise.ipynb` in Jupyter Notebook or VS Code and run all cells

## Tech Stack

- **Python** — pandas, numpy for data handling
- **scikit-learn** — model training and evaluation
- **matplotlib, seaborn** — data visualization
- **Jupyter Notebook** — analysis environment

## Author

Built as a Machine Learning Engineering project — designing an unbiased, data-driven loan approval system to replace manual, inconsistent review processes.
