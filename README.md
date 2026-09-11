# Card-fraud detection

Classical-ML pipeline on a card-transaction dataset, comparing logistic
regression, a random forest and a decision tree, with the selected model served
as a Dash web app.

Built as a group project (Group D) for a machine-learning module.

**Live demo:** https://fraud-detector-ic6y.onrender.com
*Hosted on Render's free tier — the first request after an idle period cold-starts
the container and can take up to a minute. Subsequent requests are immediate.*

**App source:** [MckyleM/Fraud_detection_app](https://github.com/MckyleM/Fraud_detection_app)

---

## Data

`card_transdata.csv` — 100,000 transactions, seven predictors and a binary
`fraud` label. Field meanings are in `InfoOnCsv.txt`:

| Field | Meaning |
| --- | --- |
| `distance_from_home` | Distance from home where the transaction happened |
| `distance_from_last_transaction` | Distance from the previous transaction |
| `ratio_to_median_purchase_price` | Purchase price relative to the median purchase price |
| `repeat_retailer` | Transaction was at a previously used retailer |
| `used_chip` | Transaction used the card chip |
| `used_pin_number` | Transaction used a PIN |
| `online_order` | Transaction was an online order |
| `fraud` | Target — transaction was fraudulent |

The classes are imbalanced: predicting "not fraud" for everything already scores
**0.92**, so that is the baseline every model below has to beat.

## Models

Split 80/20 via `train_test_split(test_size=0.2, random_state=42)`.

| Model | Result |
| --- | --- |
| Baseline (majority class) | 0.92 accuracy |
| Logistic regression (`max_iter=1000`) | 0.9662 train / **0.9694** validation accuracy |
| Random forest regressor (grid-searched) | 0.9998 train / **0.9987** unseen R² |
| Decision tree (`max_depth=6`) | **0.9997** validation accuracy — selected |

The decision tree was chosen: it matches the ensemble's performance while staying
small enough to deploy on a free-tier container, and it is directly inspectable.

Saved artifacts: `logistic.joblib`, `randomforestregressor.joblib`,
`decision_tree_model_dt.joblib`, and `final_model_dt.joblib` (the deployed model).

## Layout

| Path | What it is |
| --- | --- |
| `Step_by_step.ipynb` | The full pipeline — EDA, preprocessing, training, comparison |
| `card_transdata.csv` | Training data (76 MB) |
| `validation.csv` | Held-out validation set |
| `InfoOnCsv.txt` | Field documentation |
| `*.joblib` | Trained models |

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly category_encoders joblib
jupyter notebook Step_by_step.ipynb
```

The notebook expects `card_transdata.csv` in the repository root.

To run the web app, use the [app repository](https://github.com/MckyleM/Fraud_detection_app)
rather than this one — it carries the Dash front end, `requirements.txt` and the
Render blueprint.

## Note

The `DeployWithRender` entry in this repository is a stale git submodule
reference that does not resolve. The deployment lives in the app repository
linked above.
