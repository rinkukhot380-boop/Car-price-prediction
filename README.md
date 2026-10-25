# Used Car Price Predictor

Predicts the resale price of a used car in India from 7 inputs, and serves the model through a small Streamlit app.

**Live demo:** [https://car-price-prediction-jx9cyy2fwdy4acydedtynh.streamlit.app/]

## Dataset
- **Car details v3** (CarDekho listings, Kaggle): 8,128 rows, 13 columns.
- **1,202 exact duplicate rows were removed**, leaving **6,926 unique listings**.
- Messy text fields (`mileage`, `engine`, `max_power`) were cleaned into numbers, and `brand` was taken from the car name.

## Approach
1. **Cleaning:** removed duplicates, parsed units out of text columns, extracted the brand.
2. **Target:** modelled `log(selling_price)` because prices are heavily right-skewed. Predictions are converted back to rupees with `exp()`.
3. **Pipeline:** `ColumnTransformer` (median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns) inside one `Pipeline`, fitted on training data only, so there is no leakage.
4. **Model comparison:** Linear, Ridge and Lasso with 5-fold cross-validation.
5. **Feature selection:** Lasso (alpha = 0.01) zeroed out 44 of 53 encoded features. The app uses the 7 inputs that matter most: year, km driven, max power, engine size, fuel, owner, brand.
6. **Final model:** a Random Forest on those 7 inputs beat the linear models.

## Results
| Model | Features | CV R² (log-price) |
|---|---|---|
| Linear | all 13 | 0.881 |
| Ridge | all 13 | 0.881 |
| Lasso | all 13 | 0.877 |
| Ridge | 7 | 0.880 |
| **Random Forest** | **7** | **0.906** |

**Held-out test set (Random Forest, 7 inputs):** R² = 0.900 on log-price, mean absolute error ≈ **₹73,700**.

## Limitations
- **Luxury cars are underpriced.** Out-of-fold, Mercedes-Benz is predicted about 15% too low, Audi about 8% and BMW about 7%. Mainstream brands (e.g. Toyota, Renault) are close to unbiased. Luxury brands make up under 2% of the training data, so there are few examples to learn from.
- Trained on listings up to 2020, so prices reflect that market and will be lower than today's.
- A Random Forest cannot predict outside the range of its training data.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Files: `car_price_prediction.ipynb` (analysis), `app.py` (web app), `car_model.pkl` (trained pipeline), `car_options.pkl` (dropdown values), `requirements.txt`.

## Tools
Python · Pandas · NumPy · scikit-learn · Streamlit · joblib
