# sklearn-pipelien-ml-flow

This script demonstrates a machine learning workflow using scikit-learn. It includes the following steps:

1. **Data Loading**: Loads house data from a CSV file.
2. **Data Splitting**: Splits the data into training and testing sets.
3. **Feature Preprocessing**:
   - **Numeric Features**: Scales numeric features using `StandardScaler`.
   - **Categorical Features**: Encodes categorical features using `LabelEncoder` and `OneHotEncoder`.
4. **Model Training and Evaluation**:
   - Trains a linear regression model with only numeric features.
   - Trains another linear regression model with both numeric and categorical features.
5. **Custom Transformers**:
   - Defines a custom transformer class `MyTransformer`.
6. **Pipeline Usage**:
   - Uses pipelines to preprocess data and train models.

## Dependencies

- pandas
- numpy
- scikit-learn

## How to Run

1. Ensure you have the required dependencies installed.
2. Run the script using Python:

```bash
python sklearn-pipelien-ml-flow.py
```

This script is designed to be easily understandable and reusable for similar machine learning workflows.
