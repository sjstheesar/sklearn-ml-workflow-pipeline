import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

house_data = load_data('./data/house_data.csv')
house_data.head(3)

# Split data into features and target variable
def split_data(data, target_column):
    y = data[target_column]
    X = data.drop(columns=[target_column])
    return train_test_split(X, y, test_size=0.30, random_state=42)

X_train, X_test, y_train, y_test = split_data(house_data, 'SalePrice')

# Preprocess numeric features
def preprocess_numeric_features(data, columns):
    std_scaler = StandardScaler()
    data_scaled = std_scaler.fit_transform(data[columns])
    return pd.DataFrame(data_scaled, columns=columns)

X_train_num_scaled = preprocess_numeric_features(X_train, ['GrLivArea', 'LotArea'])
X_test_num_scaled = preprocess_numeric_features(X_test, ['GrLivArea', 'LotArea'])

# Preprocess categorical features
def preprocess_categorical_features(data, columns):
    le = LabelEncoder()
    data_encoded = le.fit_transform(data[columns])
    ohe = OneHotEncoder(sparse=False)
    data_ohe = ohe.fit_transform(data[[columns[0]]])
    return pd.DataFrame(data_ohe, columns=ohe.get_feature_names_out([columns[0]]))

X_train_cat_ohe = preprocess_categorical_features(X_train, ['Street'])
X_test_cat_ohe = preprocess_categorical_features(X_test, ['Street'])

# Combine processed features
def combine_features(num_scaled, cat_ohe):
    return pd.concat([num_scaled, cat_ohe], axis=1)

X_train_processed = combine_features(X_train_num_scaled, X_train_cat_ohe)
X_test_processed = combine_features(X_test_num_scaled, X_test_cat_ohe)

# Train and evaluate models
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return lr, rmse

model1, rmse1 = train_and_evaluate_model(X_train_num_scaled, y_train, X_test_num_scaled, y_test)
print("Model #1 :  Only 2 numeric features")
print("Intercept:", model1.intercept_)
print("Coefficient:", model1.coef_)
print('The RMSE value is {:.4f}'.format(rmse1))

model2, rmse2 = train_and_evaluate_model(X_train_processed, y_train, X_test_processed, y_test)
print("\nModel #2 : 2 numeric and 1 categorical features")
print("Intercept:", model2.intercept_)
print("Coefficient:", model2.coef_)
print('The RMSE value is {:.4f}'.format(rmse2))

# Define custom transformers
class MyTransformer(TransformerMixin, BaseEstimator):
    '''A template for a custom transformer.'''
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X

# Pipeline with numeric features only
num_pipe = Pipeline([('std_scaler', StandardScaler())])
X_test_num_scaled_new = num_pipe.fit_transform(X_test[['GrLivArea', 'LotArea']])
print("Processed Test Data - numeric feature:\n", X_test_num_scaled_new[0:3])

# Pipeline with numeric and categorical features
num_pipe_model = Pipeline([('std_scaler', StandardScaler()),
                           ('lr', LinearRegression())])
num_pipe_model.fit(X_train[['GrLivArea', 'LotArea']], y_train)
y_pred = num_pipe_model.predict(X_test[['GrLivArea', 'LotArea']])
rmse3 = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nModel #1 :  Only 2 numeric features using pipeline")
print('The RMSE value is {:.4f}'.format(rmse3))

# Define custom feature extractor
class MyFeatureExtractor(TransformerMixin, BaseEstimator):
    '''This customer tranformer extracts and returns specific feature.'''
    
    def __init__(self, feature):
        self.feature = feature
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[[self.feature]]

# Feature union
COLS_1 = ['GrLivArea', 'LotArea', 'YearBuilt']
COLS_2 = ['OverallQual', 'OverallCond']

my_pipelines = FeatureUnion([('std', Pipeline([('cols1', MyFeatureExtractor(COLS_1)),
                                              ('std', StandardScaler())])),
                             ('minmax', Pipeline([('cols2', MyFeatureExtractor(COLS_2)),
                                                 ('minmax', MinMaxScaler())]))])

output = my_pipelines.fit_transform(house_data)
print(output[0:3])
