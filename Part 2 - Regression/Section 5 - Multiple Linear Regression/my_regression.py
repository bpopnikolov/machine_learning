# Data Preprocessing Template

# Importing the libraries
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LinearRegression


def backward_elimination(y, X, vectors, sl):
    X_opt = X[:, vectors].astype(float)

    regressor_OLS = smf.OLS(endog=y, exog=X_opt).fit()

    while not all([x < sl for x in regressor_OLS.pvalues]):
        # print([x < 0.05 for x in regressor_OLS.pvalues])
        max_p = max(regressor_OLS.pvalues)
        vector_to_remove = [i for i, x in enumerate(
            regressor_OLS.pvalues) if x > sl and x == max_p]

        vectors = [x for i, x in enumerate(
            vectors) if i != vector_to_remove[0]]
        X_opt = X[:, vectors]
        # print(vector_to_remove[0])
        print(vectors)

        regressor_OLS = smf.OLS(
            endog=y, exog=X_opt.astype(float)).fit()

    return {'regressor': regressor_OLS, 'x': X_opt}


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data

le_X = LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])

colt = make_column_transformer(
    (OneHotEncoder(categories='auto'), [3]), remainder='passthrough')
X = colt.fit_transform(X)

X = X[:, 1:]
# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

vectors = [0, 1, 2, 3, 4, 5]

model = backward_elimination(y, X, vectors, 0.05)
print(model.get('regressor').summary())

test_pred = model.get('regressor').predict(model.get('x'))
print(test_pred)
print(len(test_pred))
