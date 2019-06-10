# Data Preprocessing Template

# Importing the libraries
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
# or
# X[:, 1:3] = imputer.fit(X[:, 1:3]).transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable

le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])

colt = make_column_transformer(
    (OneHotEncoder(categories='auto'), [0]), remainder='passthrough')
X = colt.fit_transform(X)

# the below method is depricated and alternative to colt
# ohe = OneHotEncoder(categories=[[0]])
# X = ohe.fit_transform(X).toarray()

# Encoding the Dependent Variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_test)
print(X_train)
# print(dataset)
# print(y)

print(y_train, y_test)
