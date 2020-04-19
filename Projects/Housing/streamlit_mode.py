import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import streamlit as st

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(husing_path=HOUSING_PATH):
    csv_path = os.path.join(husing_path, "housing.csv")
    return pd.read_csv(csv_path)


def dprint(data):
    st.dataframe(data)

def pprint(value):
    st.markdown(value)

def sprint(value):
    st.write(value)


pprint("# Predicting Median House Value")

##################################################
pprint("## 1. Quick look at the data")

with st.echo():
    housing = load_housing_data(HOUSING_PATH)
    housing_bkup = load_housing_data(HOUSING_PATH)

pprint("### Top few rows")
dprint(housing.head())
housing.info()

pprint("### Describe the features")
sprint(housing.describe())

pprint("#### Dig into \"ocean proximity\"")
sprint(housing["ocean_proximity"].value_counts())

pprint("### Histogram of each feature values")
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20, 15))
st.pyplot(plt)

##################################################
pprint("## 2. Create the test set")

pprint("#### Medium income is the important feature to predict house price, "
       "we need to make sure the train and test set should contain good sampling")

with st.echo():
    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

pprint("#### Histogram of \'income_cat\'")
housing["income_cat"].hist()
st.pyplot(plt)

pprint("### Stratified shuffle and split")
from sklearn.model_selection import StratifiedShuffleSplit

with st.echo():
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

pprint("#### Remove the `income_cat` attribute")

with st.echo():
    for _set in (strat_train_set, strat_test_set):
        _set.drop("income_cat", axis=1, inplace=True)

##################################################

pprint("## 3.Discover and visualize the data to Gain Insights")

pprint("#### Plot some attributes")

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# st.pyplot(plt)


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
st.pyplot(plt)

pprint("### Looking for correlations")
with st.echo():
    corr_matrix = housing.corr()
sprint(corr_matrix["median_house_value"].sort_values(ascending=False))

pprint("## 4. Prepare the Data for Machine Learning Algorithms")
pprint("#### Get the clean training data set")

with st.echo():
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

pprint("#### Data cleaning")
pprint("### Use SimpleImputer to replace easch attributes's missing value with "
       "the median of that attribute")
from sklearn.impute import SimpleImputer

with st.echo():
    imputer = SimpleImputer(strategy="median")

pprint("### remove non-numerical attribute to compute median")
with st.echo():
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)

pprint("#### Transform the training set by replacing missing values using trained imputer")
with st.echo():
    X = imputer.transform(housing_num)  # returns numpy array
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)  # converting to Pandas DataFrame

pprint("#### Handling Text and Categorical attributes")

## Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pprint("#### Pipeline for numerical features")
with st.echo():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scalar', StandardScaler()),
    ])

pprint("### Handle numerical and non-numerical features at once using ColumnTransformer")
with st.echo():
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

with st.echo():
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    housing_prepared = full_pipeline.fit_transform(housing)

pprint("## 5.Select and Train a Model")

pprint("### Training RandomForestRegressor")
from sklearn.ensemble import RandomForestRegressor

with st.echo():
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)

from sklearn.model_selection import cross_val_score

with st.echo():
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)

with st.echo():
    def display_scores(scores):
        sprint("Scores:" + str(scores))
        sprint("Mean:" + str(scores.mean()))
        sprint("Standard deviation:" + str(scores.std()))

    display_scores(forest_rmse_scores)

pprint("# 6. Evaluate your system on the Test Set")

with st.echo():
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = forest_reg.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

sprint("Final RMSE: " + str(final_rmse))

########################################################################################
