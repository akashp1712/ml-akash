import pandas as pd
import streamlit as st

def sprint(data):
    st.dataframe(data)

def pprint(value):
    st.markdown(value)


pprint("#### Change to show app in wide mode from the settings")

## Getting the data ##

melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)

# sprint a summary of the data in Melbourne data
pprint ("# Summary of the Data")
pprint ("## Describe the Data")
with st.echo():
    melbourne_data.describe()

sprint(melbourne_data.describe())
pprint ("\n\n")

# sprint a columns names (features) of the data in Melbourne data
with st.echo():
    melbourne_data.columns

#sprint(melbourne_data.columns)
pprint ("\n\n")

## Handling the missing values

pprint ("### Handling the missing values ###")
pprint ("## 1. drop the rows with the missing values.")
## 1. dropna
with st.echo():
    melbourne_data.dropna(subset=["BuildingArea"], inplace=False)
dropna_melbourne_data = melbourne_data.dropna(subset=["BuildingArea"], inplace=False)
sprint(dropna_melbourne_data.describe())
pprint("\n\n")

## 2. drop
pprint ("## 2. drop the entire attribute")
with st.echo():
    melbourne_data.drop("BuildingArea", axis=1, inplace=False)
drop_melbourne_data = melbourne_data.drop("BuildingArea", axis=1, inplace=False)
sprint(drop_melbourne_data.describe())
pprint("\n\n")

## 3. fillna
pprint ("## 3. fill the missing values.")
pprint ("### Fill the missing values by calculating median")

with st.echo():
    median = melbourne_data["BuildingArea"].median()
    melbourne_data["BuildingArea"].fillna(median, inplace=True)

median = melbourne_data["BuildingArea"].median()
melbourne_data["BuildingArea"].fillna(median, inplace=True)
sprint(melbourne_data.describe())
pprint("\n\n")

## 4. Imputer
pprint ("### Fill the missing values using imputer ")
with st.echo():
    melbourne_data = melbourne_data.select_dtypes(exclude=["object"]).copy()

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    imputer.fit(melbourne_data)

    X = imputer.transform(melbourne_data)
    melbourne_data_tr = pd.DataFrame(X, columns=melbourne_data.columns, index=melbourne_data.index)

melbourne_data = melbourne_data.select_dtypes(exclude=["object"]).copy()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(melbourne_data)

X = imputer.transform(melbourne_data)
melbourne_data_tr = pd.DataFrame(X, columns=melbourne_data.columns, index=melbourne_data.index)
sprint(melbourne_data_tr.describe())
pprint("\n")