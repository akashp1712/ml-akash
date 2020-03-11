import pandas as pd

## Getting the data ##

melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)

# print a summary of the data in Melbourne data
print ("### Summary of the Data ###")
print(melbourne_data.describe())
print(melbourne_data.info())

# print a columns names (features) of the data in Melbourne data
print(melbourne_data.columns)
print("\n")

## Handling the missing values

print ("### Handling the missing values ###")
print ("## 1. drop the rows with the missing values.")
## 1. dropna
dropna_melbourne_data = melbourne_data.dropna(subset=["BuildingArea"], inplace=False)
print(dropna_melbourne_data.describe())
print("\n")

## 2. drop
print ("## 2. drop the entire attribute")
drop_melbourne_data = melbourne_data.drop("BuildingArea", axis=1, inplace=False)
print(drop_melbourne_data.describe())
print("\n")

## 3. fillna
print ("## 3. fill the missing values.")
median = melbourne_data["BuildingArea"].median()
melbourne_data["BuildingArea"].fillna(median, inplace=True)
print(melbourne_data.describe())
print("\n")

## 4. Imputer
print ("### Fill the missing values using imputer ")
melbourne_data = melbourne_data.select_dtypes(exclude=["object"]).copy()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(melbourne_data)

X = imputer.transform(melbourne_data)
melbourne_data_tr = pd.DataFrame(X, columns=melbourne_data.columns,
                                 index=melbourne_data.index)
print(melbourne_data_tr.describe())
print("\n")
