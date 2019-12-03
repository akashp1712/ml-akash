import pandas as pd

## Getting the data ##

# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
print(melbourne_data.describe())

# print a columns names (features) of the data in Melbourne data
print(melbourne_data.columns)

# drop the data with the missing values
# dropna drops missing values
melbourne_data = melbourne_data.dropna(axis=0)


## Choosing features ##

# select the prediction target
y = melbourne_data.Price

# choosing features
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
# Here, train - training data, val - validation data
#

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


## Experimenting With Different Models ##

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))
