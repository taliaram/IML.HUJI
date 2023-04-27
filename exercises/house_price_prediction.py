from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


features_mean = dict()  # A dictionary that keeps the sum of every feature

def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # Removes columns that don't affect the price of the house
    X = X.drop(["id", "date", "lat", "long"], axis=1)

    if y is not None:  # If in train:
        y = y.dropna()  # Removes all the prices that are nan (not a number)
        y = y[y > 0]  # Removes the prices that are negative
        X = X.loc[y.index]  # Removes from X the rows of indexes that we removed from y

        # We will take only these items that are valid
        features_mean["bedrooms"] = X[X["bedrooms"] >= 0]["bedrooms"].mean()
        features_mean["bathrooms"] = X[X["bathrooms"] >= 0]["bathrooms"].mean()
        features_mean["sqft_living"] = X[X["sqft_living"] > 0]["sqft_living"].mean()
        features_mean["sqft_lot"] = X[X["sqft_lot"] > 0]["sqft_lot"].mean()
        features_mean["floors"] = X[X["floors"] > 0]["floors"].mean()
        features_mean["yr_built"] = X[X["yr_built"] > 0]["yr_built"].mean()
        features_mean["sqft_living15"] = X[X["sqft_living15"] > 0]["sqft_living15"].mean()
        features_mean["sqft_lot15"] = X[X["sqft_lot15"] > 0]["sqft_lot15"].mean()

    X.fillna(0)  # Switches any item that is nan (not a number) to be 0

    # If these features are invalid (negative etc) we will switch the value to the average of that feature:
    X.loc[X["bedrooms"] < 0, "bedrooms"] = features_mean["bedrooms"]
    X.loc[X["bathrooms"] < 0, "bathrooms"] = features_mean["bathrooms"]
    X.loc[X["sqft_living"] <= 0, "sqft_living"] = features_mean["sqft_living"]
    X.loc[X["sqft_lot"] <= 0, "sqft_lot"] = features_mean["sqft_lot"]
    X.loc[X["floors"] <= 0, "floors"] = features_mean["floors"]
    X.loc[X["yr_built"] <= 0, "yr_built"] = features_mean["yr_built"]
    X.loc[X["sqft_living15"] <= 0, "sqft_living15"] = features_mean["sqft_living15"]
    X.loc[X["sqft_lot15"] <= 0, "sqft_lot15"] = features_mean["sqft_lot15"]

    # We will fix these features to be in their ranges (if they are too small or too big etc):
    X.loc[X["waterfront"] < 0, "waterfront"] = 0
    X.loc[X["waterfront"] > 1, "waterfront"] = 1

    X.loc[X["view"] < 0, "view"] = 0
    X.loc[X["view"] > 4, "view"] = 4

    X.loc[X["condition"] < 1, "condition"] = 1
    X.loc[X["condition"] > 5, "condition"] = 5

    X.loc[X["grade"] < 1, "grade"] = 1
    X.loc[X["grade"] > 13, "grade"] = 13

    X.loc[X["sqft_above"] < 0, "sqft_above"] = 0

    X.loc[X["sqft_basement"] < 0, "sqft_basement"] = 0

    X.loc[X["yr_renovated"] < 0, "yr_renovated"] = 0

    X = pd.get_dummies(X, columns="zipcode")  # A categorical feature (no logical order)

    return X, y  # Returns the data frame and the prices vector


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X.columns:

        if "zipcode" in feature:
            continue  # No need for scatter

        pearson = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))

        px.scatter(x=X[feature], y=y, trendline="ols", labels=dict(x=feature, y="Price"),
                   title=f"Graph showing correlation between {feature} and the house price, "
                         f"with the Pearson Correlation of {pearson}:").write_image(output_path + "/" +
                                                                                    feature + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    # Default is 0.75
    train_X, train_y, test_X, test_y = split_train_test(df.drop(["price"], axis=1), df["price"])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, "./ex2_graphs")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
