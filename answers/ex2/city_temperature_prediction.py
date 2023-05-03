import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.fillna(0)
    df = df[df["Temp"] > -50]
    df = df[df["Temp"] < 70]
    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"].astype({"Year": "str"})
    figure = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year",
                        title="Israels temperature per day of the year")
    figure.update_xaxes(title_text="Day Of Year")
    figure.update_yaxes(title_text="Temperature")
    figure.write_image("./ex2_graphs/temperature_per_day_of_year.png")

    figure = px.bar(x=np.arange(1, 13), y=israel_df.groupby(["Month"])["Temp"].std(),
                    title="Standard deviation of the temperature of each month")
    figure.update_xaxes(title_text="Month")
    figure.update_yaxes(title_text="Temperature std")
    figure.write_image("./ex2_graphs/monthly_std_temperature.png")

    # Question 3 - Exploring differences between countries
    figure = px.line(df.groupby(["Country", "Month"], as_index=False).agg(average=("Temp", "mean"),
                                                                          std=("Temp", "std")),
                     x="Month", y="average", error_y="std", color="Country",
                     title="Monthly average temperature and std by country")
    figure.update_xaxes(title_text="Month")
    figure.update_yaxes(title_text="Average Temperature")
    figure.write_image("./ex2_graphs/monthly_average_temperature.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df["DayOfYear"], israel_df["Temp"])

    loss_per_k = np.empty(shape=(10, ))

    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_X, train_y)
        loss_per_k[k-1] = np.round(model.loss(test_X, test_y), 2)
        print(f"Loss for degree of {k} is: {loss_per_k[k-1]}")

    figure = px.bar(x=np.arange(1, 11), y=loss_per_k, text=loss_per_k, title="Loss per polynomial degree")
    figure.update_xaxes(title_text="Degree")
    figure.update_yaxes(title_text="Loss")
    figure.write_image("./ex2_graphs/polynomial_degree_losses.png")

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5)
    model.fit(israel_df["DayOfYear"], israel_df["Temp"])

    countries = ["Jordan", "South Africa", "The Netherlands"]
    error_per_country = pd.DataFrame([{
        "Country": country, "Error": np.round(model.loss(df[df["Country"] == country]["DayOfYear"],
                                                         df[df["Country"] == country]["Temp"]), 2)
    }for country in countries])

    figure = px.bar(error_per_country, x="Country", y="Error", color="Country", text="Error",
                    title="Error per country for a model fitted with Israels data")
    figure.update_xaxes(title_text="Country")
    figure.update_yaxes(title_text="Error")
    figure.write_image("./ex2_graphs/country_errors_israel_model.png")
