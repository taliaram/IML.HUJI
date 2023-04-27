from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model

    samples = np.random.normal(10, 1, 1000)  # New array of 1000 normal random variables ~ N(10,1)

    univariate_model = UnivariateGaussian()  # New univariate object

    univariate_model.fit(samples)  # Performs fit function on the samples array

    print(univariate_model.mu_, univariate_model.var_)  # Prints the fitted results

    # Question 2 - Empirically showing sample mean is consistent

    # list comprehension of the losses while the size of samples is 10, 20, 30, ... ,1000:
    losses = [np.abs(10 - univariate_model.fit(samples[0:i]).mu_) for i in range(10, 1001, 10)]

    # Plotly express line graph while the number of samples is the X axis and loss is Y axis:
    px.line(x=np.arange(10, 1001, 10), y=np.array(losses),
            labels=dict(x="Number of samples", y="Loss"), title="Losses as function of sample size").show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf_of_samples = univariate_model.pdf(samples)  # Computing pdf on each original sample

    # Plotly express scatter graph while the samples are the X axis and their PDFs are the Y axis:
    px.scatter(x=samples, y=pdf_of_samples,
               labels=dict(x="Samples", y="PDF"), title="PDF of 1000 ~N(10,1) samples").show()



def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model

    # Given inputs:
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    samples = np.random.multivariate_normal(mu, cov, 1000)  # New array of 1000 samples with given mu and cov

    multivariate_model = MultivariateGaussian()

    multivariate_model.fit(samples)

    print(multivariate_model.mu_)
    print(multivariate_model.cov_)

    # Question 5 - Likelihood evaluation

    linspace_values = np.linspace(-10, 10, 200)

    # List comprehension:
    log_likelihoods = [[MultivariateGaussian.log_likelihood(
        np.array([linspace_values[f1], 0, linspace_values[f3], 0]), cov, samples) for f1 in range(200)]
        for f3 in range(200)]

    log_likelihoods = np.array(log_likelihoods)  # Converts to np array

    px.imshow(log_likelihoods.T, x=linspace_values, y=linspace_values,
              labels=dict(x="Columns - f3", y="Rows - f1"),
              title="Likelihoods as function of f3 and f1").update_yaxes(autorange=True).show()

    # Question 6 - Maximum likelihood

    res_index_x, res_index_y = np.where(log_likelihoods == np.max(log_likelihoods))
    print("Maximum is: ", np.max(log_likelihoods))
    print("At coordinates: ", np.round(linspace_values[res_index_x][0], 3),
          np.round(linspace_values[res_index_y][0], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
