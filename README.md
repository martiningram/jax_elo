# Elo extensions in JAX

This code accompanies the paper "How to extend Elo: A Bayesian perspective", now [published in the Journal of Quantitative Analysis in Sports](https://www.degruyter.com/document/doi/10.1515/jqas-2020-0066/html).

The package allows you to fit the extensions to Elo to margin of victory and
correlated skills presented in the paper. In addition, it can be extended to other scenarios by defining a set of custom `EloFunctions`.

## Getting started

To get the repository to run, you will need to have the following requirements:

* Python 3.6+
* The python packages listed in `requirements.txt`

You can install these in the usual way, for example, by running `pip install -r
requirements.txt`.

With this done, you can install the package as usual by running `pip install -e
.`, or `python setup.py install`.

To run the tennis examples, you will additionally need to download Jeff Sackmann's ATP dataset, available here:

[Jeff's ATP data](https://github.com/JeffSackmann/tennis_atp)

## Running the tennis examples

There are two examples applying the models to tennis in the `examples` folder:

* The file "Margin model example.ipynb" walks through fitting an extension of
  Elo taking the margin of victory in account. The default settings use ATP data
  from 2010 onwards and the difference in the fraction of points won on serve as
  the margin.
* The file "Surface model example.ipynb" extends the margin model to also take
  correlated skills into account. It is fit to estimate surface-specific Elo
  ratings in tennis on clay, grass and hard courts since 2010.
  
Both models could also be used in other sports than tennis. For example, the model with correlated skills could be applied to chess to model correlations across different formats, e.g. speed chess with regular chess.

## Creating custom extensions

Digging deeper, if you like you can create your own custom Elo extension. This
requires specifying a set of `EloFunctions`. The final example, `Best of Five
extension example`, shows how a model taking into account the difference between
best of five and best of three matches in tennis can be implemented. This is
more work than using one of the pre-defined models but allows for more
flexibility.
