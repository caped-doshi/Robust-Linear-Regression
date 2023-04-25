# Robust-Linear-Regression

A python implementation of Subquantile Minimization for Robust Linear Regression.

## robust-linear-regression directory files
The following are the different baselines and our proposed method for robust-linear regression
- `SubQuantile.py`: This is our proposed method
- `smart.py`: [NeurIPS 2022] Trimmed Maximum Likelihood Estimation for Robust Learning in Generalized Linear Models.
- `term.py`: [ICLR 2021] Tilted Empirical risk minization to solve ridge regression.
- `RRM.py`: [IEEE 2020] Robust Risk Minization for Statistical Learning.
- `sever.py`: [ICML 2019] A robust meta-algorithm for stochastic optimization used in ridge regression.
- `Stir.py`: [AISTATS 2019] Reweighted Least Squares for Robust Regression.
- `CRR.py`: [NeurIPS 2017] Consistent Robust Regression used to solve the Robust Least Squares Problem.
- `sklearn_methods.py`: Various sklearn methods such as Huber, RANSAC, and Ridge. 
- `data_loader.py`: Contains the methods to load the datasets for the methods above. 
- `noise.py`: Contains the methods to add different types of noise to the features and/or labels.

## paper directory
Contains latex for the final report

## talk directory
Contains latex for the slides
