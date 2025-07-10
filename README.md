# OMSA-Practicum
This is where I have hosted all my files for the OMSA Practicum final submission!

Below is a description of all the files:

Modeling.py – The output of this file provides the answers to the 4 key questions I laid out at the beginning of this paper. The packages used are numpy, pandas, scikit-learn, seaborn, and matplotlib. With them, this file:
	
	-Loads the data
	-Trains the models (tuning hyperparameters for random forest)
	-Prints out interpretations of the coefficients for the logistic regression
	-Prints out feature importance rankings for the random forest
	-Calculates surprise firings and unexpected survivors from inactive coaches subset
	-Plots surprise firings and unexpected survivors
	-Calculates most at-risk coaches and safest coaches from active coaches subset

Data Exploration.py – This file produces most of the data used in the “Exploratory Data Analysis” section. The packages used are pandas and scipy. With them, this file:

	-Loads the data
	-Prints contingency tables for fired vs all binary predictors
	-Runs Chi-squared independence test for fired vs all binary predictors
	-Calculates point-biserial correlation for fired vs all numeric predictors
	-Prints contingency table and runs independence test for fired_midseason vs minority
	-Calculates correlation for win_pct vs final_yr_win_pct

VIF.py - This is a very simple file that only exists to calculate the variance inflation factor for each of the predictor variables. The packages used are pandas and statsmodels.

	-Loads the data
	-Calculates and prints the VIF for every predictor

Tenure Histogram.py - This is another simple program that only serves one function. It filters the data set by inactive coaches only and then plots their tenure lengths in a histogram. The packages used are pandas, matplotlib, and seaborn.

	-Loads the data
	-Filters out active coaches
	-Plots remaining coaches (inactive only) in a histogram by tenure length

NFL Coaches Data Set.csv - This is the data set I created for this project.

Sayom Ghosh-Dastidar OMSA Practicum Final Report.pdf - This file contains my final report.
