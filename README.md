# Solar-power-system-coverage-prediction

This repository contains R code for predicting solar system coverage based on the deep solar dataset. The code utilizes various classification algorithms, including logistic regression, decision trees, and random forests, to analyze patterns and provide insights into solar system adoption.

**Prerequisites:** Ensure you have R and the required libraries (ROCR, rpart, partykit, rpart.plot, caret, randomForest) installed.

**Dataset:** Place data_hw3_deepsolar.RData in the same directory as the R script.

**Running the Code:** On executing the code, the script performs the following tasks:
1. Loads the dataset and prepares the target variable.
2. Splits the data into training and test sets.
3. Fits a logistic regression model to predict solar system coverage.
4. Evaluates the model's performance using ROC curve and optimal threshold (tau).
5. Implements classification algorithms (logistic regression, decision trees, random forests) using k-fold cross-validation.
6. Computes and visualizes the mean accuracy of each classifier.

**Results and Interpretation:** This project's main goal is to predict solar system coverage. By employing a range of classification algorithms, the code calculates accuracy and provides valuable insights into model performance. Importantly, the analysis uncovers that the random forest classifier exhibits superior predictive capabilities when contrasted with alternative techniques. This observation gains further support through the visualization of mean accuracy and standard deviation, as depicted in the box plots, effectively emphasizing the heightened efficacy of the random forest approach.

