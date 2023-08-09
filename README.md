# Solar-power-system-coverage-prediction

This repository contains R code for predicting solar system coverage based on the deep solar dataset. The code utilizes various classification algorithms, including logistic regression, decision trees, and random forests, to analyze patterns and provide insights into solar system adoption.

**Introduction:**

The demand for renewable energy has led to increased attention on photovoltaic technology as it provides an environmentally-friendly way of generating power. Adopting efficient solar power systems can positively impact the environment and regional economic growth. This report aims to analyze the performance of different machine learning algorithms for classifying the potential of solar energy generation based on various features. The study compares the effectiveness of logistic regression, classification tree, and random forest algorithms, and also evaluates the impact of tuning hyper parameters in improving their performance.

**Prerequisites:** Ensure you have R and the required libraries (ROCR, rpart, partykit, rpart.plot, caret, randomForest) installed.

**Dataset:** Place data_hw3_deepsolar.RData in the same directory as the R script.

**Running the Code:** On executing the code, the script performs the following tasks:
1. Loads the dataset and prepares the target variable.
2. Splits the data into training and test sets.
3. Fits a logistic regression model to predict solar system coverage.
4. Evaluates the model's performance using ROC curve and optimal threshold (tau).
5. Implements classification algorithms (logistic regression, decision trees, random forests) using k-fold cross-validation.
6. Computes and visualizes the mean accuracy of each classifier.

**Data Description:**

The data used for this study is a subset of the DeepSolar database, consisting of 10,926 rows and 15 features. The data includes several features such as the coverage of solar power systems, average household income, employment rate, population density, total number of housing units, median housing unit value, ratio of vacant housing units, ratio of house units using gas, electricity, or oil as heating fuel, total land area, total water area, air temperature, earth temperature, and daily solar radiation. The target variable for this study is the coverage of solar power systems, which is a binary variable that takes value low if the tile has a low-to-medium number of solar power systems, and high if the tile has more than ten solar power systems.

**Data Preprocessing:**

The data was checked for missing values and outliers. No missing values were observed in the dataset. We converted the solar_system_coverage variable to a binary variable, with "high" being coded as 1 and "low" as 0.

**Data Preparation:**
To start building our models, we need to divide our data into two sets: training and testing. We'll split the data using a ratio of 80/20, meaning we'll allocate 80% of the data for training and 20% for testing.

**Model Building:**

Three different supervised learning models were used to predict the coverage of solar power systems in a given tile area. The models used were 
•	Logistic regression
•	Decision tree
•	Random forest

*Logistic Regression:*

Logistic regression is a statistical model used for binary classification problems, where the output variable takes only two values. The logistic function maps any input value to a value between 0 and 1, which is interpreted as the probability of the input value belonging to the positive class. The logistic regression model was trained using a training dataset and evaluated using a validation dataset. 

*Decision tree:*

Decision trees are a popular algorithm for classification that is easy to understand and interpret. The tree structure is constructed by recursively splitting the data based on the value of the independent variables. The decision tree model was trained using a training dataset and evaluated using a validation dataset. 

*Random forest:*

A random forest is an ensemble method that uses multiple decision trees to improve the accuracy of the classification. Each tree in the forest is built on a randomly selected subset of the data and a randomly selected subset of the features. The random forest model was trained using a training dataset and evaluated using a validation dataset. 

**Model Evaluation and Comparison:**

The target variable classes are imbalanced, with 90% being low and 10% high, so for an appropriate assessment we report multiple metrics such as accuracy, sensitivity, and specificity. Cross-validation technique is used to assess the performance of a predictive model. The goal of cross-validation is to estimate how well a model will generalize to new, unseen data. Cross-validation can help to prevent over fitting, which occurs when a model is overly complex and performs well on the training data but poorly on new, unseen data. By evaluating the model on multiple validation sets, cross-validation provides a more robust estimate of the model's generalization performance. We implement a cross-validation procedure for three classification models: random forest, logistic regression, and classification trees. The cross-validation is performed using K-fold validation. The models were evaluated using 5-fold cross-validation. The data were randomly partitioned into 5 folds of roughly equal size, and each fold was used once as a validation set while the other 4 folds were used as a training set. The cross-validation process was repeated once to obtain a more reliable estimate of the performance metrics. Additionally, the code performs two replicates of the cross-validation procedure. The results of the cross-validation are stored in lists, which contain accuracy, sensitivity, and specificity measures for each model, and the best hyper parameters for the random forest, logistic regression and classification tree models are selected.

**LOGISTIC REGRESSION:**

The first method used is logistic regression. We use the glm() function with family binomial to build the model and fit it to the training set. A summary of the model is obtained using the summary() function, which provides information on the coefficients of the features used and their significance levels. All features except water area have significant effects on the target variable. We use all features to fit the logistic regression model, but water area can be excluded as it has less effect on the target variable. We predict the target variable for the validation set using the logistic regression model, with a threshold of 0.5 and an optimal tau value obtained from a sensitivity and specificity graph. We measure the accuracy of the model by calculating the proportion of correct predictions in the validation set and create a confusion matrix to measure sensitivity and specificity. The optimal threshold value for logistic regression is found to be 0.10, which gives better accuracy (92.64%) and sensitivity (71.52%) than a threshold of 0.5.

We also use ROC curves to evaluate the performance of the logistic regression model, which plot sensitivity versus (1 - specificity) for different threshold values. The area under the curve is a measure of the model's discrimination ability, with a value of 1 indicating perfect discrimination and 0.5 indicating a random classifier. We choose the optimal threshold value based on the point on the ROC curve that maximizes the sum of sensitivity and specificity.

**ClASSIFICATION TREE:**

The second approach used is classification tree, which involves building a tree-based model using the rpart() function. This model is used to make predictions for the target variable on the validation set, and is trained on the training set using K-fold cross-validation. The hyper-parameter cp is tuned to achieve the best fit for the model, and the model is fitted for various values of cp, with 5-fold cross-validation used to evaluate the model's accuracy, sensitivity and specificity for each value of cp. The average of each measure is calculated to compare models with different cp values, and the model with the highest accuracy and sensitivity is selected. Accuracy is measured by determining the proportion of correct predictions in the test set. The results indicate that the model achieved the highest accuracy (93.8%) with a cp value of 0.004, while the highest sensitivity (82.19%) was achieved with a cp value of 0.014. Since the dataset is imbalanced with a majority of low to medium tile values, accuracy cannot be relied upon to evaluate model performance, as it may lead to overfitting. Therefore, sensitivity is considered the optimal measure to evaluate the performance of the model. Based on the sensitivity measure, the model with a cp value of 0.014 is considered the best fit for the classification tree


*Parameter tuning:*
For classification tree, the tuning parameter is cp, the complexity parameter that controls the tradeoff between tree size and goodness of fit. The tuning grid for cp consists of 101 values, ranging from 0 to 0.1 in increments of 0.001.


**RANDOM FOREST:**

The final approach utilized is random forest. To build the random forest model, the randomForest() function is used with the importance parameter set to TRUE. This parameter computes the importance of each feature in the model by measuring the mean decrease in impurity when the feature is excluded from the model. Next, the random forest model is applied to forecast the target variable for the validation set. To obtain the best fit of the model, it is trained and tested using different combinations of mtry and ntree via 5-fold cross validation, and accuracy, sensitivity, and specificity measures are used to evaluate each model. The average value of each metric is calculated to compare the different models, and the model with the highest accuracy and sensitivity for mtry and ntree is chosen. The accuracy of the model is determined by calculating the proportion of correct predictions in the test set.
Parameter tuning:
For random forest, the tuning parameters are mtry, the number of randomly selected predictor variables to consider at each split, and ntree, the number of trees in the forest. The tuning grid for mtry consists of 4 values: 3, 4, 5, and 6.The tuning grid for ntree consists of 5 values: 75, 100, 125, 150, and 200. The total number of parameter combinations to evaluate is thus 20.

The "grid_acc_rf," "grid_sens_rf," and "grid_spec_rf" data frames show the average accuracy, sensitivity, and specificity of the random forest model for different combinations of "mtry" and "ntree. The results showed that the accuracy(95% approx) and specificity(95.5%) was highest with mtry of 5 and ntree of 125. However, as the data is imbalanced with most of the data having low to medium tile, accuracy measure is not considered as the appropriate measure in evaluating the performance of the model as the model tends to over fit the data. The sensitivity of the model is highest with mtry of 3 and ntree of 125 having the sensitivity of 86.7%. Hence the optimal mtry and ntree value for fitting random forest would be 3 and 125.

**RESULT:**

Based on the evaluation metrics and analysis, we can conclude that the Random Forest classifier with 3 randomly selected predictor variables(mtry) and 125 trees in the forest(ntree) outperformed compared to other two models with optimal hyperparameters for predicting high solar power system coverage in the given dataset. It achieved an accuracy of 95% approximately on validation dataset and 54% approximately on the test dataset.

**CONCLUSION:**
In this study, we aimed to build a model to predict the coverage of solar power systems in a given tile based on various social, economic, housing, geographical, and meteorological features. We explored three different classification algorithms, namely Logistic Regression, Decision Tree, and Random Forest, and evaluated their performance using various metrics such as accuracy, sensitivity and specificity. Our experimental results showed that the Random Forest algorithm outperformed the other two algorithms, achieving an accuracy of 0.95, sensitivity of 0.866, specificity of 0.955 on validation dataset and accuracy of 0.9446, sensitivity of 0.536, specificity of 0.99 on test dataset.





