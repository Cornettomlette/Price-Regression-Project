**PREDICTING SELLING PRICES OF HOUSES IN SEATTLE, USA**

W5 - PROJECT

**Background**: For a real estate company, build a machine learning model to predict the selling prices of houses based on a variety of features on which the value of the house is evaluated.

**Objective**: The task is to build a model that will predict the price of a house based on features provided in the dataset. The senior management also wants to explore the characteristics of the houses using some business intelligence tool. One of those parameters include understanding which factors are responsible for higher property value - \$650K and above.

**Data**: The data set consists of information on some 22,000 properties.  The dataset consisted of historic data of houses sold between May 2014 to May 2015.

**Methodology**: The dataset was first imported into MySQL and an initial data exploration was done by means of SQL queries. After having a clear overview, the dataset was imported to Jupyter Notebook by establishing the connection with MySQL. After an initial check for NA and Null values, we divided the data into numerical and categorical. Using the method describe on numerical columns I got a numerical overview of the dataset. We then plotted a 'Pearson' Correlation Matrix to check the correlation between the features of the dataset. 

While checking for outliers, we saw a potential problem with outliers limiting the predictive power of the model. We checked the distributions of continuous variables by using matplotlib and seaborn, and found that all variables were positively skewed, so we transformed the data. This improved the distributions dramatically. The next step was to dummify categorical columns and concatenate the data. We went on to divide the data between the dependant and independant variables, and to split between train and test sets. We initially tried a Linear Regression Model, a KNN model, and a Decision Tree Regressor.


**Model 1 - Anna-Mariia**

My first step was to try some models on the raw data, just to have a starting point. I chose the **Linear model, KNN model** (with the value of k from 2 to 10) and the **XGBooster decision tree regressor**. The whole time these three models were the ones on which I checked how the things I did with data affect the performance of the model.  As on raw data the performance of XGBooster was the best, I decided to stick with it and try to improve it’s performance. And as we know, the main problem to overcome were the outliers of the target carriable, which the models failed to predict correctly.
The things I tried to do were:
    1) **Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise** (https://github.com/nickkunz/smogn) - it was meant to deal with outliers by oversampling them and undersampling the values, which formed the majority of data. After some trials with changed parameters, the method was unfortunately unable to improve the performance of a model. Maybe this method required more trials and trials
    2) **Hyperparameters tuning of the XGBooster model** to improve MAE – I found some parameters which helped minimize the MAE, and, respectively, the mean absolute percentage error, so the model was able to predict the actual values with more precision. But still this method never help with making the outliers more visible to the model and it didn’t improve the R2 score.
    3) Also tried to check the **MLP Regressor**, and the performance was horrible, because the objective error metrics of the model is the MSE,  which is very sensitive to outliers, which is our problem.
So, my final actions with the model was dropping irrelevant cross-corelated columns, box-cox transformation for the continuous variables and log transformation of the target variable. I used the XGBoost model and applied some parameters to improve MAE. The final score was R2 = 0.88 and MAPE of 12.48.

**Model 2 - Ernesto**

I got the best results with the linear regression and decided to stick with it (allowing my colleagues to work with the other models). I went back and did some feature engineering, creating a column to classify between houses with basement and houses without basement. I also dropped columns which I found irrelevant or highly correlated with other features. These two actions improved the model significantly.

**Model 3 - Macarena**

After doing feature engineering and classifying the houses, we were not able to increase further the performance of the model. I decided to run several models to predict the prices of the houses that were limiting the predictive power of our model. However, when conducting the predictions, our sample became smaller and smaller, the outliers were harder to predict and the overall model performance did not improve. I then attempted to classify the houses according to a location parameter, which if explored further could potentially lead to a better performance. 


