# CPS3320-final-project
first load the data from the data set.Then, check the total number (and percent) of missing values in each columns. Then, we need to treat some outliers. Then just treat the data with the graph to show the trend and pick the one colomn which is good for prediction.
Then check stationarity.
Then check Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for the first difference of the data and then draw the graph.
From the graph, we can get p, q, and d's value. 
Then take these value into the ARIMA time series model by using the training data set. 
Then see the graph of the prediction to see the accuracy.
Then using the test data set to predict to value and see its accuracy.
If you want to see the detail, just run the code. However, you should make sure that you have used the right statemodel library because the newest library does not fit with some of the functions in the training part and testing set. For more information, you can go to the website I provide in the code comment which is a website for the statemodel library, if you cannot run the code, you may find some helpful information there.
