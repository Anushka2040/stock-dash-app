## Stock Prediction App

*It is a single-page web application using Dash (a python framework) and some machine learning models which will show company information (logo, registered name and description) and stock plots based on the stock code given by the user. Also the ML model will enable the user to get predicted stock prices for the date inputted by the user.

*Used the yfinance python library to get company information and stock price history. Dash's callback functions will be used to trigger updates based on change in inputs.

*Built a machine learning model - Support Vector Regression (SVR) for predicting the stock prices. For the dataset, fetched stock price from last 60 days and splitted it into 9:1 ratio for training and testing respectively.

*Tested the model's performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the testing dataset.
