This is the third and final capstone project for the Data Science Career Track bootcamp by Springboard. 

The problem statement had us working at an in-house trading desk that makes investments in cryptocurrencies. One of the
larger portfolios is in Ethereum, the second largest crypto by market cap. The firm is wondering if it
should increase or decrease its position size with a one-year outlook in mind. You are tasked with
determining which direction Ethereum will be one year from now, and to what magnitude.

https://www.kaggle.com/varpit94/ethereum-data
Our dataset consisted of 2244 rows and 7 columns for the daily Ethereum price ranging from
2015 to 2021. We found only 4 rows consisting of missing data and handled these using a simple
forward fill imputation. We dropped the adjusted close column since this is only applicable for
traditional securities that trade during specific hours, whereas cryptocurrencies trade around the
clock and 7 days a week. We decided to create a singular “Price” feature to use in our models which
was the average mean of the open and close prices for each day.

We tested several time series analysis models, and after tuning each we decided on the best model based on a set of performance metrics. 
The final model chosen was the FB Prophet model.
FB Prophet has built-in functionality to run time series models with cross validation train-test splits,
which we did with an initial 550-day train period with 135 periods and a horizon of 275 days, roughly
equivalent to 80/20 split. Using this we performed hyperparameter tuning on changepoint_prior_scale
and seasonality_prior_scale parameters. After determining our best parameters, our forecast error
improved to an MSE of 0.0327 and MAE of 0.1433. Our 1-year price prediction forecast was $35,024, or
an 1039.3% increase from current prices.
