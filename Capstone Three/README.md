![thumbnail](https://user-images.githubusercontent.com/80617423/150162453-ac6ae709-f008-4ff1-b194-bf508fe9137b.jpg)

Ethereum: Price Prediction Analysis Final Report
By: Nicholas Roller
1.	Problem Statement
You work for an in-house trading desk that makes investments in cryptocurrencies. One of the larger portfolios is in Ethereum, the second largest crypto by market cap. The firm is wondering if it should increase or decrease its position size with a one-year outlook in mind. You are tasked with determining which direction Ethereum will be one year from now, and to what magnitude.
2.	Dataset – Cleaning & Wrangling
https://www.kaggle.com/varpit94/ethereum-data
Our dataset consisted of 2244 rows and 7 columns for the daily Ethereum price ranging from 2015 to 2021. We found only 4 rows consisting of missing data and handled these using a simple forward fill imputation. We dropped the adjusted close column since this is only applicable for traditional securities that trade during specific hours, whereas cryptocurrencies trade around the clock and 7 days a week. We decided to create a singular “Price” feature to use in our models which was the average mean of the open and close prices for each day.
3.	Exploratory Data Analysis
We first wanted to look at any seasonal trends or outliers, so we plotted boxplots aggregated by month for our entire dataset. 
 ![image](https://user-images.githubusercontent.com/80617423/150161928-ff226d04-9763-4364-8c31-520d2b55d5df.png)

We saw an interesting trend where during the summer months, June and July, price fluctuations were confined to a much smaller range, inter-quartile range, but did tend to have outliers. Where as the closer it was to the end of the year, prices could see a lot more volatility. This indicates that we could expect to see a fairly consistent seasonality throughout the year.
	We then wanted to get a wholistic view of our price data, so we plotted our timeseries.
 ![image](https://user-images.githubusercontent.com/80617423/150161954-52f27df5-285f-42b5-aa49-00d3708d9aa8.png)

	Immediately we noticed that since prices varied on such a massive scale, it may be better to view this plot on a logarithmic scale.
 ![image](https://user-images.githubusercontent.com/80617423/150161967-5b84c1e6-b30f-4bf7-945a-3f9100c4c277.png)

	To get an idea of the smoother trends within our timeseries, we plotted the log price on top of a weekly and monthly resample from Jan 2020 to present:
 ![image](https://user-images.githubusercontent.com/80617423/150161985-dfc86eb4-c55b-46f7-ba2a-b07ba486879c.png)

	The weekly resampled plot looked ideal as it still exhibited more detailed trends without as much of the noise. Our goal is to predict a price forecast one year in the future, after all, so we won’t be too worried about the more fine-grained movements. 
	We wanted to see how volume played in, so we plotted the weekly price aggregates in parallel with volume:
 ![image](https://user-images.githubusercontent.com/80617423/150162016-71020da2-467d-4ae7-b974-f943b37cedf0.png)

	We noticed a fairly exponential increased in volume as well as price.
	Finally, we wanted to see how exponential smoothing affected our trend, visually, before finally moving onto preprocessing:
 ![image](https://user-images.githubusercontent.com/80617423/150162034-3aba4769-caec-4f6e-81dc-faa3e85d7621.png)

	Looking at the lag plots for Price and LogPrice it is apparent that both have a strong correlation with their t-1 lags.
 ![image](https://user-images.githubusercontent.com/80617423/150162045-032057f8-25ca-419c-a190-8e7971dbfb2c.png)

LogPrice Lag plot: 
![image](https://user-images.githubusercontent.com/80617423/150162089-1e555c5d-c3a7-412f-9a5b-bbf451bb8c5a.png)

4.	Preprocessing
The preprocessing stage was fairly straightforward as we were analyzing a univariate time series. We reduced our data frame down into our key components, the date and price column. We converted our date column to a datetime index and set it as the index to our data frame. We create our final dependent variable data frame by taking the weekly aggregate mean of our Price column through the df.resample() method. Finally, established our train and test splits of our dataset using the following code:
split=int(len(ETH['Price']) * 0.8) 
y_train, y_test = ETH['Price'][0:split], ETH['Price'][split:len(ETH['Price'])]
5.	Final Model & Recommendations
The first key step before we actually model our time series is to establish stationarity. We can do this through a number of methods, such as taking the difference of the time series or taking it’s log, or a combination. We wrote a function to take an input timeseries and plot the timeseries and it’s rolling mean and standard deviations, so we can visually see if it is stationary. In addition, it calculates and outputs the results of the Dickey-Fuller test. Below is the output for the timeseries without any modifications: 
 ![image](https://user-images.githubusercontent.com/80617423/150162113-00a613d3-ba8d-4d90-8e2f-0a5aae761f94.png)

Our p-value is above 0.05 and our test statistic is not smaller than our critical value, therefore we cannot reject the null-hypothesis of non-stationarity.
We then take the difference of y and test again:
 ![image](https://user-images.githubusercontent.com/80617423/150162131-181f914e-e4c6-4fbc-a6da-3eb9d10a22d6.png)

Again, the same, so we continue modifying our timeseries until we achieve stationarity. This time we simply take the log of y and test again:
 ![image](https://user-images.githubusercontent.com/80617423/150162149-c37a60e6-fe92-4942-bd40-d73733ea1da4.png)

Still not quite stationary yet. Finally, we try to combine these two methods taking the log transform of y and its difference:
 ![image](https://user-images.githubusercontent.com/80617423/150162165-191ac701-d97c-44d4-b7f8-6d3b90bf5846.png)

We have finally achieved a p-value below 0.05 and a test statistic that is more negative than our critical value. We can also see visually that our timeseries does not have a trend and that the rolling mean and standard deviations remain constant from our plot.
To get an idea of the three primary components, trend, seasonality and residual, of our time series we plotted the decomposition of our y_log series:
 ![image](https://user-images.githubusercontent.com/80617423/150162190-4eee71a4-8901-4d2e-965e-4113a771b149.png)

We then tested the stationarity of our residuals to determine the Q coefficient for our ARIMA model and determined Q to be 1:
 ![image](https://user-images.githubusercontent.com/80617423/150162211-12789301-6704-4b38-b99d-88f6a82c29ae.png)

To find our P and R coefficients we plotted the ACF and PACF plots for our y_log_diff series:
 ![image](https://user-images.githubusercontent.com/80617423/150162231-83e0eadf-7067-480f-a1b9-00ceb0a1bd45.png)

From this we can guess 4 and 3 are likely going to be our best P and R values, respectively. Plugging these into our ARIMA model and adjusting them slightly we found the lowest RSS (residual sum of squares) to be 5.6154 with an ARIMA(4,1,3) model. 
 ![image](https://user-images.githubusercontent.com/80617423/150162247-770a20c7-ec20-4edd-a11f-f64c5ad49528.png)

We then defined an evaluate_arima_model function to iterate through a range of P, Q and R values to fit a model on our train set and take a timestep-wise comparison between our test data and one-set prediction and output the MSE. This is what we used to select our best ARIMA model. For this we used an 80/20 train/test split. 
With our fitted model we used the plot_predict() function to forecast 52 steps into the future, or 1 year.
 ![image](https://user-images.githubusercontent.com/80617423/150162270-b8c18b55-a654-473c-929e-bc92a2d85acb.png)

Analyzing our in-sample forecast metrics we found this model to have a mean square error (MSE) of 0.663 and mean absolute error (MAE) of 0.601. The final price predicted one year into the future was $9,070.15, or a 195% increase from the current price.
Our second model was the FB Prophet model. For this we simply restructured our data frame to have a ‘ds’ column containing our price data and a ‘y’ column with our dates. We fit the model and make a future model, for our forecast, with 52 (weekly) periods to forecast 1 year out-of-sample. Calling the plot_components() on our forecast plots our trend and yearly seasonality:
 ![image](https://user-images.githubusercontent.com/80617423/150162290-78efe93c-63f9-49db-9afe-5b7346555f80.png)

We then plot our forecast to see both the in-sample and out-of-sample forecasts:
 ![image](https://user-images.githubusercontent.com/80617423/150162299-50131ba1-8eb5-4d93-9d3a-eced56581306.png)

Our out-of-the-box FB Prophet model gives us an MSE of 0.149 and an MAE of 0.296 with a 1 year forecasted price of $27,856, or 806.1% increase from the current Ethereum price.
FB Prophet has built-in functionality to run time series models with cross validation train-test splits, which we did with an initial 550-day train period with 135 periods and a horizon of 275 days, roughly equivalent to 80/20 split. Using this we performed hyperparameter tuning on changepoint_prior_scale and seasonality_prior_scale parameters. After determining our best parameters, our forecast error improved to an MSE of 0.0327 and MAE of 0.1433. Our 1-year price prediction forecast was $35,024, or an 1039.3% increase from current prices. Below is the plotted components and forecast of our final model:
 ![image](https://user-images.githubusercontent.com/80617423/150162312-ff7b3388-0c7d-47bc-b4c3-9550079ea0a2.png)
 ![image](https://user-images.githubusercontent.com/80617423/150162332-e80fa7c9-3767-4f81-9907-074534301be7.png)

 
6.	Future Thoughts & Questions
Provided both of our models predicted an increase in the price of Ethereum one year from now, our suggestion would be that our trading firm increases their position in ETH with a one-year forecast.
As far as our model goes, we would suggest using the FB Prophet model for future forecasts. Although perhaps less interpretable than our ARIMA model, our initial testing shows it provides predictions with lower error metrics. That being said, both models could be improved given time and resources for further hyperparameter tuning. Additionally, Ethereum has historically been fairly correlated with the price of Bitcoin so a multivariate model could have the potential to produce a far stronger, albeit more complex, model.
