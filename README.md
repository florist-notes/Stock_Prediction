# Stock_Prediction

 Project started : 21:50 , 28/10/2017 | PwC challenge #9, Hack2Innovate Hackathon (2 Day), IIT-G

 Sentiment Analysis of Tweets & predicting Stock prices based on user sentiments about # Amazon
 
 Dataset : We have created 2 data sets on our own.Since we lack proper cleaned datasets for both Stock data & tweets of same timeframe in the meantime.
 
 (1) data.json - is a dictionary holding a key-value pair of sentiment ( 1 for positive & 0 for negative )
     , where "key" is the sentiment ( either 0 or 1 ) & "value" is the sample tweet (" Actual Tweet by users").
     
 (2) data.txt - sample data file holding 2 columns & 100 rows (rows denote no of cases).
 
     The two columns:
     column ( tweet_sent ) - holds data if the tweet was positive ( 1 ) or negative ( 0 ).
     column ( stock_sent ) - holds data if stock price rose ( 1 ) or fell ( 0 ).
     0 for fall/-ve sentiment & 1 for rise/+ve sentiment.
     
 Note : Since, the stock fluctuations & the user tweets are made up by us, they are synthetic.We are unable to predict and match with real case scenarios.For instance, during festive season, people generally shops more & feel happy about it.This trend during festive season generally tends to increase stock prices, because of more demand of the products.Since, here we dont have a time to time mapped real dataset of people's sentiments & stock fluctuation, We are unable to predict/verify if it really happened.
 
Future case scenario : With proper datasets, we can better train the model & hence get better accuracy.We will be able to extend & predict the same outcome of another e-commerce site during similar historic events.


Dependencies:

    01. numpy
    02. pandas
    03. matplotlib
    04. json
    05. tflearn
    06. tensorflow
    07. sklearn
    08. nltk
    09. sys
    10. random
    11. string
    12. unicodedata

To run : 

    TO RUN : >>>tweeStock.py



INPUT TEST CASES:

![Input Image](https://github.com/SKKSaikia/Stock_Prediction/blob/master/input.PNG)

OUTPUT TEST CASES:

![Output Image](https://github.com/SKKSaikia/Stock_Prediction/blob/master/output.PNG)


We can see our model performs well for the self generated test cases.In future, we can maybe get real time tweets with the help of an API & predict the stock fluctuations for the coming days & give the functionality through a webapp.



BEHIND THE SCENES:

Correlation between the two columns ( tweet_sent,stock_sent ) of data.txt found is : 85.7785299553 %
It's plot is :
![Correlation plot](https://github.com/SKKSaikia/Stock_Prediction/blob/master/corr.png)

if (correlation > 70 )

   Model : Tensorflow Deep Neural Network (DNN).
   
   Training : Gradient Descent.
   
   Test data.
   
   result.

