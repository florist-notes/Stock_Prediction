 '''
 Project started : 21:50 , 28/10/2017 | PwC challenge #9, Hack2Innovate Hackathon (2 Day), IIT-G
 Sentiment Analysis of Tweets & predicting Stock prices based on user sentiments about # Amazon

 Dataset : We have created both the data sets on our own.

 (1) data.json - Is a dictionary holding a key-value pair of sentiment ( 1 for positive & 0 for negative )
     "key" is the sentiment ( either 0 or 1 ) & "value" is the sample tweet.

 (2) data.txt - sample data file holding 2 columns & 100 rows (rows denote no of cases)
     column ( tweet_sent ) - holds data if the tweet was positive ( 1 ) or negative ( 0 ).
     column ( stock_sent ) - holds data if stock price rose ( 1 ) or fell ( 0 ).
     0 for fall/-ve sentiment & 1 for rise/+ve sentiment


 We lack proper datasets for both Stock data & tweets of same timeframe.

-----------------------------------------------------------------------------------------
Managed 2 datasets - AMZN.csv - stock data & AMZNnews.json - Amazon News
Need to clean Data & form a dataset in the form of data.txt
-----------------------------------------------------------------------------------------

 '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

data = None

with open('data.json') as json_data:
    data = json.load(json_data)
    print (data)

stock = pd.read_csv("data.txt")
print(stock)

''' ---------------------------------------------------------------------------
    Correlation Analyzer
    ---------------------------------------------------------------------------
'''

corr_stock = pd.DataFrame.corr(stock)
print('Correlation Matrix : ',corr_stock)

 # //////////////////

plt.figure(figsize=(10,10))
plt.pcolor(corr_stock)
plt.title(" Correlation of Stock Fluctuations of Data from Tweet Sentiment Expectation Vs Actual Data ")
plt.colorbar()
plt.show()

corr_p = 100 * corr_stock.iloc[0,1]
print(" Correlation Percentage : ",corr_p)

''' ---------------------------------------------------------------------------
    Cleaning the data
    ---------------------------------------------------------------------------
'''
from sklearn.ensemble import RandomForestClassifier
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import sys
import random
import string
import unicodedata
import tflearn
import tensorflow as tf

if corr_p >= 70:
        # A table structure to hold the different punctuation used
        tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

        def remove_punctuation(text):
            return text.translate(tbl)

        stemmer = LancasterStemmer()


        categories = list(data.keys())
        # “words” will hold all the unique stemmed words in all the sentences provided for training
        words = []
        # docs will hold a list of tuples with words in the sentence and category name
        docs = []

        for each_category in data.keys():
            for each_sentence in data[each_category]:
                each_sentence = remove_punctuation(each_sentence)
                w = nltk.word_tokenize(each_sentence)
                words.extend(w)
                docs.append((w, each_category))

        # stem and lower each word and remove duplicates
        words = [stemmer.stem(w.lower()) for w in words]
        words = sorted(list(set(words)))


        print (words)
        print (docs)

        training = []
        output = []
        
        # create an empty array for our output
        output_empty = [0] * len(categories)

        # create a bag of words & tell which category the current bag of words belong to
        for doc in docs:
            # initialize our bag of words(bow) for each document in the list
            bow = []
            # list of tokenized words for the pattern
            token_words = doc[0]
            # stem each word
            token_words = [stemmer.stem(word.lower()) for word in token_words]
            # create our bag of words array /// insert 1 for true 0 for false, since data into Tensorflow must be numeric tensor
            for w in words:
                bow.append(1) if w in token_words else bow.append(0)
            output_row = list(output_empty)
            output_row[categories.index(doc[1])] = 1
            training.append([bow, output_row])

        random.shuffle(training)
        training = np.array(training)

        # trainX contains the Bag of words and train_y contains the label/ category
        train_x = list(training[:,0])
        train_y = list(training[:,1])

        tf.reset_default_graph()

        ''' ---------------------------------------------------------------------------
            Building the Deep Neural Network
            ---------------------------------------------------------------------------
        '''

        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

        ''' ---------------------------------------------------------------------------
            Training the DNN
            ---------------------------------------------------------------------------
        '''

        model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save('model.tflearn')

        ''' ---------------------------------------------------------------------------
            Testing the DNN
            ---------------------------------------------------------------------------
        '''
          # let's test the mdodel for these few sentences:
        sent_1 = "Amazon is Bad"                   # 0
        sent_2 = "dussehra dhamaka offer"          # 1
        sent_3 = "I love amazon"                   # 1
        sent_4 = "Great Diwali offers by Amazon."  # 1
        sent_5 = "Bad Quality Products"            # 0
        sent_6 = "poor service"                    # 0
        sent_7 = "Great, I am loving it"           # 1
        sent_8 = "nope! not buying again"          # 0
        sent_9 = "Life is easy with Amazon"        # 1
        sent_10 = "defective products"             # 0

          # a method that takes in a sentence and list of all words
          # and returns the data in a form the can be fed to tensorflow
        def get_tf_record(sentence):
            global words
            # tokenize the pattern
            sentence_words = nltk.word_tokenize(sentence)
            # stem each word
            sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
            # bag of words
            bow = [0]*len(words)
            for s in sentence_words:
                for i,w in enumerate(words):
                    if w == s:
                        bow[i] = 1

            return(np.array(bow))
        dec = [" Stock Price may Decrease"," Stock price may Increase"]
        # we can start to predict the results for each of the 10 sentences
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_1)]))])])  # Correct Output should be : 0
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_2)]))])])  # Correct Output should be : 1
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_3)]))])])  # Correct Output should be : 1
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_4)]))])])  # Correct Output should be : 1
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_5)]))])])  # Correct Output should be : 0
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_6)]))])])  # Correct Output should be : 0
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_7)]))])])  # Correct Output should be : 1
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_8)]))])])  # Correct Output should be : 0
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_9)]))])])  # Correct Output should be : 1
        print(dec[int(categories[np.argmax(model.predict([get_tf_record(sent_10)]))])]) # Correct Output should be : 0

        x=0
        if(categories[np.argmax(model.predict([get_tf_record(sent_1)]))] == '0'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_2)]))] == '1'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_3)]))] == '1'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_4)]))] == '1'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_5)]))] == '0'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_6)]))] == '0'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_7)]))] == '1'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_8)]))] == '0'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_9)]))] == '1'):
            x+=1
        if(categories[np.argmax(model.predict([get_tf_record(sent_10)]))] == '0'):
            x+=1
        correctness = (x/10)*100
        print(" Correct Result Percentage is : ",correctness)





''' ---------------------------------------------------------------------------

    ---------------------------------------------------------------------------
'''
