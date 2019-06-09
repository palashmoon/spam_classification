#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('spam.tsv' , delimiter= '\t' , quoting= 3)
dataset = dataset.drop(["Unnamed: 2" , "Unnamed: 3" , "Unnamed: 4"] , axis= 1)
dataset.rename(columns = {'v1' : 'class' , 'v2' : 'message'}, inplace = True)
dataset.head()


#cleaning of dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
corpus = []
for i in range(0, 5573):
    review = re.sub('[^a-zA-z]' , ' ' , dataset['message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words(['english']))]
    review = ' '.join(review)
    corpus.append(review)  
    
    
#create a bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[: , 0].values

#test and train
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size = 0.30 , random_state = 0)

#use nayive basyes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)


#find the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test , y_pred)*100


#use another nayie bayes model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train , y_train)
y_pred1 = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test , y_pred1)*100