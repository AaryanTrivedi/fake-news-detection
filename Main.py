import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv (r"C:\Users\skryl\Desktop\Project\train.csv")
data.head()

data.shape
data.isnull().sum() 
df1 = data.fillna('')
df1['content'] = df1['author'] + ' ' + df1['title']

stemmer = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower() 
    stemmed_content = stemmed_content.split() 
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')] 
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df1['content'] = df1['content'].apply(stemming)
df1['content'].head()

X = df1.content.values
y = df1.label.values
X = TfidfVectorizer().fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)
model = LogisticRegression()
model.fit(X_train, y_train)

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, y_train)
print(training_accuracy)

X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, y_test)
print(testing_accuracy)

X_sample = X_test[1] #Need to create a system that helps us searc
prediction = model.predict(X_sample)
if prediction == 0:
    print('The NEWS is Real!')
else:
    print('The NEWS is Fake!')
