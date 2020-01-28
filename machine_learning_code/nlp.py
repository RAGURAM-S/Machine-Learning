import pandas

""" quoting parameter is set to 3 to ignore the double  quotes"""

dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Natural_Language_Processing/Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

#cleaning the text in the dataset
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
corpus = []
for i in range(0, 1000):
    review = dataset['Review'][i]
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# creation of bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 1500)
x = vectorizer.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, prediction)
confusion_matrix = metrics.confusion_matrix(y_test, prediction)
classification_report = metrics.classification_report(y_test, prediction)
precision = metrics.precision_score(y_test, prediction)
f1_score = metrics.f1_score(y_test, prediction)
recall = metrics.recall_score(y_test, prediction)
