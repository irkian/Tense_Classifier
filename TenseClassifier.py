import numpy as np
import pandas as pd
import nltk
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import pickle
import warnings

warnings.filterwarnings('ignore')


df = pd.read_excel('tense.xlsx')
df1 = df
df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates(keep='first')

def clean_data(sentences):
    sentences = re.sub('[^a-zA-Z]', ' ', sentences)
    sentences = sentences.lower()
    return sentences
df['Sentences'] = df['Sentences'].apply(clean_data)


lab = LabelEncoder()

#perform label encoding on 'team' column
df['Tense'] = lab.fit_transform(df['Tense'])

#view updated DataFrame
print(df)

tense_mapping = dict(zip(df['Tense'], df1['Tense']))

corpus = list(df['Sentences'])
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()
y = df['Tense'].values


tf_transformer = TfidfTransformer()
X = tf_transformer.fit_transform(X).toarray()

tfidfVectorizer = TfidfVectorizer(max_features=1000)
X = tfidfVectorizer.fit_transform(corpus).toarray()

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.20, random_state=120)

lr = LogisticRegression()
lr.fit(X_train_s, y_train_s)

pickle.dump(lr, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

pickle.dump(tfidfVectorizer, open("vectorizer.pkl", "wb"))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))







