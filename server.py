from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer





import re

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    tenseMapping = {
        11: 'Simple Present',
        6: 'Present Continuous',
        7: 'Present Perfect',
        8: 'Present Perfect Continuous',
        10: 'Simple Past',
        3: 'Past Continuous ',
        4: 'Past Continuous ',
        5: 'Past Perfect',
        9: 'Past Perfect Continuous',
        0: 'Past Perfect Continuous',
        1: 'Simple Future',
        2: 'Simple Future'}

    sentence = request.form['Your_Sentence']
    pprint(sentence)
    sentence = sentence.lower()  # Convert to lowercase
    sentence = re.sub(r'\W', ' ', sentence)  # Remove non-alphanumeric characters
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove extra whitespaces



    custom_sentence_vector = vectorizer.transform([sentence]).toarray()




    predicted_tense = model.predict(custom_sentence_vector)[0]
    predicted_tense = tenseMapping[predicted_tense]

    # Print the predicted tense
    return render_template('index.html', pred=predicted_tense)


if __name__ == '__main__':
    app.debug = True
    app.run()
