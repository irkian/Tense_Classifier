import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer


__model = None

def load_saved_artifcats():
    print("Loading saved artifacts....... Start")

    with open("./artifacts/TenseClassifier.pickle", "rb") as f:
        __model = pickle.load(f)
    print("Loading saved artifacts........ Done")

def predictTense(sentence):
    tense_mapping = \
        {11: 'Simple Present',
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
    sentence = sentence.lower()  # Convert to lowercase
    sentence = re.sub(r'\W', ' ', sentence)  # Remove non-alphanumeric characters
    sentence = re.sub(r'\s+', ' ', sentence)  # Remove extra whitespaces
    # Apply any additional preprocessing steps if needed

    custom_sentence_vector = TfidfVectorizer().fit_transform([sentence]).toarray()

    predicted_tense = __model.predict(custom_sentence_vector)[0]

    # Print the predicted tense
    return "Predicted Tense:", tense_mapping[predicted_tense]

if __name__ == "__main__":
    load_saved_artifcats()
    print(predictTense("He drinks tea at breakfast"))
