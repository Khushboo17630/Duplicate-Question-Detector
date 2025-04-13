from flask import Flask, request, render_template
import pickle
import numpy as np
from nltk.corpus import stopwords
import nltk
from fuzzywuzzy import fuzz
import distance
import re

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
STOP_WORDS = stopwords.words("english")

# Preprocessing
def preprocess(q):
    q = str(q).lower().strip()
    q = re.sub(r'\W', ' ', q)
    q = re.sub(r'\s+', ' ', q)
    return q

# Common words
def common_words(q1, q2):
    w1 = set(q1.split())
    w2 = set(q2.split())
    return len(w1 & w2)

def total_words(q1, q2):
    w1 = set(q1.split())
    w2 = set(q2.split())
    return len(w1) + len(w2)

# Token Features
def fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    token_features = [0.0] * 8
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    return token_features

# Length Features
def fetch_length_features(q1, q2):
    length_features = [0.0]*3
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    return length_features

# Fuzzy Features
def fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features

# Combine all features
def query_point_creator(q1, q2):
    input_query = []
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))
    input_query.append(common_words(q1, q2))
    input_query.append(total_words(q1, q2))
    input_query.append(round(common_words(q1, q2)/total_words(q1, q2), 2))

    # Token, length, fuzzy features
    input_query += fetch_token_features(q1, q2)
    input_query += fetch_length_features(q1, q2)
    input_query += fetch_fuzzy_features(q1, q2)

    # BOW features
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, -1), q1_bow, q2_bow))

# Routes
@app.route('/')
def home():
    return render_template('index.html', q1="", q2="", prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    action = request.form['action']
    q1 = request.form.get('question1', '')
    q2 = request.form.get('question2', '')

    if action == 'Clear':
        return render_template('index.html', prediction_text="", q1="", q2="")

    features = query_point_creator(q1, q2)
    prediction = model.predict(features)[0]
    result = "Duplicate" if prediction == 1 else "Not Duplicate"
    return render_template('index.html', prediction_text=result, q1=q1, q2=q2)

if __name__ == "__main__":
    app.run(debug=True)
