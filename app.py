# Importing the Libraries
import logging
import os
import sys
import urllib
import flask
from flask import Flask, request, render_template
from flask_cors import CORS
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from textblob import TextBlob

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
def perform_sentiment_analysis_cnn(text):
    # create the model
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    text = text.replace("+", " ")
    analysis = TextBlob(text)
    if not analysis.sentiment.polarity < 0:
        return 'POSITIVE'
    else:
        return 'NEGATIVE'

# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='templates')

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/')
def main():
    return render_template('main.html')


# Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    text = urllib.parse.unquote(url)
    outcome = perform_sentiment_analysis_cnn(text)
    return render_template('main.html', prediction_text='It is the ' + outcome + ' sentiment')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)

