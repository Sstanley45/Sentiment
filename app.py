#from operator import index
from flask import Flask, request, render_template, jsonify
from model import SentimentRecommenderModel

app = Flask(__name__)

sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    # get the review text from the html form
    review_text = request.form["reviewText"]
    print(review_text)
    pred_sentiment = sentiment_model.classify_sentiment(review_text)
    print(pred_sentiment)
    return render_template("index.html", sentiment=pred_sentiment)


if __name__ == '__main__':
    app.run()
