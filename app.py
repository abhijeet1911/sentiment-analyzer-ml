from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# 1. Chhota sa data set (Training data)
# Naya aur bada Dataset
train_texts = [
    "I love this", "This is great", "I am happy", "This is wonderful", "Excellent work", "Amazing",
    "This is bad", "I hate this", "I am sad", "This is terrible", "Horrible experience", "Worst",
    "It is okay", "Could be better", "Not bad but not good"
]
train_labels = [
    "Positive", "Positive", "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative", "Negative", "Negative",
    "Neutral", "Neutral", "Neutral"
]

# 2. Model train karna
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_texts)
model = MultinomialNB()
model.fit(X, train_labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    user_query = request.args.get('q')
    if not user_query:
        return jsonify({"answer": "Kuch toh likho bhai!"})
    
    text_vector = vectorizer.transform([user_query])
    prediction = model.predict(text_vector)[0]
    
    # Model ki confidence (probability) nikalna
    probs = model.predict_proba(text_vector)
    confidence = max(probs[0]) * 100
    
    return jsonify({"answer": f"Sentiment: {prediction} ({confidence:.1f}% sure)"})
    
   

if __name__ == '__main__':
    app.run(debug=True, port=5000)