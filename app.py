from flask import Flask, request, render_template
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Define preprocessing function with lemmatization
lemmatizer = WordNetLemmatizer()

def preprocess(content):
    content = re.sub('[^a-zA-Z]', ' ', content)  # Remove special characters
    content = content.lower()  # Convert to lowercase
    content = content.split()  # Tokenize
    content = [lemmatizer.lemmatize(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Apply preprocessing to the content column
news_df['content'] = news_df['content'].apply(preprocess)

# Vectorize data using TF-IDF with n-grams
X = news_df['content'].values
y = news_df['label'].values

vector = TfidfVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Calculate initial accuracy
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

# Prediction function with probabilities
def prediction(input_text):
    input_data = vector.transform([input_text])
    probabilities = model.predict_proba(input_data)  # Get probabilities
    prediction = model.predict(input_data)  # Get class prediction
    return prediction[0], probabilities[0]  # Return predicted class and probabilities

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    # Handle POST request
    input_text = request.form['news_article']
    if not input_text.strip():  # Error handling for empty input
        return render_template('index.html', prediction_text="Error: Please enter a valid news article.")

    pred, probabilities = prediction(input_text)
    if pred == 1:
        result = 'Fake News'
        info = "This story is highly questionable and may be spreading misinformation."
        confidence = f"Confidence: {probabilities[1] * 100:.2f}%"
    else:
        result = 'Real News'
        info = "This article seems authentic and is consistent with legitimate reports."
        confidence = f"Confidence: {probabilities[0] * 100:.2f}%"

    return render_template('index.html', 
                           prediction_text=result, 
                           info_text=info,
                           confidence_text=confidence
                           )


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
