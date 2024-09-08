import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load('fakenew.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Sample new
new_news_title = "Breaking: New Vaccine Discovered"
new_news_content = """Scientists have announced the discovery of a new vaccine that can potentially eradicate many diseases.
It is still in the early phases of testing but shows great promise."""

# Processing the new
new_news_vector = tfidf_vectorizer.transform([new_news_content])
prediction = model.predict(new_news_vector)

print(prediction)
