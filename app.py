import streamlit as st
from PIL import Image
import pandas as pd
from pickle import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


# Set the page configuration with the title and background image
st.set_page_config(
    page_title="Condition and Drug Name Prediction",
    page_icon=":pill:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and vectorizer
model = load(open('model.pkl', 'rb'))
vectorizer = load(open('TfidfVectorizer.pkl', 'rb'))

image = Image.open("medication-review.jpg")
st.image(image)

html_temp="""
<div style ="background-color:Black;padding:10px">
<h2 style="color:white;text-align:center;"> Condition and Drug Name Prediction </h2>
"""
st.markdown(html_temp,unsafe_allow_html=True)

# Load the data
df2 = pd.read_table('drugsCom_raw (1).tsv')
condition1 = ['Depression','High Blood Pressure','Diabetes, Type 2']
df1 = df2[df2['condition'].isin(condition1)]
X = df1.drop(['Unnamed: 0','rating','date','usefulCount','drugName'],axis=1)

# Clean the reviews
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmatized_words)

X['review_clean'] = X['review'].apply(review_to_words)

# Split the data into train and test sets
X_feat = X['review_clean']
y = X['condition']
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, stratify=y, test_size=0.2, random_state=0)

# Vectorize the text data
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Train the model
pass_tf1 = RandomForestClassifier()
pass_tf1.fit(tfidf_train, y_train)

# Create text input for user to enter review
text = st.text_input('Enter the Text: ')

# Create predict button to predict condition and recommended drugs
if st.button('Predict'):
    test = vectorizer.transform([text])
    pred1 = pass_tf1.predict(test)[0]
    st.subheader("Condition:")
    st.write(pred1)

    drug_ratings = df2[df2["condition"] == pred1].groupby("drugName")["rating"].mean()
    recommended_drugs = drug_ratings.nlargest(3).index.tolist()
    st.subheader("Recommended Drugs:")
    for i, drug in enumerate(recommended_drugs):
        st.write(i+1, drug)
