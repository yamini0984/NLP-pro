import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('TfidfVectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Drug Recommender")

input_sms = st.text_area("Enter the message")


if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    st.subheader("You are going through the condition known:")
    st.write(result)
    # 4. Display
    if result == 'Depression':
        st.subheader("Recommended Drugs:")
        st.write('Sertraline')
        st.write('Zoloft')
        st.write('Viibryd')

    elif result == 'High Blood Pressure':
        st.subheader("Recommended Drugs:")
        st.write('Losartan')
        st.write('Aldactone')
        st.write('Spironolactone')
    else:
        st.subheader("Recommended Drugs:")
        st.write('Victoza')
        st.write('Canagliflozin')
        st.write('Invokana')