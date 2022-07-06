import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
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
            if i not in stopwords.words('oromiffa') and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
    return " ".join(y)
cv=pickle.load(open('vectorizer.pkl' ,'rb'))
model = pickle.load(open('model.pkl', "rb"))
st.title("Hate_speech detection ")
input_txt = st.text_input(" Enter the Text :")
if st.button('predict'):
    # 1.preprocess
    transform_txt = transform_text(input_txt)
    # 2.vectorize
    vector_input = cv.transform([transform_txt])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.Display
    if result == 1:
        st.header("Normal")
    else:
        st.header("Hate")
