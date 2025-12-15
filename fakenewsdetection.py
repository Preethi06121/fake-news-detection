import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv("fake_or_real_news.csv")

data = data[["title", "label"]]

data["title"] = data["title"].fillna("").astype(str)
data["label"] = data["label"].fillna("").astype(str)

data = data[
    (data["title"].str.strip() != "") &
    (data["label"].str.strip() != "")
]

x = data["title"]
y = data["label"]

cv = CountVectorizer(stop_words="english")
x = cv.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(xtrain, ytrain)

st.title("Fake News Detection System")

user = st.text_area("Enter Any News Headline:")

if user:
    input_data = cv.transform([user])
    result = model.predict(input_data)[0]
    st.write(result)
