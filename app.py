import streamlit as st
import pickle
import re
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_word=set(stopwords.words('english'))
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
vect=CountVectorizer()


vector_form=pickle.load(open('vector.pkl','rb'))
load_model=pickle.load(open('model.pkl','rb'))

def datapreprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text_token = word_tokenize(text)
    filtered_text = [w for w in text_token if w not in stop_word]
    return ' '.join(filtered_text)


def stemming(data):
    text=[stemmer.stem(word) for word in data]
    return data





def fake_news(news):
    datapreprocess(news)
    stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction=load_model.predict(vector_form1)
    return prediction


def main():
    st.title("Fake News Detection App")

    # Get user input
    input_text = st.text_area("Enter the news text:", "")

    if st.button("Check"):
        if input_text.strip() == "":
            st.warning("Please enter some news text.")
        else:
            # Make predictions and display the result
            prediction = fake_news(input_text)
            if prediction == 0:
                st.error("Real News")
            else:
                st.success("Fake News")


                

if __name__ == "__main__":
    main()
