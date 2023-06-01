import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from pydub import AudioSegment
import tempfile

import manuel
import sendMail


def main():
    itx = ""
    # Set the title of the app
    st.title("YouTube Summariser")

    # Create two text fields
    url = st.text_input("Enter Youtube URL")
    email_id = st.text_input("Enter Email ID")

    # Create a submit button
    submit_button = st.button("Submit")

    # Check if the submit button is clicked
    if submit_button:
        video_id = url.split("=")[1]  # video iD
        # Display the result in a text area
        YouTubeTranscriptApi.get_transcript(video_id)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result = ""
        for i in transcript:
            result += ' ' + i['text']
        # print(result)
        print(len(result))

        summarizer = pipeline('summarization')

        num_iters = int(len(result) / 1000)
        summarized_text = []
        for i in range(0, num_iters + 1):
            start = 0
            start = i * 1000
            end = (i + 1) * 1000
            print("input text \n" + result[start:end])
            out = summarizer(result[start:end])
            out = out[0]
            out = out['summary_text']
            print("Summarized text\n" + out)
            # textSumModel2=manuel.decode_sequence(result[start:end])
            summarized_text.append(out)
            # manuel summary
        #  for i in range(0, 100):
        #    itx += manuel.decode_sequence(result[start:end].reshape(1, manuel.max_text_len)) + "\n"
        #  print(len(str(summarized_text)))
        num_iters = int(len(result) / 100)
        textSumModel2 = []

        textSumModel2 = manuel.decode_sequene(manuel.tocharSeq(result), result)

        # printing in text field
        # result=summarized_text
        result = findBestSummary(textSumModel2, summarized_text)
        # printScore(result,summarized_text)
        sendMail.sendMailtrans(email_id, url, result)
        st.text_area("Transcript", result)
    #  st.text_area("Transcript", itx)


def printScore(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the scores
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1


def findBestSummary(summaryr1, summaryr2):
    stop_words = set(stopwords.words('english'))

    # Preprocess the summaries
    summary1 = listToString(summaryr1).lower()
    summary2 = listToString(summaryr2).lower()
    sum1 = summary1
    sum2 = summary2
    # Tokenize the summaries
    tokens1 = nltk.word_tokenize(summary1)
    tokens2 = nltk.word_tokenize(summary2)

    # Remove stop words
    tokens1 = [token for token in tokens1 if token.isalnum() and token not in stop_words]
    tokens2 = [token for token in tokens2 if token.isalnum() and token not in stop_words]

    # Convert tokens back to strings
    summary1 = ' '.join(tokens1)
    summary2 = ' '.join(tokens2)

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([summary1, summary2])

    # Calculate cosine similarity between the vectors
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(similarity_matrix)
    # Determine the best summary based on cosine similarity
    if similarity_matrix[0, 1] > similarity_matrix[1, 0]:
        return sum1
    else:
        return sum2


if __name__ == "__main__":
    main()
