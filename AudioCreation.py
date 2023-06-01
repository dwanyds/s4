# Any results you write to the current directory are saved as output.

# %% [markdown]
# **Resources - **
# 1. https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/
# 2. https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/
# 3. https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
# 4. https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:43:43.266302Z","iopub.execute_input":"2023-05-11T08:43:43.266578Z","iopub.status.idle":"2023-05-11T08:43:44.398800Z","shell.execute_reply.started":"2023-05-11T08:43:43.266531Z","shell.execute_reply":"2023-05-11T08:43:44.397771Z"}}
summary = pd.read_csv('news_summary.csv', encoding='iso-8859-1')
raw = pd.read_csv('news_summary_more.csv', encoding='iso-8859-1')

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:43:44.400275Z","iopub.execute_input":"2023-05-11T08:43:44.400567Z","iopub.status.idle":"2023-05-11T08:43:44.472253Z","shell.execute_reply.started":"2023-05-11T08:43:44.400513Z","shell.execute_reply":"2023-05-11T08:43:44.471477Z"}}
pre1 = raw.iloc[:, 0:2].copy()
# pre1['head + text'] = pre1['headlines'].str.cat(pre1['text'], sep =" ")

pre2 = summary.iloc[:, 0:6].copy()
pre2['text'] = pre2['author'].str.cat(
    pre2['date'].str.cat(pre2['read_more'].str.cat(pre2['text'].str.cat(pre2['ctext'], sep=" "), sep=" "), sep=" "),
    sep=" ")

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:43:44.473387Z","iopub.execute_input":"2023-05-11T08:43:44.473798Z","iopub.status.idle":"2023-05-11T08:43:44.520878Z","shell.execute_reply.started":"2023-05-11T08:43:44.473755Z","shell.execute_reply":"2023-05-11T08:43:44.520067Z"}}
pre = pd.DataFrame()
pre['text'] = pd.concat([pre1['text'], pre2['text']], ignore_index=True)
pre['summary'] = pd.concat([pre1['headlines'], pre2['headlines']], ignore_index=True)

# %% [markdown]
# **Seq2Seq LSTM Modelling**
# ![final.jpg](attachment:final.jpg)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:43:44.523680Z","iopub.execute_input":"2023-05-11T08:43:44.524100Z","iopub.status.idle":"2023-05-11T08:43:44.533480Z","shell.execute_reply.started":"2023-05-11T08:43:44.524057Z","shell.execute_reply":"2023-05-11T08:43:44.532829Z"}}
# LSTM with Attention
# pip install keras-self-attention

pre['text'][:10]

# %% [markdown]
# > **Perform Data Cleansing**

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:43:44.536483Z","iopub.execute_input":"2023-05-11T08:43:44.536798Z","iopub.status.idle":"2023-05-11T08:43:44.559280Z","shell.execute_reply.started":"2023-05-11T08:43:44.536741Z","shell.execute_reply":"2023-05-11T08:43:44.558225Z"}}
import re


# Removes non-alphabetic characters:
def text_strip(column):
    for row in column:

        # ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!

        row = re.sub("(\\t)", ' ', str(row)).lower()  # remove escape charecters
        row = re.sub("(\\r)", ' ', str(row)).lower()
        row = re.sub("(\\n)", ' ', str(row)).lower()

        row = re.sub("(__+)", ' ', str(row)).lower()  # remove _ if it occors more than one time consecutively
        row = re.sub("(--+)", ' ', str(row)).lower()  # remove - if it occors more than one time consecutively
        row = re.sub("(~~+)", ' ', str(row)).lower()  # remove ~ if it occors more than one time consecutively
        row = re.sub("(\+\++)", ' ', str(row)).lower()  # remove + if it occors more than one time consecutively
        row = re.sub("(\.\.+)", ' ', str(row)).lower()  # remove . if it occors more than one time consecutively

        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower()  # remove <>()|&©ø"',;?~*!

        row = re.sub("(mailto:)", ' ', str(row)).lower()  # remove mailto:
        row = re.sub(r"(\\x9\d)", ' ', str(row)).lower()  # remove \x9* in text
        row = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower()  # replace INC nums to INC_NUM
        row = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower()  # replace CM# and CHG# to CM_NUM

        row = re.sub("(\.\s+)", ' ', str(row)).lower()  # remove full stop at end of words(not between)
        row = re.sub("(\-\s+)", ' ', str(row)).lower()  # remove - at end of words(not between)
        row = re.sub("(\:\s+)", ' ', str(row)).lower()  # remove : at end of words(not between)

        row = re.sub("(\s+.\s+)", ' ', str(row)).lower()  # remove any single charecters hanging between 2 spaces

        # Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            repl_url = url.group(3)
            row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(row))
        except:
            pass  # there might be emails with no url in them

        row = re.sub("(\s+)", ' ', str(row)).lower()  # remove multiple spaces

        # Should always be last
        row = re.sub("(\s+.\s+)", ' ', str(row)).lower()  # remove any single charecters hanging between 2 spaces

        yield row


# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:43:44.561104Z","iopub.execute_input":"2023-05-11T08:43:44.561431Z","iopub.status.idle":"2023-05-11T08:43:44.573730Z","shell.execute_reply.started":"2023-05-11T08:43:44.561374Z","shell.execute_reply":"2023-05-11T08:43:44.572669Z"}}
brief_cleaning1 = text_strip(pre['text'])
brief_cleaning2 = text_strip(pre['summary'])

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:43:44.576611Z","iopub.execute_input":"2023-05-11T08:43:44.576962Z","iopub.status.idle":"2023-05-11T08:53:26.986152Z","shell.execute_reply.started":"2023-05-11T08:43:44.576902Z","shell.execute_reply":"2023-05-11T08:53:26.985041Z"}}
from time import time
import spacy

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])  # disabling Named Entity Recognition for speed

# Taking advantage of spaCy .pipe() method to speed-up the cleaning process:
# If data loss seems to be happening(i.e len(text) = 50 instead of 75 etc etc) in this cell , decrease the batch_size parametre

t = time()
print("Test by dingu 1 ")
# Batch the data points into 5000 and run on all cores for faster preprocessing
text = [str(doc) for doc in nlp.pipe(brief_cleaning1, batch_size=100)]

# Takes 7-8 mins
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:53:26.988238Z","iopub.execute_input":"2023-05-11T08:53:26.988623Z","iopub.status.idle":"2023-05-11T08:56:11.989521Z","shell.execute_reply.started":"2023-05-11T08:53:26.988535Z","shell.execute_reply":"2023-05-11T08:56:11.988488Z"}}
# Taking advantage of spaCy .pipe() method to speed-up the cleaning process:


t = time()
print("Test by dingu 2 ")
# Batch the data points into 5000 and run on all cores for faster preprocessing
summary = ['_START_ ' + str(doc) + ' _END_' for doc in nlp.pipe(brief_cleaning2, batch_size=100)]

# Takes 7-8 mins
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:11.991898Z","iopub.execute_input":"2023-05-11T08:56:11.992582Z","iopub.status.idle":"2023-05-11T08:56:12.048311Z","shell.execute_reply.started":"2023-05-11T08:56:11.992520Z","shell.execute_reply":"2023-05-11T08:56:12.047556Z"}}
pre['cleaned_text'] = pd.Series(text)
pre['cleaned_summary'] = pd.Series(summary)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:12.050126Z","iopub.execute_input":"2023-05-11T08:56:12.050854Z","iopub.status.idle":"2023-05-11T08:56:12.056488Z","shell.execute_reply.started":"2023-05-11T08:56:12.050789Z","shell.execute_reply":"2023-05-11T08:56:12.054960Z"}}
text_count = []
summary_count = []

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:12.058494Z","iopub.execute_input":"2023-05-11T08:56:12.059107Z","iopub.status.idle":"2023-05-11T08:56:12.859561Z","shell.execute_reply.started":"2023-05-11T08:56:12.058832Z","shell.execute_reply":"2023-05-11T08:56:12.858509Z"}}
for sent in pre['cleaned_text']:
    text_count.append(len(sent.split()))
for sent in pre['cleaned_summary']:
    summary_count.append(len(sent.split()))

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:12.861652Z","iopub.execute_input":"2023-05-11T08:56:12.862223Z","iopub.status.idle":"2023-05-11T08:56:12.867717Z","shell.execute_reply.started":"2023-05-11T08:56:12.862001Z","shell.execute_reply":"2023-05-11T08:56:12.866831Z"}}
# Model to summarize the text between 0-15 words for Summary and 0-100 words for Text
max_text_len = 100
max_summary_len = 15

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:12.869005Z","iopub.execute_input":"2023-05-11T08:56:12.869260Z","iopub.status.idle":"2023-05-11T08:56:13.775895Z","shell.execute_reply.started":"2023-05-11T08:56:12.869215Z","shell.execute_reply":"2023-05-11T08:56:13.775118Z"}}
# Select the Summaries and Text between max len defined above

cleaned_text = np.array(pre['cleaned_text'])
cleaned_summary = np.array(pre['cleaned_summary'])

short_text = []
short_summary = []

for i in range(len(cleaned_text)):
    if (len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_text[i].split()) <= max_text_len):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

post_pre = pd.DataFrame({'text': short_text, 'summary': short_summary})

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:13.777352Z","iopub.execute_input":"2023-05-11T08:56:13.777865Z","iopub.status.idle":"2023-05-11T08:56:13.849671Z","shell.execute_reply.started":"2023-05-11T08:56:13.777815Z","shell.execute_reply":"2023-05-11T08:56:13.848875Z"}}
# Add sostok and eostok at
post_pre['summary'] = post_pre['summary'].apply(lambda x: 'sostok ' + x + ' eostok')

# %% [markdown]
# **SEQ2SEQ MODEL BUILDING **

# %% [markdown]
# Split the data to TRAIN and VALIDATION sets

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:13.851152Z","iopub.execute_input":"2023-05-11T08:56:13.851683Z","iopub.status.idle":"2023-05-11T08:56:14.883024Z","shell.execute_reply.started":"2023-05-11T08:56:13.851633Z","shell.execute_reply":"2023-05-11T08:56:14.881932Z"}}
from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(np.array(post_pre['text']), np.array(post_pre['summary']), test_size=0.1,
                                            random_state=0, shuffle=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:14.884917Z","iopub.execute_input":"2023-05-11T08:56:14.885242Z","iopub.status.idle":"2023-05-11T08:56:24.226413Z","shell.execute_reply.started":"2023-05-11T08:56:14.885179Z","shell.execute_reply":"2023-05-11T08:56:24.225211Z"}}
# Lets tokenize the text to get the vocab count , you can use Spacy here also

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

# %% [markdown]
# **RARE WORD ANALYSIS FOR X i.e 'text'**
# * tot_cnt gives the size of vocabulary (which means every unique words in the text)
#
# * cnt gives me the no. of rare words whose count falls below threshold
#
# * tot_cnt - cnt gives me the top most common words

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:24.228031Z","iopub.execute_input":"2023-05-11T08:56:24.228355Z","iopub.status.idle":"2023-05-11T08:56:24.312679Z","shell.execute_reply.started":"2023-05-11T08:56:24.228295Z","shell.execute_reply":"2023-05-11T08:56:24.311543Z"}}
thresh = 4

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
print("Total Coverage of rare words:", (freq / tot_freq) * 100)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:24.314528Z","iopub.execute_input":"2023-05-11T08:56:24.314858Z","iopub.status.idle":"2023-05-11T08:56:41.519044Z","shell.execute_reply.started":"2023-05-11T08:56:24.314800Z","shell.execute_reply":"2023-05-11T08:56:41.518080Z"}}

# prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
x_tokenizer.fit_on_texts(list(x_tr))

# convert text sequences into integer sequences (i.e one-hot encodeing all the words)
x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

# padding zero upto maximum length
x_tr = pad_sequences(x_tr_seq, maxlen=max_text_len, padding='post')
x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

# size of vocabulary ( +1 for padding token)
x_voc = x_tokenizer.num_words + 1

print("Size of vocabulary in X = {}".format(x_voc))

# %% [markdown]
# **RARE WORD ANALYSIS FOR Y i.e 'summary'**
# * tot_cnt gives the size of vocabulary (which means every unique words in the text)
#
# * cnt gives me the no. of rare words whose count falls below threshold
#
# * tot_cnt - cnt gives me the top most common words

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:41.520539Z","iopub.execute_input":"2023-05-11T08:56:41.520873Z","iopub.status.idle":"2023-05-11T08:56:44.440653Z","shell.execute_reply.started":"2023-05-11T08:56:41.520803Z","shell.execute_reply":"2023-05-11T08:56:44.439903Z"}}
# prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:44.442433Z","iopub.execute_input":"2023-05-11T08:56:44.442836Z","iopub.status.idle":"2023-05-11T08:56:44.480821Z","shell.execute_reply.started":"2023-05-11T08:56:44.442764Z","shell.execute_reply":"2023-05-11T08:56:44.479786Z"}}
thresh = 6

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
print("Total Coverage of rare words:", (freq / tot_freq) * 100)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:44.482657Z","iopub.execute_input":"2023-05-11T08:56:44.483083Z","iopub.status.idle":"2023-05-11T08:56:51.419148Z","shell.execute_reply.started":"2023-05-11T08:56:44.483012Z","shell.execute_reply":"2023-05-11T08:56:51.417981Z"}}
# prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
y_tokenizer.fit_on_texts(list(y_tr))

# convert text sequences into integer sequences (i.e one hot encode the text in Y)
y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

# padding zero upto maximum length
y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

# size of vocabulary
y_voc = y_tokenizer.num_words + 1
print("Size of vocabulary in Y = {}".format(y_voc))

# %% [markdown]
# We will now remove "Summary" i.e Y (both train and val) which has only _START_ and _END_

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:51.421985Z","iopub.execute_input":"2023-05-11T08:56:51.422372Z","iopub.status.idle":"2023-05-11T08:56:56.146243Z","shell.execute_reply.started":"2023-05-11T08:56:51.422312Z","shell.execute_reply":"2023-05-11T08:56:56.145205Z"}}
ind = []
for i in range(len(y_tr)):
    cnt = 0
    for j in y_tr[i]:
        if j != 0:
            cnt = cnt + 1
    if (cnt == 2):
        ind.append(i)

y_tr = np.delete(y_tr, ind, axis=0)
x_tr = np.delete(x_tr, ind, axis=0)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:56.147771Z","iopub.execute_input":"2023-05-11T08:56:56.148087Z","iopub.status.idle":"2023-05-11T08:56:56.679244Z","shell.execute_reply.started":"2023-05-11T08:56:56.148029Z","shell.execute_reply":"2023-05-11T08:56:56.678363Z"}}
ind = []
for i in range(len(y_val)):
    cnt = 0
    for j in y_val[i]:
        if j != 0:
            cnt = cnt + 1
    if (cnt == 2):
        ind.append(i)

y_val = np.delete(y_val, ind, axis=0)
x_val = np.delete(x_val, ind, axis=0)

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:56.680880Z","iopub.execute_input":"2023-05-11T08:56:56.681192Z","iopub.status.idle":"2023-05-11T08:56:59.696073Z","shell.execute_reply.started":"2023-05-11T08:56:56.681137Z","shell.execute_reply":"2023-05-11T08:56:59.694956Z"}}
from keras import backend as K
import gensim
from numpy import *
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

print("Size of vocabulary from the w2v model = {}".format(x_voc))

K.clear_session()

latent_dim = 300
embedding_dim = 200

# Encoder
encoder_inputs = Input(shape=(max_text_len,))

# embedding layer
enc_emb = Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)

# encoder lstm 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# encoder lstm 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# encoder lstm 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

# embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# dense layer
decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:59.697794Z","iopub.execute_input":"2023-05-11T08:56:59.698226Z","iopub.status.idle":"2023-05-11T08:56:59.774398Z","shell.execute_reply.started":"2023-05-11T08:56:59.698154Z","shell.execute_reply":"2023-05-11T08:56:59.773471Z"}}
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:59.775962Z","iopub.execute_input":"2023-05-11T08:56:59.776295Z","iopub.status.idle":"2023-05-11T08:56:59.781883Z","shell.execute_reply.started":"2023-05-11T08:56:59.776230Z","shell.execute_reply":"2023-05-11T08:56:59.780787Z"}}
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

# %% [markdown]
# **Start fitting the model with the data**

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T08:56:59.783752Z","iopub.execute_input":"2023-05-11T08:56:59.784141Z","iopub.status.idle":"2023-05-11T09:45:37.575570Z","shell.execute_reply.started":"2023-05-11T08:56:59.784065Z","shell.execute_reply":"2023-05-11T09:45:37.574677Z"}}
history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=1,
                    callbacks=[es], batch_size=128,
                    validation_data=([x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))

# %% [markdown]
# **Visualize the model learning**

# %% [markdown]
# **Next, let’s build the dictionary to convert the index to word for target and source vocabulary:**

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T09:45:37.578114Z","iopub.execute_input":"2023-05-11T09:45:37.578494Z","iopub.status.idle":"2023-05-11T09:45:37.585719Z","shell.execute_reply.started":"2023-05-11T09:45:37.578434Z","shell.execute_reply":"2023-05-11T09:45:37.583347Z"}}
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T09:45:37.587116Z","iopub.execute_input":"2023-05-11T09:45:37.588680Z","iopub.status.idle":"2023-05-11T09:45:37.997714Z","shell.execute_reply.started":"2023-05-11T09:45:37.588626Z","shell.execute_reply":"2023-05-11T09:45:37.996879Z"}}
# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2,
                                                    initial_state=[decoder_state_input_h, decoder_state_input_c])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


# %% [markdown]
# **We are defining a function below which is the implementation of the inference process**

# %% [code] {"execution":{"iopub.status.busy":"2023-05-11T09:45:37.999363Z","iopub.execute_input":"2023-05-11T09:45:37.999816Z","iopub.status.idle":"2023-05-11T09:45:38.011795Z","shell.execute_reply.started":"2023-05-11T09:45:37.999613Z","shell.execute_reply":"2023-05-11T09:45:38.010588Z"}}
def decode_sequence(input_seq, result):
    # Encode the input as state vectors.
    print(result)
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if (sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def tocharSeq(txttest):
    # prepare a tokenizer for reviews our data
    our_tok = Tokenizer(num_words=tot_cnt - globals()["cnt"])
    our_tok.fit_on_texts(list(txttest))
    # convert text sequences into integer sequences (i.e one hot encode the text in Y)
    our_seq = our_tok.texts_to_sequences(txttest)

    kl = pad_sequences(our_seq, maxlen=max_text_len, padding='post')
    ind = []
    for i in range(len(kl)):
        cnt = 0
        for j in kl[i]:
            if j != 0:
                cnt = cnt + 1
        if (cnt == 2):
            ind.append(i)

    kl = np.delete(kl, ind, axis=0)
    print(kl)
    return kl

