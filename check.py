import numpy as np
import pickle
import re
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from keras.models import model_from_json

def clean(sentence):
    text = re.sub("[^А-Яа-я]", " ", sentence).lower()
    return text

def load(direct):
    with open(direct, 'rb') as file:
        text = file.read().decode("utf-8")
    return text

filename = 'conan_seq.txt'
text = load(filename)
sequences = text.split('\n')

json_file = open("model.json", "r")
load_model_json = json_file.read()
json_file.close()
model = model_from_json(load_model_json)
model.load_weights("model.h5")

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    seed_text = clean(seed_text)
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return seed_text + " " + " ".join(result)

# line = 'Чего нет, того нет, зато передо мной и стоит  начищенный '
line = 'Все наводит на такое толкование. И если мы примем  мою'

print(generate_seq(model, tokenizer, 10, line, 5))

