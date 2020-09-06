import re
import codecs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('russian'))

def load(direct):
    file = open(direct, 'rb')
    lines = []
    for line in file:
        line = line.decode("utf-8").strip().lower()
        if len(line) == 0:
            continue
        lines.append(line)
    file.close()
    return ' '.join(lines)

def clean(data):
    text = re.sub("[^А-Яа-я]", " ", data).lower()
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return text

def make_seq(data, seqlen=10, step=1):
    length = seqlen+step
    sequences = []
    for i in range(length, count_words):
        seq = data[i - length:i]
        line = " ".join(seq)
        sequences.append(line)
    return sequences

def save(lines, filename):
    data = '\n'.join(lines)
    with codecs.open(filename, 'w', 'utf-8') as file:
        file.write(data)

name_file = 'The Hound of the Baskervilles.txt'
SEQLEN = 2
STEP = 1

text_line = load(name_file)
clean_text = clean(text_line)
print(clean_text[:200])

count_words = len(clean_text)
print('Count of sequence: {0}'.format(count_words))

sequences = make_seq(clean_text)

save(sequences, 'conan_seq.txt')
