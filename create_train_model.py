import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Embedding
from keras.layers import Activation
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

def load(direct):
    with open(direct, 'rb') as file:
        text = file.read().decode("utf-8")
    return text

filename = 'conan_seq.txt'
text = load(filename)
sequences = text.split('\n')

print(f'First sequences :\n {sequences[0]}')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)
print(sequences)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size : {0}'.format(vocab_size))

sequences = np.array(sequences)
print('Shape of sequences -> ', sequences.shape)

X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)
print(X.shape, y.shape)

SEQLEN = 10
HIDDEN_SIZE = 100
BATCH_SIZE = 128
NUM_EPOCHS = 500
VERBOSE = 1

model = Sequential()
model.add(Embedding(vocab_size, SEQLEN, input_length=X.shape[1]))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(LSTM(HIDDEN_SIZE))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=VERBOSE)

print(history.history.keys())

fig1 = plt.figure(1)
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# plt.show()

# summarize history for loss
fig2 = plt.figure(2)
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.show()

plot_model(model, to_file='model_rnn.png', show_shapes=True, show_layer_names=True)

# сериализация модели в JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

