import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow import keras


# Important Variables
vocab_size = 800
embedding_dim = 14
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
# Total size 29
training_size = 29
num_epochs = 110

# Defining the class names
class_names = ["What", "Variants", "Symptoms", "Feeling", "Precautions"]

# Loading in the json data
with open("chatbot-dataset.json", 'r') as f:
    datastore = json.load(f)

# Making the seperate lists for the sentence and if it has profanity or not
sentences = []
labels = []

# Looping over and grabbing all elements listed above
for item in datastore:
    sentences.append(item['sentence'])
    labels.append(item['tag'])

# Creating the training and the testing data
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
print(testing_labels)
print(training_labels)

print(training_sentences)

# Creating the keras tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Converting the texts to sequencies using the tokenizer
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Making the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dense(700, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(124, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='sigmoid')
])

# Loading the model
model = keras.models.load_model("covid-assistance-ai/")

# Compiling and fitting the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fitting the model
# history = model.fit(training_padded, training_labels, epochs=num_epochs,
#                     validation_data=(testing_padded, testing_labels), verbose=2)

# Getting the model summary
model.summary()

# Saving the model
model.save("covid-assistance-ai/")


# Reversing the word index
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_sentence(text):
    # Fancy Looking List Comprehension
    return ' '.join([reverse_word_index.get(i, '?') for i in text])



sentence = ["What to do if I feel like vomiting"]
results = []

print(sentence)

final_response = 0

# Getting the predictions to display
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length,
                       padding=padding_type, truncating=trunc_type)
# print(model.predict(padded))
prediction = model.predict(padded)

print(prediction)

# Getting the prediction and returning it mapped to the clsss names list defined above
response = class_names[np.argmax(prediction)]
print(np.argmax(prediction))
print(response)


# Getting the response
if response == "What":
    print("Coronavirus is a deadly virus that was originated in China in 2019.")

elif response == "Variants":
    print("Alpha, Beta, Gamma and Delta are the main variants")

elif response == "Symptoms":
    print("Most common symptoms: fever dry cough tiredness Less common symptoms: aches and pains sore throat diarrhoea conjunctivitis headache loss of taste or smell a rash on skin, or discolouration of fingers or toes Serious symptoms: difficulty breathing or shortness of breath chest pain or pressure loss of speech or movement")

elif response == "Feeling":
    print("You should visit a doctor and check your oxygen levels and if they fall below 90 then you should seek immediate medical attention. Also check to see if the symptom that you are showing actually is a COVID symptom cause sometimes you will just have some common disease")

elif response == "Precautions":
    print("You should always wear a mask and a face shield is optional")