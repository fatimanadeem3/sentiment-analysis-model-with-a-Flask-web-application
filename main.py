import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib


train_df = pd.read_csv("twitter_training.csv", header=None, encoding='utf-8')
train_df.columns = ['ID', 'Topic', 'Sentiment', 'Text']

texts = train_df['Text'].astype(str).values
labels = train_df['Sentiment'].values

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(categorical_labels.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(padded_sequences, categorical_labels, epochs=5, batch_size=64, validation_split=0.1)

# Save model and tokenizer
model.save("sentiment_lstm_model.h5")
joblib.dump(tokenizer, "tokenizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
