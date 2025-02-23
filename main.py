import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------
# 1. PRÉPARATION DES DONNÉES AMÉLIORÉE
# ---------------------------------------------------

# Téléchargement du dataset complet
filepath = tf.keras.utils.get_file(
    'shakespeare.txt', 
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Nettoyage et normalisation du texte
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = re.sub(r'[^\w\s]','', text)  # Suppression de la ponctuation
text = re.sub(r'\s+', ' ', text)     # Suppression des espaces multiples

# Paramètres
VOCAB_SIZE = 50    # Taille maximale du vocabulaire (top 50 caractères)
SEQ_LENGTH = 60     # Augmentation de la longueur des séquences
STEP_SIZE = 3       # Pas de décalage

# Création du vocabulaire limité
char_counts = {char: text.count(char) for char in set(text)}
sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])[:VOCAB_SIZE-1]
chars = [char for char, _ in sorted_chars] + ['<UNK>']  # Ajout d'un token inconnu

char2idx = {char: i for i, char in enumerate(chars)}
idx2char = {i: char for i, char in enumerate(chars)}

# Fonction de conversion avec gestion des caractères inconnus
def char_to_idx(char):
    return char2idx[char] if char in char2idx else char2idx['<UNK>']

# ---------------------------------------------------
# 2. PRÉPARATION DES SÉQUENCES OPTIMISÉE
# ---------------------------------------------------

# Création des séquences avec fenêtre glissante
sentences = []
next_chars = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i + SEQ_LENGTH])
    next_chars.append(text[i + SEQ_LENGTH])

# Conversion numérique optimisée
x = np.zeros((len(sentences), SEQ_LENGTH), dtype=np.int32)
y = np.zeros((len(sentences),), dtype=np.int32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t] = char_to_idx(char)
    y[i] = char_to_idx(next_chars[i])

# Création du dataset TensorFlow avec split explicite
dataset = tf.data.Dataset.from_tensor_slices((x, y))

# Split manuel pour train/validation
dataset = dataset.shuffle(10000)
train_size = int(0.8 * len(x))
val_size = len(x) - train_size

train_dataset = dataset.take(train_size).batch(128).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).batch(128).prefetch(tf.data.AUTOTUNE)
# ---------------------------------------------------
# 3. ARCHITECTURE DU MODÈLE AMÉLIORÉE
# ---------------------------------------------------

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    LSTM(128, kernel_regularizer='l2'),  # Régularisation L2 ajoutée
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(VOCAB_SIZE, activation='softmax')
])


# ---------------------------------------------------
# 4. ENTRAÎNEMENT AVEC OPTIMISATIONS
# ---------------------------------------------------

optimizer = RMSprop(learning_rate=0.005, clipnorm=1.0)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_generator.h5', save_best_only=True)
]


history = model.fit(
    train_dataset,
    epochs=4,
    validation_data=val_dataset,  # Utilisation du dataset de validation explicite
    callbacks=callbacks
)


# ---------------------------------------------------
# 5. GÉNÉRATION DE TEXTE AVEC TOP-K + TEMPÉRATURE
# ---------------------------------------------------

def sample(preds, temperature=1.0, top_k=10):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    # Top-k filtering
    indices = np.argpartition(preds, -top_k)[-top_k:]
    probs = preds[indices]
    probs /= probs.sum()
    return np.random.choice(indices, p=probs)

def generate_text(length, temperature=1.0, top_k=10):
    start_idx = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = []
    sentence = text[start_idx:start_idx + SEQ_LENGTH]
    generated.append(sentence)
    
    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH), dtype=np.int32)
        for t, char in enumerate(sentence[-SEQ_LENGTH:]):
            x_pred[0, t] = char_to_idx(char)
            
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample(preds, temperature, top_k)
        next_char = idx2char[next_idx]
        
        generated.append(next_char)
        sentence = sentence[1:] + next_char
    
    return ''.join(generated)

# ---------------------------------------------------
# 6. TEST AVEC DIFFÉRENTS PARAMÈTRES
# ---------------------------------------------------

params = [
    (0.2, 5),   # Conservateur
    (0.5, 10),  # Équilibré
    (1.0, 15),  # Créatif
    (1.5, 20)   # Très imaginatif
]

for temp, k in params:
    print(f"\n=== Température {temp} / Top-K {k} ===")
    print(generate_text(500, temp, k))
    print("="*50)