import argparse
import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Fonction pour préparer les données
def prepare_data():
    # Téléchargement du dataset complet
    filepath = tf.keras.utils.get_file(
        'shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    )

    # Nettoyage et normalisation du texte
    text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
    text = re.sub(r'[^\w\s]', '', text)  # Suppression de la ponctuation
    text = re.sub(r'\s+', ' ', text)     # Suppression des espaces multiples

    # Paramètres
    VOCAB_SIZE = 50    # Taille maximale du vocabulaire (top 50 caractères)
    SEQ_LENGTH = 60    # Longueur des séquences
    STEP_SIZE = 3      # Pas de décalage

    # Création du vocabulaire limité
    char_counts = {char: text.count(char) for char in set(text)}
    sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])[:VOCAB_SIZE-1]
    chars = [char for char, _ in sorted_chars] + ['<UNK>']  # Ajout d'un token inconnu

    char2idx = {char: i for i, char in enumerate(chars)}
    idx2char = {i: char for i, char in enumerate(chars)}

    # Fonction de conversion avec gestion des caractères inconnus
    def char_to_idx(char):
        return char2idx[char] if char in char2idx else char2idx['<UNK>']

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
    dataset = dataset.shuffle(10000)
    train_size = int(0.8 * len(x))
    val_size = len(x) - train_size

    train_dataset = dataset.take(train_size).batch(128).prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(train_size).batch(128).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, VOCAB_SIZE, SEQ_LENGTH, char2idx, idx2char, text

# Fonction pour créer et compiler le modèle
def create_model(vocab_size, seq_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.3),
        LSTM(128, kernel_regularizer='l2'),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])

    optimizer = RMSprop(learning_rate=0.005, clipnorm=1.0)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

# Fonction pour entraîner le modèle
def train_model(model, train_dataset, val_dataset, epochs, batch_size):
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('best_generator.h5', save_best_only=True)
    ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    return model

# Fonction pour générer du texte
def generate_text(model, char2idx, idx2char, text, seq_length, length, temperature, top_k):
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

    start_idx = random.randint(0, len(text) - seq_length - 1)
    generated = []
    sentence = text[start_idx:start_idx + seq_length]
    generated.append(sentence)

    for _ in range(length):
        x_pred = np.zeros((1, seq_length), dtype=np.int32)
        for t, char in enumerate(sentence[-seq_length:]):
            x_pred[0, t] = char2idx[char]  # Correction ici : utilisation de char2idx[char]

        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample(preds, temperature, top_k)
        next_char = idx2char[next_idx]

        generated.append(next_char)
        sentence = sentence[1:] + next_char

    return ''.join(generated)

def main():
    parser = argparse.ArgumentParser(description='Générateur de texte basé sur Shakespeare')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle')
    parser.add_argument('--generate', action='store_true', help='Générer du texte')
    parser.add_argument('--length', type=int, default=500, help='Longueur du texte généré')
    parser.add_argument('--temperature', type=float, default=1.0, help='Température pour le sampling')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k pour le sampling')
    parser.add_argument('--model_path', type=str, default='best_generator.h5', help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--epochs', type=int, default=4, help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--batch_size', type=int, default=128, help='Taille des lots d\'entraînement')

    args = parser.parse_args()

    if args.train:
        train_dataset, val_dataset, vocab_size, seq_length, char2idx, idx2char, text = prepare_data()
        model = create_model(vocab_size, seq_length)
        model = train_model(model, train_dataset, val_dataset, args.epochs, args.batch_size)
        model.save(args.model_path)
    elif args.generate:
        model = load_model(args.model_path)
        _, _, vocab_size, seq_length, char2idx, idx2char, text = prepare_data()
        generated_text = generate_text(model, char2idx, idx2char, text, seq_length, args.length, args.temperature, args.top_k)
        print(generated_text)
    else:
        print("Veuillez spécifier --train pour entraîner le modèle ou --generate pour générer du texte.")

if __name__ == "__main__":
    main()