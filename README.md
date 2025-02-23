# 🎭 Générateur de Texte Shakespeareien avec LSTM
    Un modèle de génération de texte capable d'écrire dans le style de Shakespeare, utilisant un réseau de neurones LSTM bidirectionnel. Parfait pour explorer la génération de texte créatif avec l'IA !

# ✨ Fonctionnalités
- Génération de texte pseudo-shakespearien

- Contrôle de la créativité via température (0.1 à 2.0)

- Mécanisme Top-K sampling pour plus de diversité

- Modèle LSTM bidirectionnel avec couches d'embedding

- Prétraitement intelligent du texte (gestion des caractères spéciaux)

- Early stopping et validation automatique

# 📦 Installation
Clonez le dépôt :

    https://github.com/JAAFOURAMouez/shakespeare-generator.git
    cd shakespeare-generator

# 🚀 Utilisation
Entraînement du modèle :

    python3 main.py --train --epochs 30 --batch_size 64

Génération de texte :

    python3 main.py --generate --length 500 --temperature 0.7 --top_k 15
