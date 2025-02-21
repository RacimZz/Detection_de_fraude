# Projet de Détection de Fraude Documentaire

Ce projet utilise des techniques de traitement d'image et de texte pour détecter les fraudes dans les documents. Le modèle combine l'extraction de caractéristiques d'images et de texte, et utilise des algorithmes de machine learning pour classer les documents comme authentiques ou frauduleux.

## Prérequis

### Bibliothèques Python nécessaires :

- `opencv-python` : pour le traitement d'image
- `pytesseract` : pour l'extraction de texte à partir d'images
- `scikit-learn` : pour l'algorithme de machine learning
- `scikit-image` : pour l'extraction de caractéristiques d'images (HOG, LBP)
- `nltk` : pour le traitement du texte
- `tensorflow` : pour l'entraînement de modèles de réseaux de neurones (optionnel, pour les approches plus avancées)
- `matplotlib` : pour la visualisation des résultats

### Installation des dépendances

Clonez le projet et installez les dépendances avec `pip` :

```bash
git clone <URL-du-projet>
cd <dossier-du-projet>
pip install -r requirements.txt
```

Si vous n'avez pas encore de fichier `requirements.txt`, vous pouvez installer les bibliothèques nécessaires individuellement avec :

```bash
pip install opencv-python pytesseract scikit-learn scikit-image nltk tensorflow matplotlib
```

### Tesseract OCR

Pour l'extraction de texte à partir des images, ce projet utilise **Tesseract OCR**. Vous pouvez télécharger et installer Tesseract depuis [ici](https://github.com/tesseract-ocr/tesseract).

Une fois installé, assurez-vous que Tesseract est ajouté à votre `PATH` système ou spécifiez son chemin dans le script Python comme suit :

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Fonctionnalités

1. **Prétraitement des images** : 
   - Conversion en niveaux de gris
   - Binarisation
   - Amélioration de la qualité de l'image (contraste, bruit)

2. **Extraction de caractéristiques d'image** :
   - Local Binary Pattern (LBP)
   - Histogram of Oriented Gradients (HOG)
   - Optionnel : Réseaux de neurones convolutifs (CNN)

3. **Extraction de texte** :
   - Utilisation de Tesseract OCR pour extraire le texte des images.

4. **Machine Learning** :
   - Classificateur basé sur RandomForest, SVM ou CNN pour prédire si un document est authentique ou frauduleux.

## Utilisation

### 1. Préparer les données

Avant d'exécuter le script, assurez-vous que vos images de test sont bien placées dans les dossiers appropriés (`images/authentiques/` et `images/frauduleuses/`).

### 2. Exécuter le script principal

Vous pouvez exécuter le script `fraud_detection.py` pour entraîner et tester le modèle de détection de fraude :

```bash
python src/fraud_detection.py
```

### 3. Visualiser les résultats

Le modèle générera des résultats, y compris la précision de la détection des documents authentiques et frauduleux. Ces résultats peuvent être affichés avec `matplotlib` ou simplement affichés dans la console.

### 4. Exemple d'extraction de texte depuis une image

Vous pouvez également tester l'extraction de texte à partir d'une image avec le script suivant :

```python
from preprocessing import extract_text_from_image

img_path = 'images/test_document.png'
text = extract_text_from_image(img_path)
print("Texte extrait : ", text)
```

### 5. Entraîner le modèle sur des données spécifiques

Si vous souhaitez entraîner un modèle personnalisé, vous pouvez ajuster le jeu de données et les paramètres du modèle dans `fraud_detection.py` :

```python
from sklearn.ensemble import RandomForestClassifier

# Exemple d'entraînement avec un classificateur RandomForest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)  # Entraînement sur les données prétraitées
```

## Dépannage

- **Tesseract non trouvé** : Si vous recevez une erreur indiquant que Tesseract n'est pas installé, assurez-vous de l'avoir correctement installé et configuré.
- **Problèmes de bibliothèque manquante** : Vérifiez que toutes les dépendances sont installées via `pip install -r requirements.txt`.

## Auteurs

- **Racim ZENATI**

## Licence

Ce projet est sous la licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.
```

---

Ce README est maintenant structuré pour GitHub et prêt à être utilisé dans ton projet. Il comprend toutes les sections nécessaires, avec des titres et sous-titres organisés avec des `#` pour une meilleure lisibilité.
