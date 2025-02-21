import cv2
import pytesseract
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
import tensorflow as tf
from tensorflow.keras import layers, models
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Fonction pour charger et prétraiter l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger l'image à partir de {image_path}")
    
    # Redimensionner pour uniformiser les tailles
    img = cv2.resize(img, (400, 400))
    
    # Améliorer l'image (seuillage pour mieux identifier les caractères)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    return img

# Fonction pour extraire des caractéristiques LBP (Local Binary Pattern)
def extract_lbp_features(image):
    radius = 3
    n_points = 24
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

# Fonction pour extraire des caractéristiques HOG (Histogram of Oriented Gradients)
def extract_hog_features(image):
    # Calculer HOG
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd

# Fonction pour extraire du texte d'une image
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Fonction pour créer un jeu de données d'exemple avec des caractéristiques avancées
def create_dataset(image_paths, labels):
    features = []
    texts = []
    
    for img_path in image_paths:
        img = preprocess_image(img_path)
        lbp_features = extract_lbp_features(img)
        hog_features = extract_hog_features(img)
        text = extract_text_from_image(img_path)
        
        # Ajouter les caractéristiques LBP et HOG dans un vecteur unique
        image_features = np.hstack([lbp_features, hog_features])
        features.append(image_features)
        texts.append(text)
    
    # Traitement du texte avec TF-IDF pour extraire des caractéristiques plus pertinentes
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=100)
    text_features = tfidf_vectorizer.fit_transform(texts).toarray()
    
    # Combinaison des caractéristiques image et texte
    data = np.hstack([features, text_features])
    
    return data, labels

# Exemple de données (remplacer par vos propres images et étiquettes)
image_paths = ["FRAUD.png", "AUTHENTIQUE.png"]  # Exemple de chemins d'images
labels = ["fraudulent", "genuine"]  # "fraudulent" ou "genuine"

# Encode les labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Créer le jeu de données
X, y = create_dataset(image_paths, labels_encoded)

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner un modèle de classification
# Option 1 : Utilisation d'un RandomForest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# Option 2 : Utilisation d'un Support Vector Machine (SVM)
clf_svm = SVC(kernel='linear')
clf_svm.fit(X_train, y_train)

# Option 3 : Utilisation d'un CNN pour une classification d'image plus complexe
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Pour utiliser CNN, on doit préparer des images avec des dimensions adaptées
cnn_images = [cv2.imread(img_path) for img_path in image_paths]
cnn_images_resized = [cv2.resize(img, (64, 64)) for img in cnn_images]  # Resize à 64x64 pixels
cnn_images_normalized = np.array(cnn_images_resized) / 255.0

cnn_model = create_cnn_model((64, 64, 3))
cnn_model.fit(cnn_images_normalized, y, epochs=5, batch_size=2)

# Prédictions
y_pred_rf = clf_rf.predict(X_test)
y_pred_svm = clf_svm.predict(X_test)

# Prédictions avec CNN
cnn_test_images = np.array([cv2.resize(cv2.imread(img), (64, 64)) for img in image_paths]) / 255.0
y_pred_cnn = cnn_model.predict(cnn_test_images)
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)

# Évaluer les résultats
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)

print(f"Précision RandomForest : {accuracy_rf * 100:.2f}%")
print(f"Précision SVM : {accuracy_svm * 100:.2f}%")
print(f"Précision CNN : {accuracy_cnn * 100:.2f}%")
