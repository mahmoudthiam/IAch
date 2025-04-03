#C:/Users/thiam/Desktop/IATech/Strategie-Nationale-de-Developpement-2025-2029.pdf

from shiny import App, ui, render
import fitz  # PyMuPDF
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pathlib
import warnings

# ------------------------- CONFIGURATION -------------------------
# Désactive les warnings inutiles
warnings.filterwarnings("ignore")

# Configuration des chemins
BASE_DIR = pathlib.Path(__file__).parent
PDF_DIR = BASE_DIR / "documents"
PDF_DIR.mkdir(exist_ok=True)  # Crée le dossier s'il n'existe pas

# ------------------------- MODELE & INDEX -------------------------
try:
    # Initialisation du modèle
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Gestion de FAISS (avec fallback sur scikit-learn)
    try:
        import faiss
        FAISS_AVAILABLE = True
        print("FAISS disponible - utilisation de l'indexation accélérée")
    except ImportError:
        FAISS_AVAILABLE = False
        print("FAISS non disponible - utilisation de scikit-learn")
        
except Exception as e:
    print(f"Erreur d'initialisation: {e}")
    model = None
    FAISS_AVAILABLE = False

# ------------------------- FONCTIONS PRINCIPALES -------------------------
def lire_pdfs(dossier):
    """Lit tous les PDFs d'un dossier et retourne leur texte"""
    textes = []
    try:
        for fichier in os.listdir(dossier):
            if fichier.endswith(".pdf"):
                with fitz.open(dossier / fichier) as doc:
                    textes.extend(page.get_text("text").strip() for page in doc if page.get_text("text").strip())
    except Exception as e:
        print(f"Erreur lecture PDF: {e}")
    return textes if textes else ["Aucun document valide trouvé"]

def initialiser_systeme():
    """Initialise les embeddings et le système de recherche"""
    documents = lire_pdfs("C:/Users/thiam/Desktop/IATech/")
    corpus = " ".join(documents)
    
    if model:
        embeddings = model.encode(documents, convert_to_numpy=True)
        dim = embeddings.shape[1]
        
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
        else:
            index = NearestNeighbors(n_neighbors=5, metric="cosine")
            index.fit(embeddings)
    else:
        embeddings = None
        index = None
    
    return documents, corpus, index

# Initialisation
documents, corpus, index = initialiser_systeme()

def rechercher_reponse(question):
    """Trouve la réponse la plus pertinente à une question"""
    if not model:
        return "Système NLP non initialisé"
    
    try:
        # Découpage du corpus
        phrases = [p for p in corpus.split(". ") if p]
        
        # Recherche TF-IDF
        vect = TfidfVectorizer()
        tfidf = vect.fit_transform(phrases + [question])
        scores = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
        
        # Sélection des 3 meilleures réponses
        top_idx = np.argsort(scores)[-3:][::-1]
        return ". ".join(phrases[i] for i in top_idx)
    
    except Exception as e:
        return f"Erreur: {str(e)}"

# ------------------------- INTERFACE UTILISATEUR -------------------------
app_ui = ui.page_fluid(
    ui.tags.style("""
        :root {
            --primary: #00A86B;
            --bg-dark: #121212;
            --bg-light: #1E1E1E;
            --text: white;
        }
        body {
            background: var(--bg-dark);
            color: var(--text);
            font-family: 'Segoe UI', sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat-box {
            background: var(--bg-light);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        textarea, input {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #333;
            background: #2d2d2d;
            color: var(--text);
        }
        button {
            background: var(--primary);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
        }
        .response {
            margin-top: 20px;
            padding: 15px;
            background: #252525;
            border-radius: 5px;
            white-space: pre-wrap;
        }
    """),
    
    ui.div(
        ui.h2("🤖 IATech Assistant", class_="text-center"),
        ui.div(
            ui.input_text_area("question", "Posez votre question:", 
                             placeholder="Comment puis-je vous aider?", 
                             rows=3),
            ui.div(
                ui.input_action_button("submit", "Envoyer", class_="btn-primary"),
                class_="text-center"
            ),
            ui.output_ui("reponse"),
            class_="chat-box"
        ),
        class_="container"
    )
)

# ------------------------- LOGIQUE SERVEUR -------------------------
def server(input, output, session):
    @output
    @render.ui
    def reponse():
        if not input.submit():
            return ui.div("Votre réponse apparaîtra ici...", class_="response")
        
        question = input.question().strip()
        if not question:
            return ui.div("Veuillez poser une question valide", class_="response")
        
        reponse = rechercher_reponse(question)
        return ui.div(reponse, class_="response")

# ------------------------- LANCEMENT -------------------------
app = App(app_ui, server)