import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download required resources
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# Define your synonym groups
SYNONYM_GROUPS = [
    {"quick", "fast", "rapid", "speedy", "swift", "prompt"},
    {"intelligent", "smart", "clever", "bright", "brilliant", "wise"},
    {"happy", "joyful", "cheerful", "glad", "delighted", "content", "pleased"},
    {"sad", "unhappy", "sorrowful", "depressed", "miserable", "down"},
    {"angry", "mad", "furious", "irate", "annoyed", "outraged"},
    {"car", "automobile", "vehicle", "ride"},
    {"job", "work", "occupation", "profession", "career"},
    {"house", "home", "residence", "dwelling", "abode"},
    {"big", "large", "huge", "gigantic", "massive", "enormous"},
    {"small", "little", "tiny", "miniature", "petite"},
    {"start", "begin", "commence", "initiate", "launch"},
    {"end", "finish", "conclude", "terminate", "complete"},
    {"run", "sprint", "jog", "dash"},
    {"walk", "stroll", "saunter", "amble"},
    {"see", "observe", "view", "watch", "spot", "glimpse"},
    {"say", "tell", "speak", "utter", "state", "declare"},
    {"think", "ponder", "consider", "reflect", "contemplate"},
    {"eat", "consume", "devour", "ingest", "feast"},
    {"help", "assist", "aid", "support", "serve"},
    {"buy", "purchase", "acquire", "obtain", "procure"},
    {"beautiful", "pretty", "gorgeous", "lovely", "attractive", "stunning"},
    {"ugly", "unattractive", "hideous", "unsightly"},
    {"important", "significant", "crucial", "vital", "essential"},
    {"hard", "difficult", "challenging", "tough"},
    {"easy", "simple", "effortless", "straightforward"},
]


# Build word-to-representative map
SYNONYM_MAP = {}
for group in SYNONYM_GROUPS:
    rep = sorted(group)[0]  # Choose first word alphabetically as standard
    for word in group:
        SYNONYM_MAP[word] = rep

def normalize_tokens(tokens):
    normalized = []
    for token in tokens:
        if token in stop_words or token in string.punctuation:
            continue
        token_lower = token.lower()
        # Substitute with synonym if in map
        normalized_word = SYNONYM_MAP.get(token_lower, token_lower)
        normalized.append(normalized_word)
    return normalized

def preprocess(text):
    tokens = word_tokenize(text)
    normalized = normalize_tokens(tokens)
    return ' '.join(normalized)

def calculate_similarity(para1, para2):
    # Preprocess with synonym normalization
    para1_clean = preprocess(para1)
    para2_clean = preprocess(para2)

    # Vectorize using CountVectorizer 
    vectorizer = CountVectorizer().fit([para1_clean, para2_clean])
    vecs = vectorizer.transform([para1_clean, para2_clean])

    # Cosine similarity
    return cosine_similarity(vecs[0], vecs[1])[0][0]

if __name__ == "__main__":
    para1 = input("Enter the first sentence:\n")
    para2 = input("Enter the second sentence:\n")

    score = calculate_similarity(para1, para2)


    print("sentence 1:", para1 ,"\n", "sentence 2:",para2)
    print(f"\n🧠 Cosine Similarity Score (with synonym normalization): {score:.4f}")
