"""
topic_model.py
--------------
Text cleaning, LDA topic modeling, and topic-over-time analysis.
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Hardcoded English stopwords (no network download required)
_STOP_EN = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn","also","even","much","still","though","every","any","else",
}

# Simple rule-based lemmatizer (no download required)
def _simple_lemma(word):
    """Very basic English lemmatization for common suffixes."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if stem.endswith(stem[-1]) and len(stem) > 3:
            return stem[:-1]
        return stem
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("er") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word
_EXTRA_STOP = {
    "rt", "amp", "via", "http", "https", "co", "t", "u", "s", "n",
    "would", "could", "make", "get", "got", "like", "one", "said",
    "say", "know", "think", "people", "new", "time", "year", "day",
    "want", "going", "come", "way", "look", "us", "im", "th", "re",
    "dont", "doesnt", "its", "ive", "weve", "youre", "theyre",
    "trump", "hillary", "clinton", "donald",   # remove candidates: topics about *content*
}
STOP_ALL = _STOP_EN | _EXTRA_STOP


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)   # keep hashtag word
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        _simple_lemma(w)
        for w in text.split()
        if w not in STOP_ALL and len(w) > 2
    ]
    return " ".join(tokens)


def fit_lda(docs: pd.Series, n_topics: int = 8, n_features: int = 6000,
            max_iter: int = 20, random_state: int = 42):
    """Fit LDA and return (model, vectorizer, doc-topic matrix)."""
    vec = CountVectorizer(
        max_features=n_features,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    dtm = vec.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method="online",
        batch_size=4096,
        random_state=random_state,
        n_jobs=-1,
    )
    doc_topics = lda.fit_transform(dtm)
    return lda, vec, doc_topics


def get_top_words(lda, vectorizer, n_words: int = 12):
    """Return top words per topic as a dict."""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:n_words]
        topics[idx] = [feature_names[i] for i in top_idx]
    return topics


# Hand-assigned topic labels based on top words (adjust after inspection)
TOPIC_LABELS = {
    0: "Black Culture / Targeting",
    1: "Anti-Obama / Email Scandal",
    2: "White House / Political",
    3: "Tea Party / Conservative",
    4: "Police / Anti-Media",
    5: "Tea Party / Anti-Islam",
    6: "Religious / Ambiguous",
    7: "Pro-Trump / MAGA",
}


def assign_dominant_topic(doc_topics: np.ndarray) -> np.ndarray:
    return doc_topics.argmax(axis=1)


def topic_prevalence_over_time(tweets: pd.DataFrame, doc_topics: np.ndarray,
                                topic_labels: dict) -> pd.DataFrame:
    """Monthly topic share DataFrame."""
    df = tweets[["ym"]].copy().reset_index(drop=True)
    for i, label in topic_labels.items():
        df[label] = doc_topics[:, i]
    df = df.groupby("ym")[list(topic_labels.values())].mean().reset_index()
    # normalise each row to sum to 1
    cols = list(topic_labels.values())
    df[cols] = df[cols].div(df[cols].sum(axis=1), axis=0)
    return df
