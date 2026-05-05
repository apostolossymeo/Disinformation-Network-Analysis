"""
network.py
----------
Hashtag campaign analysis, URL domain amplification,
and retweet/mention network analysis.
"""

import re
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats


# ─── Hashtag analysis ────────────────────────────────────────────────────────

# Manual political lean labels for top hashtags (populated after inspection)
HASHTAG_LEAN = {
    # Pro-Trump / Right
    "maga": "Pro-Trump", "trump": "Pro-Trump", "trump2016": "Pro-Trump",
    "makeamericagreatagain": "Pro-Trump", "trumptrain": "Pro-Trump",
    "draintheswamp": "Pro-Trump", "americafirst": "Pro-Trump",
    "deplorable": "Pro-Trump", "deplorables": "Pro-Trump",
    "votetrump": "Pro-Trump", "trumppence16": "Pro-Trump",
    # Anti-Clinton
    "crookedhillary": "Anti-Clinton", "lockherup": "Anti-Clinton",
    "hillary": "Anti-Clinton", "clintonemails": "Anti-Clinton",
    "neverhillary": "Anti-Clinton", "imwithher": "Anti-Clinton",
    "hillaryclinton": "Anti-Clinton",
    # Race / Division
    "blacklivesmatter": "Race/BLM", "blm": "Race/BLM",
    "blacktwitter": "Race/BLM", "tcot": "Conservative",
    # Anti-Islam / Terror
    "islamkills": "Anti-Islam", "stopislam": "Anti-Islam",
    "banislam": "Anti-Islam", "norefugees": "Anti-Islam",
    "islamterror": "Anti-Islam",
    # Election integrity
    "voterfraud": "Election Integrity", "riggedelection": "Election Integrity",
    "draintheswamp": "Pro-Trump",
    # Misc political
    "electionday": "Neutral", "election2016": "Neutral",
    "debate": "Neutral", "presidentialdebate": "Neutral",
}

LEAN_COLORS = {
    "Pro-Trump":        "#C0392B",
    "Anti-Clinton":     "#E74C3C",
    "Race/BLM":         "#8E44AD",
    "Anti-Islam":       "#D35400",
    "Election Integrity": "#F39C12",
    "Conservative":     "#2980B9",
    "Neutral":          "#7F8C8D",
    "Other":            "#BDC3C7",
}


def top_hashtags(tweets: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    all_ht = [h.lower() for lst in tweets["hashtag_list"] for h in lst if h]
    counts = Counter(all_ht).most_common(n)
    df = pd.DataFrame(counts, columns=["hashtag", "count"])
    df["lean"] = df["hashtag"].map(lambda h: HASHTAG_LEAN.get(h, "Other"))
    return df


def hashtag_timeline(tweets: pd.DataFrame,
                     hashtags_of_interest: list,
                     freq: str = "W") -> pd.DataFrame:
    """Weekly count of specific hashtags."""
    records = []
    tweets_copy = tweets.copy()
    tweets_copy["ht_lower"] = tweets_copy["hashtag_list"].apply(
        lambda lst: [h.lower() for h in lst]
    )
    for ht in hashtags_of_interest:
        mask = tweets_copy["ht_lower"].apply(lambda lst: ht.lower() in lst)
        sub = tweets_copy.loc[mask, "ts"].resample(freq).size()
        sub.name = ht
        records.append(sub)
    return pd.concat(records, axis=1).fillna(0)


# ─── URL / domain analysis ────────────────────────────────────────────────────

DOMAIN_CLASS = {
    # Known right-wing / fringe
    "breitbart.com": "Right-fringe", "infowars.com": "Right-fringe",
    "dailycaller.com": "Right-leaning", "theblaze.com": "Right-leaning",
    "wnd.com": "Right-fringe", "truthfeed.com": "Right-fringe",
    "conservativetreehouse.com": "Right-fringe",
    # Mainstream
    "foxnews.com": "Mainstream-R", "cnn.com": "Mainstream-L",
    "nytimes.com": "Mainstream-L", "washingtonpost.com": "Mainstream-L",
    "nbcnews.com": "Mainstream-L", "msnbc.com": "Mainstream-L",
    "reuters.com": "Wire", "apnews.com": "Wire",
    # Social / aggregator
    "youtube.com": "Video", "youtu.be": "Video",
    "twitter.com": "Social", "facebook.com": "Social",
    "reddit.com": "Social",
    # Russian state
    "rt.com": "Russian State", "sputniknews.com": "Russian State",
    "ria.ru": "Russian State",
}


def top_domains(tweets: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    all_domains = [d for lst in tweets["domains"] for d in lst if d]
    counts = Counter(all_domains).most_common(n)
    df = pd.DataFrame(counts, columns=["domain", "count"])
    df["class"] = df["domain"].map(lambda d: DOMAIN_CLASS.get(d, "Other"))
    return df


# ─── Network (coordination) analysis ─────────────────────────────────────────

def build_retweet_network(tweets: pd.DataFrame):
    """
    Build directed retweet network among IRA accounts.
    Returns edge list DataFrame.
    """
    rt = tweets[tweets["is_retweet"]].copy()
    # Extract retweeted user from text: "RT @username:"
    rt["rt_source"] = rt["text"].str.extract(r"^RT @(\w+):", expand=False)
    rt = rt.dropna(subset=["rt_source", "user_key"])
    edges = rt.groupby(["user_key", "rt_source"]).size().reset_index(name="weight")
    return edges


def posting_heatmap_data(tweets: pd.DataFrame) -> pd.DataFrame:
    """Return hour-of-day × day-of-week posting counts."""
    t = tweets[["hour", "weekday"]].copy()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        t.groupby(["weekday", "hour"]).size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    return pivot


def coordination_score(tweets: pd.DataFrame,
                       window_minutes: int = 15) -> pd.DataFrame:
    """
    Per-account: fraction of tweets posted within `window_minutes` of another
    account's tweet. High score ≈ coordinated posting.
    """
    t = tweets[["user_key", "ts"]].sort_values("ts").copy()
    t["ts_epoch"] = t["ts"].astype(np.int64) // 10**9
    window_sec = window_minutes * 60

    scores = {}
    users = t["user_key"].unique()
    # Vectorised: for each account, count how many of its tweets land
    # within `window_sec` of any tweet from a *different* account.
    all_epochs = t["ts_epoch"].values
    for user in users:
        mask = t["user_key"] == user
        user_epochs = t.loc[mask, "ts_epoch"].values
        other_epochs = t.loc[~mask, "ts_epoch"].values
        hits = 0
        for ep in user_epochs:
            diffs = np.abs(other_epochs - ep)
            if diffs.min() <= window_sec:
                hits += 1
        scores[user] = hits / len(user_epochs) if len(user_epochs) > 0 else 0

    df = pd.Series(scores, name="coordination_score").reset_index()
    df.columns = ["user_key", "coordination_score"]
    return df.sort_values("coordination_score", ascending=False)
