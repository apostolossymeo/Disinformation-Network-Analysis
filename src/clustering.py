"""
clustering.py
-------------
Account typology clustering (K-means + PCA),
Gini / Lorenz curve for amplification inequality,
and z-score event spike detection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import zscore


# ─── Account Typology Clustering ─────────────────────────────────────────────

CLUSTER_LABELS = {
    0: "Amplifier Bots",
    1: "Cultural Infiltrators",
    2: "Human Operators",
    3: "Dormant Nodes",
}

CLUSTER_COLORS = {
    "Amplifier Bots":       "#C0392B",
    "Cultural Infiltrators":"#8E44AD",
    "Human Operators":      "#1A3A5C",
    "Dormant Nodes":        "#7F8C8D",
}

CLUSTER_DESCRIPTIONS = {
    "Amplifier Bots":
        "High posting velocity, high retweet share, low original content. "
        "Mechanistic amplification of other accounts' messages.",
    "Cultural Infiltrators":
        "Moderate activity, English-language, cultural/entertainment content. "
        "Builds authentic-seeming followings before political deployment.",
    "Human Operators":
        "Lower velocity, higher follower-to-friend ratio, higher engagement. "
        "Likely human-operated accounts with strategic narrative function.",
    "Dormant Nodes":
        "Minimal activity, very low engagement. Pre-registered placeholder "
        "accounts or accounts retired after brief operational use.",
}


def cluster_accounts(users: pd.DataFrame, n_clusters: int = 4,
                     random_state: int = 42) -> pd.DataFrame:
    """
    K-means clustering of IRA accounts on behavioral features.
    Returns users DataFrame with 'cluster_id' and 'cluster_label' columns.
    """
    features = [
        "dormancy_days", "posting_velocity",
        "follower_friend_ratio", "rt_share", "avg_retweets",
    ]
    df = users[features].copy()
    df = df.fillna(df.median())

    # Clip extreme outliers (>99th pctile) before scaling
    for col in features:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # PCA for visualisation (2 components)
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # K-means
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=20, max_iter=500)
    labels = km.fit_predict(X)

    # Silhouette score
    sil = silhouette_score(X, labels)

    # Assign semantic labels by inspecting cluster centroids
    # Sort clusters by posting_velocity (ascending) to get consistent assignment
    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=features
    )
    # Map raw cluster IDs → semantic labels by velocity rank
    # (lowest velocity = Dormant, highest = Amplifier)
    velocity_order = centroids["posting_velocity"].rank().astype(int) - 1
    semantic_map = {
        velocity_order[velocity_order == 3].index[0]: "Amplifier Bots",
        velocity_order[velocity_order == 2].index[0]: "Human Operators",
        velocity_order[velocity_order == 1].index[0]: "Cultural Infiltrators",
        velocity_order[velocity_order == 0].index[0]: "Dormant Nodes",
    }

    users = users.copy()
    users["cluster_id"]    = labels
    users["cluster_label"] = users["cluster_id"].map(semantic_map)
    users["pca_x"]         = X_pca[:, 0]
    users["pca_y"]         = X_pca[:, 1]

    return users, centroids, pca, sil


def cluster_summary(users: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per cluster."""
    features = [
        "dormancy_days", "posting_velocity",
        "follower_friend_ratio", "rt_share",
        "avg_retweets", "total_tweets",
    ]
    summary = (
        users.groupby("cluster_label")[features]
        .agg(["mean", "median"])
        .round(2)
    )
    counts = users["cluster_label"].value_counts().rename("n_accounts")
    return summary, counts


# ─── Lorenz Curve + Gini Coefficient ─────────────────────────────────────────

def lorenz_gini(series: pd.Series):
    """
    Compute Lorenz curve and Gini coefficient for a non-negative series.
    Returns (lorenz_x, lorenz_y, gini).
    """
    vals = np.sort(series.dropna().clip(lower=0).values)
    n = len(vals)
    if n == 0 or vals.sum() == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.0
    cum_vals  = np.cumsum(vals)
    lorenz_y  = np.concatenate([[0], cum_vals / cum_vals[-1]])
    lorenz_x  = np.linspace(0, 1, n + 1)
    # Gini = 1 - 2 * area under Lorenz curve (trapezoid rule)
    area = np.trapezoid(lorenz_y, lorenz_x) if hasattr(np, "trapezoid") else np.trapz(lorenz_y, lorenz_x)
    gini = 1 - 2 * area
    return lorenz_x, lorenz_y, round(gini, 4)


def amplification_gini(users: pd.DataFrame) -> dict:
    """Gini coefficients for retweet and follower distributions."""
    rt_x, rt_y, rt_gini = lorenz_gini(users["avg_retweets"].fillna(0))
    fl_x, fl_y, fl_gini = lorenz_gini(users["followers_count"].fillna(0))
    tw_x, tw_y, tw_gini = lorenz_gini(users["total_tweets"].fillna(0))
    return {
        "retweet":   (rt_x, rt_y, rt_gini),
        "followers": (fl_x, fl_y, fl_gini),
        "tweets":    (tw_x, tw_y, tw_gini),
    }


# ─── Event Spike Detection ────────────────────────────────────────────────────

ELECTORAL_EVENTS = {
    "1st Presidential\nDebate":  "2016-09-26",
    "Access Hollywood\n/ WikiLeaks": "2016-10-07",
    "2nd Presidential\nDebate":  "2016-10-09",
    "3rd Presidential\nDebate":  "2016-10-19",
    "Comey Letter\n(FBI)":       "2016-10-28",
    "Comey Clears\nClinton":     "2016-11-06",
    "Election Day":              "2016-11-08",
}


def detect_spikes(tweets: pd.DataFrame,
                  window: int = 7,
                  z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Z-score-based spike detection on daily tweet volume.
    Returns daily DataFrame with volume, rolling stats, z-scores, and spike flag.
    """
    daily = tweets.set_index("ts").resample("1D")["text"].count().rename("volume")
    daily = daily[(daily.index >= "2016-01-01") & (daily.index <= "2017-01-01")]

    roll_mean = daily.rolling(window, center=True, min_periods=3).mean()
    roll_std  = daily.rolling(window, center=True, min_periods=3).std()

    z = (daily - roll_mean) / roll_std.replace(0, np.nan)

    df = pd.DataFrame({
        "volume":     daily,
        "roll_mean":  roll_mean,
        "roll_std":   roll_std,
        "z_score":    z,
        "is_spike":   z.abs() > z_threshold,
    })
    return df
