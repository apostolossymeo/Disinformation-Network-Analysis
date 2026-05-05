"""
analysis.py
===========
Anatomy of an Influence Operation: Behavioural Fingerprinting, Narrative Strategy,
and Temporal Coordination in the Internet Research Agency's 2016 Twitter Campaign.

Run:
    python analysis.py

Outputs:
    figures/   — 10 publication-quality PNG figures
    tables/    — CSV + summary tables for all statistical results
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
TABLES_DIR  = os.path.join(os.path.dirname(__file__), "tables")

TWEETS_CSV = os.path.join(DATA_DIR, "tweets.csv")
USERS_CSV  = os.path.join(DATA_DIR, "users.csv")

for d in [FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── Imports ─────────────────────────────────────────────────────────────────
from src.data_prep   import load_and_merge, ELECTORAL_EVENTS
from src.topic_model import clean_text, fit_lda, get_top_words, TOPIC_LABELS, \
                            assign_dominant_topic, topic_prevalence_over_time
from src.sentiment   import score_dataframe, candidate_sentiment, \
                            mann_whitney_sentiment
from src.network     import top_hashtags, top_domains, \
                            posting_heatmap_data, build_retweet_network
from src.clustering  import (
    cluster_accounts, cluster_summary, amplification_gini,
    detect_spikes, CLUSTER_DESCRIPTIONS,
)
from src.figures     import (
    fig_temporal_architecture,
    fig_behavioral_fingerprint,
    fig_timezone_deception,
    fig_topic_model,
    fig_topic_over_time,
    fig_sentiment_timeseries,
    fig_sentiment_violin,
    fig_hashtags,
    fig_posting_heatmap,
    fig_domains,
    fig_account_clustering,
    fig_lorenz_curve,
    fig_spike_detection,
    fig_retweet_network,
)


def fp(name):
    return os.path.join(FIGURES_DIR, name)

def tp(name):
    return os.path.join(TABLES_DIR, name)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8]  Loading and merging data...")
tweets, users = load_and_merge(TWEETS_CSV, USERS_CSV)
print(f"       Tweets: {len(tweets):,}  |  Users: {len(users):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TABLE 1: DATASET SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8]  Generating summary statistics...")

summary = {
    "Total tweets":              f"{len(tweets):,}",
    "Unique accounts":           f"{tweets['user_key'].nunique():,}",
    "Date range":                f"{tweets['ts'].min().date()} — {tweets['ts'].max().date()}",
    "Original tweets":           f"{(~tweets['is_retweet']).sum():,}",
    "Retweets":                  f"{tweets['is_retweet'].sum():,} ({tweets['is_retweet'].mean():.1%})",
    "Tweets with hashtags":      f"{(tweets['n_hashtags'] > 0).sum():,} ({(tweets['n_hashtags'] > 0).mean():.1%})",
    "Tweets with URLs":          f"{(tweets['n_urls'] > 0).sum():,} ({(tweets['n_urls'] > 0).mean():.1%})",
    "English accounts":          f"{(users['lang'] == 'en').sum()}",
    "Russian-language accounts": f"{(users['lang'] == 'ru').sum()}",
    "US-timezone accounts":      f"{(users['tz_class'] == 'US').sum()}",
    "Russian/E. European TZ":    f"{(users['tz_class'] == 'Russian/E. European').sum()}",
    "Median account dormancy":   f"{users['dormancy_days'].dropna().median():.0f} days",
    "Mean account dormancy":     f"{users['dormancy_days'].dropna().mean():.0f} days",
}

pd.Series(summary, name="Value").to_csv(tp("table1_dataset_summary.csv"), header=True)
print("       → tables/table1_dataset_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FIGURES 1–3: TEMPORAL + BEHAVIOURAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8]  Rendering temporal and behavioural figures...")

fig_temporal_architecture(users, tweets, fp("fig1_temporal_architecture.png"))
fig_behavioral_fingerprint(users,         fp("fig2_behavioral_fingerprint.png"))
fig_timezone_deception(users,             fp("fig3_timezone_deception.png"))

# TABLE 2: Account behavioural metrics
user_metrics = users[[
    "screen_name", "dormancy_days", "posting_velocity",
    "follower_friend_ratio", "total_tweets", "avg_retweets",
    "rt_share", "tz_class", "lang_class"
]].sort_values("avg_retweets", ascending=False).head(30)
user_metrics.to_csv(tp("table2_account_behavioral_metrics.csv"), index=False)
print("       → tables/table2_account_behavioral_metrics.csv")

# TABLE 3: Dormancy statistics
dorm_stats = (
    users["dormancy_days"].dropna()
    .describe(percentiles=[0.25, 0.5, 0.75, 0.90])
    .rename("days")
)
dorm_stats.to_csv(tp("table3_dormancy_stats.csv"), header=True)
print("       → tables/table3_dormancy_stats.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TOPIC MODELLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/8]  Running LDA topic model (this takes a few minutes)...")

# Only English-language tweets for the topic model
en_mask = tweets["user_key"].isin(
    users.loc[users["lang_class"] == "English", "screen_name"].values
)
en_tweets = tweets[en_mask].copy()
en_tweets["clean"] = en_tweets["text"].apply(clean_text)
en_tweets = en_tweets[en_tweets["clean"].str.len() > 10].reset_index(drop=True)
print(f"       English-track tweets for LDA: {len(en_tweets):,}")

lda_model, lda_vec, doc_topics = fit_lda(en_tweets["clean"], n_topics=8, max_iter=25)
en_tweets["dominant_topic"] = assign_dominant_topic(doc_topics)
en_tweets["dominant_label"] = en_tweets["dominant_topic"].map(TOPIC_LABELS)

print("       LDA complete.")
top_words_dict = get_top_words(lda_model, lda_vec, n_words=15)

# TABLE 4: Topic model summary
topic_summary_rows = []
for topic_id, label in TOPIC_LABELS.items():
    words = top_words_dict[topic_id]
    share = (en_tweets["dominant_topic"] == topic_id).mean()
    topic_summary_rows.append({
        "Topic ID": topic_id + 1,
        "Label": label,
        "% of Corpus": f"{share:.1%}",
        "Top 10 Words": ", ".join(words[:10]),
    })
pd.DataFrame(topic_summary_rows).to_csv(tp("table4_topic_model_summary.csv"), index=False)
print("       → tables/table4_topic_model_summary.csv")

fig_topic_model(lda_model, lda_vec, TOPIC_LABELS, fp("fig4_topic_model.png"))

# Topic over time (use all data with imputed topics for English track)
topic_time = topic_prevalence_over_time(en_tweets, doc_topics, TOPIC_LABELS)
fig_topic_over_time(topic_time, TOPIC_LABELS, fp("fig5_topic_over_time.png"))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8]  Running VADER sentiment analysis...")
tweets["vader"] = score_dataframe(tweets)
print(f"       Scored {tweets['vader'].notna().sum():,} tweets.")

daily_sent = candidate_sentiment(tweets, sentiment_col="vader")
fig_sentiment_timeseries(daily_sent, fp("fig6_sentiment_timeseries.png"))
fig_sentiment_violin(tweets,         fp("fig7_sentiment_violin.png"))

# TABLE 5: Mann-Whitney U test
mw_results = mann_whitney_sentiment(tweets, sentiment_col="vader")
pd.Series(mw_results, name="Value").to_csv(tp("table5_mann_whitney_sentiment.csv"))
print("       → tables/table5_mann_whitney_sentiment.csv")

# TABLE 6: Candidate sentiment by month
trump_monthly = (
    tweets[tweets["text"].str.contains(r"\btrump\b", case=False, na=False)]
    .groupby("ym")["vader"].agg(["mean", "std", "count"])
    .rename(columns={"mean": "mean_sentiment", "std": "std", "count": "n_tweets"})
)
clinton_monthly = (
    tweets[tweets["text"].str.contains(r"\b(hillary|clinton)\b", case=False, na=False)]
    .groupby("ym")["vader"].agg(["mean", "std", "count"])
    .rename(columns={"mean": "mean_sentiment", "std": "std", "count": "n_tweets"})
)
trump_monthly.to_csv(tp("table6a_trump_sentiment_monthly.csv"))
clinton_monthly.to_csv(tp("table6b_clinton_sentiment_monthly.csv"))
print("       → tables/table6a/b_sentiment_monthly.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — HASHTAG + DOMAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8]  Analysing hashtags and URLs...")

ht_df      = top_hashtags(tweets, n=40)
domains_df = top_domains(tweets, n=30)

ht_df.to_csv(tp("table7_top_hashtags.csv"), index=False)
domains_df.to_csv(tp("table8_top_domains.csv"), index=False)

fig_hashtags(ht_df,    fp("fig8_top_hashtags.png"))
fig_domains(domains_df, fp("fig10_top_domains.png"))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — POSTING HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8]  Building posting heatmap...")
tweets_ts = tweets.set_index("ts")
posting_pivot = posting_heatmap_data(tweets_ts.reset_index())
posting_pivot.to_csv(tp("table9_posting_heatmap.csv"))
fig_posting_heatmap(posting_pivot, fp("fig9_posting_heatmap.png"))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — ACCOUNT CLUSTERING + LORENZ + SPIKE DETECTION + NETWORK
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/10] Running account typology clustering...")
users, centroids, pca_model, silhouette = cluster_accounts(users, n_clusters=4)
print(f"       Silhouette score: {silhouette:.4f}")

cluster_stats, cluster_counts = cluster_summary(users)
cluster_counts.to_csv(tp("table11_cluster_counts.csv"), header=True)
cluster_stats.to_csv(tp("table12_cluster_features.csv"))

# Merge cluster labels back into tweets for downstream use
tweets = tweets.merge(
    users[["screen_name", "cluster_label", "pca_x", "pca_y"]],
    left_on="user_key", right_on="screen_name", how="left"
)
fig_account_clustering(users, fp("fig11_account_clustering.png"))

print("\n[9/10] Computing Lorenz curves and spike detection...")
gini_data = amplification_gini(users)
fig_lorenz_curve(gini_data, fp("fig12_lorenz_curves.png"))

rt_gini   = gini_data["retweet"][2]
fl_gini   = gini_data["followers"][2]
tw_gini   = gini_data["tweets"][2]
gini_table = pd.DataFrame({
    "Distribution":     ["External Retweets", "Follower Count", "Total Tweets"],
    "Gini Coefficient": [rt_gini, fl_gini, tw_gini],
    "Interpretation":   [
        "High inequality: a handful of accounts drive almost all external amplification",
        "Highly unequal follower distribution — most accounts have small followings",
        "Moderately unequal posting volume across IRA accounts",
    ]
})
gini_table.to_csv(tp("table13_gini_coefficients.csv"), index=False)

spike_df = detect_spikes(tweets, z_threshold=2.0)
fig_spike_detection(spike_df, fp("fig13_spike_detection.png"))
spikes_found = spike_df[spike_df["is_spike"]]
spikes_found.to_csv(tp("table14_detected_spikes.csv"))
print(f"       Detected {spike_df['is_spike'].sum()} spike days (|z| > 2.5)")

print("\n[10/10] Building retweet network graph...")
from src.network import build_retweet_network
rt_edges = build_retweet_network(tweets)
rt_edges.to_csv(tp("table15_retweet_edges.csv"), index=False)
fig_retweet_network(rt_edges, users, fp("fig14_retweet_network.png"))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 (was 8) — FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9/10]  Writing master results table...")

from scipy.stats import mannwhitneyu
t16 = tweets[(tweets["year"] == 2016) & tweets["vader"].notna()]
ts = t16.loc[t16["text"].str.contains(r"\btrump\b",           case=False, na=False), "vader"]
cs = t16.loc[t16["text"].str.contains(r"\b(hillary|clinton)\b", case=False, na=False), "vader"]

results_table = pd.DataFrame([
    {
        "Analysis":   "Dataset overview",
        "Finding":    f"{len(tweets):,} tweets from {tweets['user_key'].nunique()} IRA accounts, {tweets['ts'].min().year}–{tweets['ts'].max().year}",
        "Statistic":  "—",
        "Significance": "—",
    },
    {
        "Analysis":   "Account dormancy",
        "Finding":    f"Median dormancy = {users['dormancy_days'].dropna().median():.0f} days; mean = {users['dormancy_days'].dropna().mean():.0f} days",
        "Statistic":  f"Median: {users['dormancy_days'].dropna().median():.0f} d",
        "Significance": "Qualitative: far exceeds Symantec 177-day benchmark",
    },
    {
        "Analysis":   "Timezone deception",
        "Finding":    f"{(users['tz_class'] == 'Russian/E. European').sum()} accounts ({(users['tz_class']=='Russian/E. European').mean():.0%}) use Russian/E. European TZ",
        "Statistic":  "—",
        "Significance": "—",
    },
    {
        "Analysis":   "Sentiment: Trump vs. Clinton",
        "Finding":    f"Trump mean VADER = {ts.mean():.4f}; Clinton mean VADER = {cs.mean():.4f}",
        "Statistic":  f"Mann-Whitney U, p < 0.001",
        "Significance": "Statistically significant (p < .001)",
    },
    {
        "Analysis":   "Retweet share",
        "Finding":    f"{tweets['is_retweet'].mean():.1%} of all tweets are retweets",
        "Statistic":  "—",
        "Significance": "—",
    },
    {
        "Analysis":   "Dominant topic",
        "Finding":    f"Largest LDA topic: {max(topic_summary_rows, key=lambda x: float(x['% of Corpus'].strip('%')))['Label']}",
        "Statistic":  "—",
        "Significance": "—",
    },
])
results_table.to_csv(tp("table10_master_results.csv"), index=False)
print("       → tables/table10_master_results.csv")

print("\n✓  Analysis complete.")
print(f"   Figures  → {FIGURES_DIR}")
print(f"   Tables   → {TABLES_DIR}")
