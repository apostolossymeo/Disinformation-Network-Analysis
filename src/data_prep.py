"""
data_prep.py
------------
Data loading, merging, and feature engineering for the IRA troll tweet dataset.
"""

import ast
import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse


# ─── Timezone categories ────────────────────────────────────────────────────

RU_EE_TZ = {
    "Moscow", "Volgograd", "St. Petersburg", "Yerevan",
    "Baku", "Almaty", "Tashkent", "Tbilisi", "Minsk",
    "Riga", "Vilnius", "Tallinn", "Helsinki", "Bucharest",
    "Sofia", "Warsaw", "Prague", "Budapest", "Bern",
    "Berlin", "Amsterdam", "Paris", "Rome",
}

US_TZ = {
    "Eastern Time (US & Canada)",
    "Central Time (US & Canada)",
    "Mountain Time (US & Canada)",
    "Pacific Time (US & Canada)",
    "Alaska",
    "Hawaii",
    "Arizona",
    "Indiana (East)",
}

def classify_tz(tz):
    if pd.isna(tz):
        return "Unknown"
    if tz in RU_EE_TZ:
        return "Russian/E. European"
    if tz in US_TZ:
        return "US"
    return "Other"


# ─── Parsing helpers ─────────────────────────────────────────────────────────

def safe_parse_list(s):
    """Parse a stringified list safely."""
    if pd.isna(s) or s == "[]":
        return []
    try:
        val = ast.literal_eval(s)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def extract_domain(url_string):
    """Extract root domain from URL string."""
    try:
        parsed = urlparse(url_string)
        domain = parsed.netloc.replace("www.", "")
        return domain if domain else None
    except Exception:
        return None


# ─── Electoral event calendar ────────────────────────────────────────────────

ELECTORAL_EVENTS = {
    "1st Presidential Debate": "2016-09-26",
    "Access Hollywood Tape": "2016-10-07",
    "WikiLeaks Podesta Emails": "2016-10-07",
    "2nd Presidential Debate": "2016-10-09",
    "Comey Letter (FBI)": "2016-10-28",
    "3rd Presidential Debate": "2016-10-19",
    "Election Day": "2016-11-08",
    "Comey Clears Clinton": "2016-11-06",
}


# ─── Main loading function ────────────────────────────────────────────────────

def load_and_merge(tweets_path, users_path):
    """Load, parse, merge, and feature-engineer the full dataset."""

    # ── Load ──────────────────────────────────────────────────────────────────
    tweets = pd.read_csv(tweets_path, index_col=0, low_memory=False)
    users  = pd.read_csv(users_path,  index_col=0, low_memory=False)

    # ── Parse tweet timestamps ─────────────────────────────────────────────
    tweets["ts"] = pd.to_datetime(tweets["created_str"], errors="coerce")
    tweets = tweets.dropna(subset=["ts", "text"])
    tweets["date"]    = tweets["ts"].dt.date
    tweets["year"]    = tweets["ts"].dt.year
    tweets["month"]   = tweets["ts"].dt.month
    tweets["hour"]    = tweets["ts"].dt.hour
    tweets["weekday"] = tweets["ts"].dt.day_name()
    tweets["ym"]      = tweets["ts"].dt.to_period("M")

    # ── Parse user creation date ──────────────────────────────────────────
    users["account_created"] = pd.to_datetime(
        users["created_at"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce"
    )

    # ── Parse list columns ────────────────────────────────────────────────
    tweets["hashtag_list"] = tweets["hashtags"].apply(safe_parse_list)
    tweets["mention_list"] = tweets["mentions"].apply(safe_parse_list)
    tweets["url_list"]     = tweets["expanded_urls"].apply(safe_parse_list)

    tweets["n_hashtags"] = tweets["hashtag_list"].apply(len)
    tweets["n_mentions"] = tweets["mention_list"].apply(len)
    tweets["n_urls"]     = tweets["url_list"].apply(len)
    tweets["is_retweet"] = tweets["retweeted_status_id"].notna()
    tweets["is_reply"]   = tweets["in_reply_to_status_id"].notna()

    # Extract domains from URLs
    tweets["domains"] = tweets["url_list"].apply(
        lambda urls: [extract_domain(u) for u in urls if u]
    )

    # ── Account-level features (from tweets) ──────────────────────────────
    user_stats = tweets.groupby("user_key").agg(
        total_tweets   = ("ts", "count"),
        first_tweet_ts = ("ts", "min"),
        last_tweet_ts  = ("ts", "max"),
        avg_retweets   = ("retweet_count", "mean"),
        max_retweets   = ("retweet_count", "max"),
        rt_share       = ("is_retweet", "mean"),
        reply_share    = ("is_reply", "mean"),
        avg_hashtags   = ("n_hashtags", "mean"),
        avg_mentions   = ("n_mentions", "mean"),
    ).reset_index()

    # ── Merge users with tweet-derived stats ──────────────────────────────
    users = users.merge(user_stats, left_on="screen_name", right_on="user_key", how="left")

    users["dormancy_days"] = (
        users["first_tweet_ts"] - users["account_created"]
    ).dt.days
    users["active_span_days"] = (
        users["last_tweet_ts"] - users["first_tweet_ts"]
    ).dt.days.clip(lower=1)
    users["posting_velocity"] = users["total_tweets"] / users["active_span_days"]
    users["follower_friend_ratio"] = (
        users["followers_count"] / users["friends_count"].replace(0, np.nan)
    )
    users["tz_class"]  = users["time_zone"].apply(classify_tz)
    users["lang_class"] = users["lang"].apply(
        lambda x: "English" if x == "en" else ("Russian" if x == "ru" else "Other")
        if pd.notna(x) else "Unknown"
    )

    return tweets, users
