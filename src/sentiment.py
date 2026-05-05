"""
sentiment.py
------------
VADER sentiment analysis + event-driven time series.
"""

import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import stats

_ANALYZER = SentimentIntensityAnalyzer()

ELECTORAL_EVENTS = {
    "1st Debate":           "2016-09-26",
    "Access Hollywood /\nWikiLeaks": "2016-10-07",
    "2nd Debate":           "2016-10-09",
    "3rd Debate":           "2016-10-19",
    "Comey Letter":         "2016-10-28",
    "Comey Clears Clinton": "2016-11-06",
    "Election Day":         "2016-11-08",
}


def vader_score(text: str) -> float:
    return _ANALYZER.polarity_scores(str(text))["compound"]


def score_dataframe(tweets: pd.DataFrame, text_col: str = "text",
                    sample_n: int = None) -> pd.Series:
    """Return compound VADER scores. Optionally sample for speed."""
    if sample_n and len(tweets) > sample_n:
        idx = tweets.sample(sample_n, random_state=42).index
        scores = pd.Series(np.nan, index=tweets.index)
        scores.loc[idx] = tweets.loc[idx, text_col].apply(vader_score)
        return scores
    return tweets[text_col].apply(vader_score)


def candidate_sentiment(tweets: pd.DataFrame,
                        sentiment_col: str = "vader") -> pd.DataFrame:
    """
    Filter tweets mentioning each candidate (by regex) and
    return daily mean compound sentiment per candidate.
    Only 2016 tweets, Oct–Nov window for precision.
    """
    t16 = tweets[tweets["year"] == 2016].copy()

    trump_mask   = t16["text"].str.contains(
        r"\btrump\b", case=False, na=False, regex=True)
    clinton_mask = t16["text"].str.contains(
        r"\b(hillary|clinton)\b", case=False, na=False, regex=True)

    def daily_mean(mask):
        return (
            t16.loc[mask, ["ts", sentiment_col]]
            .set_index("ts")
            .resample("1D")[sentiment_col]
            .mean()
        )

    trump_daily   = daily_mean(trump_mask).rename("Trump")
    clinton_daily = daily_mean(clinton_mask).rename("Clinton")

    df = pd.concat([trump_daily, clinton_daily], axis=1).dropna(how="all")
    return df


def mann_whitney_sentiment(tweets: pd.DataFrame,
                           sentiment_col: str = "vader") -> dict:
    """Mann-Whitney U test comparing Trump-mentioning vs Clinton-mentioning tweets."""
    t16 = tweets[(tweets["year"] == 2016) & tweets[sentiment_col].notna()].copy()

    trump_scores = t16.loc[
        t16["text"].str.contains(r"\btrump\b", case=False, na=False), sentiment_col
    ]
    clinton_scores = t16.loc[
        t16["text"].str.contains(r"\b(hillary|clinton)\b", case=False, na=False),
        sentiment_col,
    ]

    u_stat, p_val = stats.mannwhitneyu(
        trump_scores.dropna(), clinton_scores.dropna(), alternative="two-sided"
    )
    # effect size: rank-biserial correlation
    n1, n2 = len(trump_scores.dropna()), len(clinton_scores.dropna())
    r = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "Trump N": n1,
        "Clinton N": n2,
        "Trump mean": round(trump_scores.mean(), 4),
        "Clinton mean": round(clinton_scores.mean(), 4),
        "U statistic": round(u_stat, 0),
        "p-value": f"{p_val:.2e}",
        "Effect size (r)": round(r, 4),
    }
