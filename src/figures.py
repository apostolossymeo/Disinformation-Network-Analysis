"""
figures.py
----------
All publication-quality figure generation for the IRA analysis.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

# ─── Global style ─────────────────────────────────────────────────────────────

PALETTE = {
    "red":       "#C0392B",
    "blue":      "#1A3A5C",
    "gold":      "#D4A017",
    "mid_grey":  "#7F8C8D",
    "light_grey":"#ECF0F1",
    "dark":      "#2C3E50",
    "teal":      "#1A7A6E",
    "orange":    "#E67E22",
}

CANDIDATE_COLORS = {"Trump": PALETTE["red"], "Clinton": PALETTE["blue"]}

EVENT_COLORS = {
    "1st Debate":           "#2980B9",
    "Access Hollywood /\nWikiLeaks": "#8E44AD",
    "2nd Debate":           "#2980B9",
    "3rd Debate":           "#2980B9",
    "Comey Letter":         "#E74C3C",
    "Comey Clears Clinton": "#27AE60",
    "Election Day":         "#2C3E50",
}


def set_style():
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Georgia", "Times New Roman"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.edgecolor":    "#4A4A4A",
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "grid.color":        "#E0E0E0",
        "grid.linewidth":    0.6,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "legend.frameon":    False,
        "legend.fontsize":   9,
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
    })


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Figure 1: Temporal Architecture ─────────────────────────────────────────

def fig_temporal_architecture(users, tweets, outpath):
    set_style()
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: Account creation timeline
    ax1 = fig.add_subplot(gs[0, :])
    uc = users["account_created"].dropna()
    uc_monthly = uc.dt.to_period("M").value_counts().sort_index()
    uc_monthly.index = uc_monthly.index.to_timestamp()
    ax1.fill_between(uc_monthly.index, uc_monthly.values,
                     alpha=0.7, color=PALETTE["blue"], linewidth=0)
    ax1.plot(uc_monthly.index, uc_monthly.values,
             color=PALETTE["blue"], linewidth=1.5)
    ax1.axvline(pd.Timestamp("2016-11-08"), color=PALETTE["red"],
                linewidth=1.8, linestyle="--", label="Election Day")
    ax1.axvline(pd.Timestamp("2016-01-01"), color=PALETTE["mid_grey"],
                linewidth=1.2, linestyle=":", label="2016 begins", alpha=0.8)
    ax1.set_xlim(pd.Timestamp("2009-01-01"), pd.Timestamp("2018-01-01"))
    ax1.set_title("A  ·  IRA Account Creation Timeline", fontweight="bold", loc="left")
    ax1.set_ylabel("Accounts Created")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Panel B: Tweet volume timeline
    ax2 = fig.add_subplot(gs[1, :])
    tw_daily = tweets.set_index("ts").resample("W")["text"].count()
    ax2.fill_between(tw_daily.index, tw_daily.values,
                     alpha=0.65, color=PALETTE["teal"], linewidth=0)
    ax2.plot(tw_daily.index, tw_daily.values,
             color=PALETTE["teal"], linewidth=1.2)

    # Annotate key events
    event_dates = {
        "Election\nDay":    "2016-11-08",
        "Access\nHollywood": "2016-10-07",
        "Comey\nLetter":    "2016-10-28",
    }
    for label, d in event_dates.items():
        x = pd.Timestamp(d)
        ax2.axvline(x, color=PALETTE["red"], linewidth=1.4,
                    linestyle="--", alpha=0.85)
        y_pos = tw_daily.reindex([x], method="nearest").values
        if len(y_pos):
            ax2.text(x + pd.Timedelta(days=6), float(y_pos[0]) * 1.05,
                     label, fontsize=7.5, color=PALETTE["red"],
                     va="bottom", ha="left")

    ax2.set_xlim(pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01"))
    ax2.set_title("B  ·  Weekly Tweet Volume", fontweight="bold", loc="left")
    ax2.set_ylabel("Tweets per Week")

    fig.suptitle(
        "Temporal Architecture of the IRA Twitter Operation (2014–2017)",
        fontsize=14, fontweight="bold", y=1.01
    )
    save(fig, outpath)


# ─── Figure 2: Behavioral Fingerprinting ─────────────────────────────────────

def fig_behavioral_fingerprint(users, outpath):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Behavioral Fingerprinting of IRA Accounts",
        fontsize=14, fontweight="bold"
    )

    # Panel A: Dormancy distribution
    ax = axes[0, 0]
    dorm = users["dormancy_days"].dropna()
    dorm = dorm[dorm >= 0]
    ax.hist(dorm, bins=30, color=PALETTE["blue"], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(dorm.median(), color=PALETTE["red"], linewidth=2,
               linestyle="--", label=f"Median: {dorm.median():.0f} d")
    ax.axvline(177, color=PALETTE["gold"], linewidth=1.8, linestyle=":",
               label="Symantec benchmark: 177 d")
    ax.set_title("A  ·  Account Dormancy Period", fontweight="bold", loc="left")
    ax.set_xlabel("Days Between Account Creation and First Tweet")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Panel B: Posting velocity
    ax = axes[0, 1]
    vel = users["posting_velocity"].dropna()
    vel = vel[vel < vel.quantile(0.97)]  # clip outliers for visibility
    ax.hist(vel, bins=30, color=PALETTE["teal"], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(vel.median(), color=PALETTE["red"], linewidth=2,
               linestyle="--", label=f"Median: {vel.median():.2f} tw/day")
    ax.set_title("B  ·  Posting Velocity", fontweight="bold", loc="left")
    ax.set_xlabel("Tweets per Active Day")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Panel C: Follower / friend ratio
    ax = axes[1, 0]
    ffr = users["follower_friend_ratio"].dropna()
    ffr = ffr[ffr < ffr.quantile(0.95)]
    ax.hist(ffr, bins=30, color=PALETTE["orange"], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color=PALETTE["dark"], linewidth=1.5,
               linestyle=":", label="Ratio = 1.0 (parity)")
    ax.axvline(ffr.median(), color=PALETTE["red"], linewidth=2,
               linestyle="--", label=f"Median: {ffr.median():.2f}")
    ax.set_title("C  ·  Follower / Friend Ratio", fontweight="bold", loc="left")
    ax.set_xlabel("Followers ÷ Friends")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Panel D: Timezone classification
    ax = axes[1, 1]
    tz_counts = users["tz_class"].value_counts()
    colors_tz = [PALETTE["red"] if "Russian" in t else PALETTE["blue"]
                 if t == "US" else PALETTE["mid_grey"]
                 for t in tz_counts.index]
    bars = ax.barh(tz_counts.index, tz_counts.values,
                   color=colors_tz, alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, tz_counts.values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, color=PALETTE["dark"])
    ax.set_title("D  ·  Account Timezone Classification", fontweight="bold", loc="left")
    ax.set_xlabel("Number of Accounts")

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 3: Language & Timezone Deception ─────────────────────────────────

def fig_timezone_deception(users, outpath):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Timezone Deception: Claimed Location vs. Account Language",
        fontsize=13, fontweight="bold"
    )

    # Panel A: Top timezones bar
    ax = axes[0]
    tz_top = users["time_zone"].value_counts().head(12)
    ru_mask = [t in {
        "Volgograd", "Moscow", "Baku", "Yerevan", "Bern",
        "St. Petersburg", "Berlin", "Amsterdam"
    } for t in tz_top.index]
    bar_colors = [PALETTE["red"] if m else PALETTE["blue"] for m in ru_mask]
    bars = ax.barh(tz_top.index[::-1], tz_top.values[::-1],
                   color=bar_colors[::-1], alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, tz_top.values[::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)
    legend_handles = [
        mpatches.Patch(color=PALETTE["red"],  label="Russian / E. European TZ"),
        mpatches.Patch(color=PALETTE["blue"], label="US / Other TZ"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    ax.set_title("A  ·  Top Account Timezones", fontweight="bold", loc="left")
    ax.set_xlabel("Number of Accounts")

    # Panel B: Language distribution stacked by timezone class
    ax = axes[1]
    cross = pd.crosstab(users["tz_class"], users["lang_class"])
    cross = cross.loc[["US", "Russian/E. European", "Other", "Unknown"]]
    cross.plot(
        kind="bar", ax=ax, stacked=True,
        color=[PALETTE["blue"], PALETTE["red"], PALETTE["mid_grey"]],
        alpha=0.88, edgecolor="white", linewidth=0.4,
        rot=25
    )
    ax.set_title("B  ·  Language Distribution by Timezone Class",
                 fontweight="bold", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Accounts")
    ax.legend(title="Account Language", fontsize=8)

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 4: LDA Topic Model ───────────────────────────────────────────────

def fig_topic_model(lda, vectorizer, topic_labels, outpath, n_words=10):
    set_style()
    n_topics = len(topic_labels)
    ncols = 2
    nrows = (n_topics + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
    axes = axes.flatten()

    feature_names = vectorizer.get_feature_names_out()
    topic_colors = [
        PALETTE["red"], PALETTE["blue"], PALETTE["teal"],
        PALETTE["orange"], PALETTE["gold"], PALETTE["dark"],
        "#8E44AD", "#27AE60"
    ]

    for idx, (topic_id, label) in enumerate(topic_labels.items()):
        ax = axes[idx]
        comp = lda.components_[topic_id]
        top_idx = comp.argsort()[::-1][:n_words]
        top_words = [feature_names[i] for i in top_idx]
        top_weights = comp[top_idx] / comp[top_idx].sum()

        ax.barh(top_words[::-1], top_weights[::-1],
                color=topic_colors[idx % len(topic_colors)],
                alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_title(f"Topic {topic_id + 1}  ·  {label}",
                     fontweight="bold", loc="left", fontsize=11)
        ax.set_xlabel("Normalised Weight")
        ax.tick_params(labelsize=8.5)

    # Hide unused panels
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("LDA Topic Model: Dominant Narrative Clusters in IRA Tweets (n=8)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 5: Topic Prevalence Over Time ────────────────────────────────────

def fig_topic_over_time(topic_time_df, topic_labels, outpath):
    set_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    cols = list(topic_labels.values())
    plot_colors = [
        PALETTE["red"], PALETTE["blue"], PALETTE["teal"],
        PALETTE["orange"], PALETTE["gold"], PALETTE["dark"],
        "#8E44AD", "#27AE60"
    ]

    df = topic_time_df.copy()
    df["ym_ts"] = df["ym"].dt.to_timestamp()
    df_plot = df[df["ym_ts"] >= pd.Timestamp("2015-01-01")]

    x = df_plot["ym_ts"].values
    ys = np.array([df_plot[c].values for c in cols])

    ax.stackplot(x, ys, labels=cols, colors=plot_colors, alpha=0.8)

    ax.axvline(pd.Timestamp("2016-11-08"), color="black",
               linewidth=2, linestyle="--", label="Election Day", zorder=5)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Topic Prevalence Over Time (Monthly Share of IRA Corpus)",
                 fontweight="bold")
    ax.set_ylabel("Share of Tweets")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlabel("")

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 6: VADER Sentiment Time Series ───────────────────────────────────

def fig_sentiment_timeseries(daily_sentiment, outpath):
    from src.sentiment import ELECTORAL_EVENTS
    set_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    for candidate, color in CANDIDATE_COLORS.items():
        if candidate in daily_sentiment.columns:
            s = daily_sentiment[candidate].dropna()
            s_smooth = s.rolling(7, center=True).mean()
            ax.plot(s.index, s.values, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(s_smooth.index, s_smooth.values, color=color,
                    linewidth=2.2, label=f"{candidate} (7-day MA)")

    ax.axhline(0, color=PALETTE["mid_grey"], linewidth=0.8, linestyle=":")

    y_min = daily_sentiment.min().min()
    for label, date in ELECTORAL_EVENTS.items():
        x = pd.Timestamp(date)
        ax.axvline(x, color=PALETTE["dark"], linewidth=1.0,
                   linestyle="--", alpha=0.6)
        ax.text(x, y_min - 0.04, label.replace(" /\n", "\n"),
                fontsize=6.5, ha="center", va="top", color=PALETTE["dark"],
                rotation=0)

    ax.set_xlim(pd.Timestamp("2016-09-01"), pd.Timestamp("2016-11-15"))
    ax.set_title("VADER Sentiment Toward Trump vs. Clinton in IRA Tweets\n"
                 "(September – November 2016, 7-day Moving Average)",
                 fontweight="bold")
    ax.set_ylabel("VADER Compound Score")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 7: Sentiment Distribution (Violin) ───────────────────────────────

def fig_sentiment_violin(tweets, outpath):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Sentiment Distribution: Trump-Mentioning vs. Clinton-Mentioning IRA Tweets",
                 fontweight="bold", fontsize=13)

    t16 = tweets[(tweets["year"] == 2016) & tweets["vader"].notna()].copy()
    trump_s   = t16.loc[t16["text"].str.contains(r"\btrump\b",           case=False, na=False), "vader"]
    clinton_s = t16.loc[t16["text"].str.contains(r"\b(hillary|clinton)\b", case=False, na=False), "vader"]

    plot_data = pd.DataFrame({
        "Candidate": ["Trump"] * len(trump_s) + ["Clinton"] * len(clinton_s),
        "VADER Score": pd.concat([trump_s, clinton_s]).values,
    })

    # Panel A: Violin
    ax = axes[0]
    sns.violinplot(data=plot_data, x="Candidate", y="VADER Score",
                   palette={"Trump": PALETTE["red"], "Clinton": PALETTE["blue"]},
                   inner="box", ax=ax, linewidth=0.8)
    ax.axhline(0, color=PALETTE["mid_grey"], linewidth=0.8, linestyle=":")
    ax.set_title("A  ·  Distribution (Violin + Box)", fontweight="bold", loc="left")

    # Panel B: KDE overlaid
    ax = axes[1]
    for label, data, color in [
        ("Trump",   trump_s,   PALETTE["red"]),
        ("Clinton", clinton_s, PALETTE["blue"]),
    ]:
        kde = gaussian_kde(data, bw_method=0.2)
        xs = np.linspace(-1, 1, 400)
        ax.fill_between(xs, kde(xs), alpha=0.3, color=color)
        ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)
        ax.axvline(data.mean(), color=color, linewidth=1.5, linestyle="--",
                   label=f"{label} mean={data.mean():.3f}")
    ax.axvline(0, color=PALETTE["mid_grey"], linewidth=0.8, linestyle=":")
    ax.set_title("B  ·  Kernel Density Estimate", fontweight="bold", loc="left")
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8.5)

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 8: Top Hashtags ───────────────────────────────────────────────────

def fig_hashtags(ht_df, outpath):
    from src.network import LEAN_COLORS
    set_style()
    fig, ax = plt.subplots(figsize=(11, 9))

    df = ht_df.head(25).copy()
    colors = [LEAN_COLORS.get(l, LEAN_COLORS["Other"]) for l in df["lean"]]

    bars = ax.barh(df["hashtag"][::-1], df["count"][::-1],
                   color=colors[::-1], alpha=0.88, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, df["count"][::-1]):
        ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8.5)

    legend_handles = [
        mpatches.Patch(color=v, label=k) for k, v in LEAN_COLORS.items()
        if k in df["lean"].values
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)
    ax.set_title("Top 25 Hashtags in IRA Tweets, Classified by Political Lean",
                 fontweight="bold")
    ax.set_xlabel("Occurrence Count")
    ax.set_ylabel("")

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 9: Posting Heatmap ───────────────────────────────────────────────

def fig_posting_heatmap(posting_pivot, outpath):
    set_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    sns.heatmap(
        posting_pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        annot=False,
        fmt=".0f",
        cbar_kws={"label": "Tweet Count", "shrink": 0.7},
    )
    ax.set_title("IRA Posting Activity Heatmap: Hour of Day × Day of Week",
                 fontweight="bold")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("")

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 10: Top Domains (URL Amplification) ───────────────────────────────

def fig_domains(domains_df, outpath):
    set_style()
    fig, ax = plt.subplots(figsize=(11, 8))

    DOMAIN_PALETTE = {
        "Right-fringe":  PALETTE["red"],
        "Right-leaning": PALETTE["orange"],
        "Mainstream-R":  "#E8A020",
        "Mainstream-L":  PALETTE["blue"],
        "Wire":          PALETTE["teal"],
        "Russian State": "#8E44AD",
        "Video":         "#27AE60",
        "Social":        PALETTE["mid_grey"],
        "Other":         PALETTE["light_grey"],
    }

    df = domains_df.head(20).copy()
    colors = [DOMAIN_PALETTE.get(c, DOMAIN_PALETTE["Other"]) for c in df["class"]]

    bars = ax.barh(df["domain"][::-1], df["count"][::-1],
                   color=colors[::-1], alpha=0.88, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, df["count"][::-1]):
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8.5)

    legend_handles = [
        mpatches.Patch(color=v, label=k)
        for k, v in DOMAIN_PALETTE.items()
        if k in df["class"].values
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)
    ax.set_title("Top 20 Domains Amplified by IRA Accounts\n(by URL count in tweets)",
                 fontweight="bold")
    ax.set_xlabel("Link Count")

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 11: Account Typology Clustering ───────────────────────────────────

def fig_account_clustering(users, outpath):
    from src.clustering import CLUSTER_COLORS, CLUSTER_DESCRIPTIONS
    set_style()
    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Panel A: PCA scatter
    ax1 = fig.add_subplot(gs[0, 0])
    for label, color in CLUSTER_COLORS.items():
        mask = users["cluster_label"] == label
        if mask.sum() == 0:
            continue
        ax1.scatter(
            users.loc[mask, "pca_x"],
            users.loc[mask, "pca_y"],
            c=color, label=label,
            s=60, alpha=0.78, edgecolors="white", linewidths=0.4
        )
    ax1.set_title("A  ·  Account Typology — PCA Projection (K=4)",
                  fontweight="bold", loc="left")
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.legend(fontsize=8.5, markerscale=1.2)

    # Panel B: Feature profiles (radar / bar comparison)
    ax2 = fig.add_subplot(gs[0, 1])
    features = ["dormancy_days", "posting_velocity",
                "follower_friend_ratio", "rt_share", "avg_retweets"]
    feat_labels = ["Dormancy\n(days)", "Velocity\n(tw/day)",
                   "Follower/\nFriend Ratio", "RT\nShare", "Avg\nRetweets"]
    n_feat = len(features)
    bar_w = 0.18
    x = np.arange(n_feat)

    cluster_labels = [l for l in CLUSTER_COLORS if l in users["cluster_label"].values]
    for i, label in enumerate(cluster_labels):
        mask = users["cluster_label"] == label
        sub = users[mask][features]
        # Normalise to [0, 1] across all accounts for comparability
        all_vals = users[features]
        normed = (sub.mean() - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-9)
        ax2.bar(x + i * bar_w, normed.values,
                width=bar_w, label=label,
                color=CLUSTER_COLORS[label], alpha=0.85,
                edgecolor="white", linewidth=0.4)

    ax2.set_xticks(x + bar_w * 1.5)
    ax2.set_xticklabels(feat_labels, fontsize=8.5)
    ax2.set_title("B  ·  Normalised Feature Profile per Cluster",
                  fontweight="bold", loc="left")
    ax2.set_ylabel("Normalised Score (0–1)")
    ax2.legend(fontsize=7.5, ncol=2)

    fig.suptitle("IRA Account Typology: K-Means Clustering on Behavioral Features",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 12: Lorenz Curve ──────────────────────────────────────────────────

def fig_lorenz_curve(gini_data, outpath):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Amplification Inequality in the IRA Network: Lorenz Curves",
                 fontsize=13, fontweight="bold")

    series_map = {
        "retweet":   ("External Retweets Received", PALETTE["red"]),
        "followers": ("Follower Count", PALETTE["blue"]),
        "tweets":    ("Total Tweets", PALETTE["teal"]),
    }

    for ax, (key, (label, color)) in zip(axes, series_map.items()):
        lx, ly, gini = gini_data[key]
        ax.plot(lx, ly, color=color, linewidth=2.5, label=f"Lorenz curve")
        ax.plot([0, 1], [0, 1], color=PALETTE["mid_grey"],
                linewidth=1.2, linestyle="--", label="Perfect equality")
        ax.fill_between(lx, ly, [0] + list(np.linspace(0, 1, len(lx) - 1)),
                        alpha=0.12, color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(label, fontweight="bold", loc="left")
        ax.set_xlabel("Cumulative Share of Accounts")
        ax.set_ylabel("Cumulative Share of " + label.split(" ")[0])
        ax.text(0.05, 0.88, f"Gini = {gini:.3f}",
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, linewidth=1.2))
        ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 13: Event Spike Detection ────────────────────────────────────────

def fig_spike_detection(spike_df, outpath):
    from src.clustering import ELECTORAL_EVENTS
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={"height_ratios": [2, 1]},
                                    sharex=True)
    fig.suptitle("Event-Driven Spike Detection in IRA Tweet Volume (2016)",
                 fontsize=13, fontweight="bold")

    # Panel A: Volume + rolling mean + spike highlights
    ax1.fill_between(spike_df.index, spike_df["volume"],
                     alpha=0.35, color=PALETTE["teal"])
    ax1.plot(spike_df.index, spike_df["volume"],
             color=PALETTE["teal"], linewidth=0.9, alpha=0.7)
    ax1.plot(spike_df.index, spike_df["roll_mean"],
             color=PALETTE["dark"], linewidth=2, label="7-day rolling mean")

    # Shade ±1σ band
    ax1.fill_between(
        spike_df.index,
        spike_df["roll_mean"] - spike_df["roll_std"],
        spike_df["roll_mean"] + spike_df["roll_std"],
        alpha=0.15, color=PALETTE["dark"], label="±1σ band"
    )

    # Highlight spikes
    spikes = spike_df[spike_df["is_spike"]]
    ax1.scatter(spikes.index, spikes["volume"],
                color=PALETTE["red"], zorder=5, s=40,
                label=f"Spike (|z| > 2.5)", edgecolors="white", linewidths=0.5)

    # Electoral event lines
    for label, date in ELECTORAL_EVENTS.items():
        x = pd.Timestamp(date)
        if x in spike_df.index or (x >= spike_df.index.min() and x <= spike_df.index.max()):
            ax1.axvline(x, color=PALETTE["red"], linewidth=1.0,
                        linestyle="--", alpha=0.65)
            ax1.text(x + pd.Timedelta(days=1),
                     spike_df["volume"].max() * 0.95,
                     label, fontsize=6.5, color=PALETTE["red"],
                     va="top", ha="left", rotation=0)

    ax1.set_title("A  ·  Daily Tweet Volume with Spike Detection",
                  fontweight="bold", loc="left")
    ax1.set_ylabel("Tweets per Day")
    ax1.legend(fontsize=8.5, loc="upper left")

    # Panel B: Z-score
    ax2.axhline(0, color=PALETTE["mid_grey"], linewidth=0.8)
    ax2.axhline(2.5,  color=PALETTE["red"], linewidth=1.0,
                linestyle=":", alpha=0.7, label="Threshold (z=±2.5)")
    ax2.axhline(-2.5, color=PALETTE["red"], linewidth=1.0, linestyle=":", alpha=0.7)
    ax2.fill_between(spike_df.index,
                     spike_df["z_score"].clip(-5, 5), 0,
                     where=spike_df["z_score"] > 2.5,
                     color=PALETTE["red"], alpha=0.5)
    ax2.fill_between(spike_df.index,
                     spike_df["z_score"].clip(-5, 5), 0,
                     where=spike_df["z_score"] < -2.5,
                     color=PALETTE["blue"], alpha=0.5)
    ax2.plot(spike_df.index, spike_df["z_score"].clip(-5, 5),
             color=PALETTE["dark"], linewidth=1.2)
    ax2.set_title("B  ·  Z-Score (Rolling 7-day baseline)",
                  fontweight="bold", loc="left")
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=8)
    ax2.set_ylim(-5, 5)

    for label, date in ELECTORAL_EVENTS.items():
        x = pd.Timestamp(date)
        ax2.axvline(x, color=PALETTE["red"], linewidth=1.0,
                    linestyle="--", alpha=0.65)

    fig.tight_layout()
    save(fig, outpath)


# ─── Figure 14: Retweet Network Graph ────────────────────────────────────────

def fig_retweet_network(edges_df, users, outpath):
    import networkx as nx
    set_style()
    fig, ax = plt.subplots(figsize=(13, 12))
    ax.set_facecolor("#F8F8F8")
    fig.set_facecolor("#F8F8F8")

    # Build directed graph — only internal IRA retweets
    ira_accounts = set(users["screen_name"].str.lower().dropna())
    internal = edges_df[
        edges_df["rt_source"].str.lower().isin(ira_accounts) &
        edges_df["user_key"].str.lower().isin(ira_accounts)
    ].copy()

    if len(internal) < 5:
        # Fallback: use all edges
        internal = edges_df.copy()

    # Keep top edges by weight for readability
    internal = internal.nlargest(200, "weight")

    G = nx.DiGraph()
    for _, row in internal.iterrows():
        G.add_edge(row["user_key"], row["rt_source"], weight=row["weight"])

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "Insufficient internal retweet edges\nfor network graph",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        save(fig, outpath)
        return

    # Node size ∝ total retweet volume received
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    # Map cluster labels to colors
    node_cluster = dict(zip(
        users["screen_name"].str.lower(),
        users["cluster_label"]
    ))
    from src.clustering import CLUSTER_COLORS
    node_colors = [
        CLUSTER_COLORS.get(node_cluster.get(n, ""), PALETTE["mid_grey"])
        for n in G.nodes()
    ]
    node_sizes = [
        max(40, min(800, in_degree.get(n, 1) * 30))
        for n in G.nodes()
    ]

    pos = nx.spring_layout(G, k=1.8, iterations=60, seed=42)

    # Draw edges
    edges = G.edges(data=True)
    edge_weights = [d.get("weight", 1) for _, _, d in edges]
    max_w = max(edge_weights) if edge_weights else 1
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[0.3 + 1.5 * (w / max_w) for w in edge_weights],
        alpha=0.35,
        edge_color=PALETTE["mid_grey"],
        arrows=True, arrowsize=8,
        connectionstyle="arc3,rad=0.1"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.88,
        linewidths=0.5,
        edgecolors="white"
    )

    # Label top-5 by in-degree only
    top_nodes = sorted(in_degree, key=in_degree.get, reverse=True)[:5]
    label_dict = {n: n for n in top_nodes if n in G.nodes}
    nx.draw_networkx_labels(
        G, pos, labels=label_dict, ax=ax,
        font_size=7, font_color=PALETTE["dark"],
        font_weight="bold"
    )

    # Legend for clusters
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=9, label=l)
        for l, c in CLUSTER_COLORS.items()
        if l in (node_cluster.get(n) for n in G.nodes())
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=8.5, title="Account Type", title_fontsize=9)

    ax.set_title(
        "Internal IRA Retweet Network\n"
        "(Node size ∝ inbound retweet weight; colour = account typology cluster)",
        fontweight="bold", fontsize=12
    )
    ax.axis("off")
    fig.tight_layout()
    save(fig, outpath)
