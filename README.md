# Anatomy of an Influence Operation
## Behavioural Fingerprinting, Narrative Strategy, and Temporal Coordination in the Internet Research Agency's 2016 Twitter Campaign

---

### Overview

This repository contains the full computational analysis pipeline for a study of the Internet Research Agency's (IRA) Twitter influence operation during the 2016 U.S. presidential election. Using 203,461 tweets from 454 confirmed IRA-affiliated accounts (Kaggle / NBC dataset), the analysis pursues four research questions:

1. **RQ1 — Behavioural Fingerprinting**: What account-level signatures — dormancy period, posting velocity, follower/friend asymmetry, timezone mismatch — characterise the IRA network?
2. **RQ2 — Narrative Strategy**: What content clusters structured the IRA's messaging, and how did topic prevalence shift across the electoral timeline?
3. **RQ3 — Candidate Targeting**: How did IRA sentiment toward Trump vs. Clinton differ, and does this difference hold under statistical testing?
4. **RQ4 — Operational Architecture**: What posting patterns and amplification structures reveal coordinated campaign behaviour?

---

### Key Findings

| Finding | Statistic |
|---|---|
| Median account dormancy | **781 days** (mean: 626 days) |
| Accounts with Russian/E. European timezone | **104 (23%)** |
| Russian-language accounts | **90 (20%)** despite mostly US location claims |
| Trump-mentioning tweets VADER score | **−0.009** (near-neutral) |
| Clinton-mentioning tweets VADER score | **−0.076** (negative) |
| Sentiment difference (Mann-Whitney U) | p = 9.14 × 10⁻⁵² |
| Retweet share | **19.5%** of all original posts |
| Dominant LDA topic | Black Culture / Targeting (15.6% of English-track corpus) |

**The dormancy finding is particularly striking.** A median of 781 days between account creation and first IRA-linked tweet — over two years — indicates a pre-registration infrastructure established long before the 2016 electoral cycle. This substantially exceeds the 177-day benchmark documented in Symantec's IRA analysis and suggests systematic, multi-year operational preparation rather than reactive opportunism.

**The sentiment asymmetry is statistically robust.** Trump-mentioning tweets cluster near neutral (VADER ≈ −0.009), while Clinton-mentioning tweets are systematically more negative (VADER ≈ −0.076). The Mann-Whitney U test (p < 10⁻⁵¹) confirms this is not a sampling artefact, consistent with the dual-strategy hypothesis: bolstering Trump while depressing Clinton.

**The Black Culture targeting cluster** replicates documented IRA tactics: using entertainment, music, and cultural content to build authentic-seeming followings in Black Twitter communities before deploying divisive political content.

---

### Repository Structure

```
ira_analysis/
├── analysis.py               # Main pipeline — run this
├── src/
│   ├── data_prep.py          # Data loading, merging, feature engineering
│   ├── topic_model.py        # LDA topic modelling (8 topics)
│   ├── sentiment.py          # VADER scoring + Mann-Whitney tests
│   ├── network.py            # Hashtag, URL, and posting heatmap analysis
│   └── figures.py            # All publication-quality figure generation
├── figures/                  # Generated PNG figures (10 total)
├── tables/                   # Generated CSV tables (10 total)
├── data/                     # Place tweets.csv and users.csv here
│   └── README_data.md        # Data acquisition instructions
├── requirements.txt
└── README.md
```

---

### Data

The dataset is not included in this repository. Download it from:
- **Kaggle**: [Russian Troll Tweets](https://www.kaggle.com/datasets/vikasg/russian-troll-tweets)
- **Original source**: NBC News / FiveThirtyEight IRA dataset

Place `tweets.csv` and `users.csv` in the `data/` directory before running.

---

### Installation

```bash
git clone https://github.com/your-username/ira_analysis.git
cd ira_analysis
pip install -r requirements.txt
python analysis.py
```

---

### Figures

| Figure | Description |
|---|---|
| `fig1_temporal_architecture` | Account creation + weekly tweet volume timeline with electoral events |
| `fig2_behavioral_fingerprint` | Dormancy, velocity, follower ratio, timezone distributions |
| `fig3_timezone_deception` | Timezone vs. claimed location breakdown |
| `fig4_topic_model` | LDA topic word-weight charts (8 topics) |
| `fig5_topic_over_time` | Monthly topic prevalence stacked area chart |
| `fig6_sentiment_timeseries` | VADER time series for Trump/Clinton with event annotations |
| `fig7_sentiment_violin` | Sentiment distribution comparison (violin + KDE) |
| `fig8_top_hashtags` | Top 25 hashtags by political lean |
| `fig9_posting_heatmap` | Hour × weekday posting intensity heatmap |
| `fig10_top_domains` | Top 20 amplified domains by source classification |

---

### Methodological Notes

**Topic Modelling.** LDA was fitted on 35,912 English-track tweets (273 English-language accounts) using a CountVectorizer with bigram support (n-gram range 1–2), vocabulary of 6,000 features, min document frequency of 5, and max document frequency of 0.85. Eight topics were selected; `online` learning with batch size 4,096 was used for computational efficiency. Stop words include standard English stopwords plus Twitter-specific noise (`rt`, `amp`) and the candidate names themselves (to surface *content* themes rather than mention patterns).

**Sentiment Analysis.** VADER (Valence Aware Dictionary and sEntiment Reasoner) was chosen over AFINN for its calibration to social media text, including handling of capitalisation, punctuation, and emoticons. All 203,461 tweets were scored. Candidate-specific analyses isolate tweets containing `\btrump\b` and `\b(hillary|clinton)\b` respectively.

**Dormancy.** Computed as days between `users.created_at` (Twitter account creation) and the earliest tweet timestamp for that account in the dataset. Accounts with no matched tweets in the IRA dataset are excluded from dormancy calculations.

**Statistical testing.** The Mann-Whitney U test was selected over Student's t-test due to non-normality of VADER compound score distributions (bimodal structure visible in Fig. 7). Effect size reported as rank-biserial correlation r.

---

### References

- Bovet, A. & Makse, H.A. (2019). Influence of fake news in Twitter during the 2016 US presidential election. *Nature Communications*.
- Linvill, D.L. & Warren, P.L. (2020). Troll factories: Manufacturing specialized disinformation on Twitter. *Political Communication*.
- Badawy, A. et al. (2018). Analyzing the Digital Traces of Political Manipulation. *IEEE/ACM ASONAM*.
- Golovchenko, Y. et al. (2020). Cross-Platform State Propaganda. *Political Communication*.
- Bail, C.A. et al. (2019). Assessing the Russian Internet Research Agency's impact. *PNAS*.

---

### Citation

If you use this analysis, please cite the original dataset (NBC News / FiveThirtyEight) and reference the relevant primary research literature listed above.
