# Project 23: News Summariser with Evaluation

Comparing a simple baseline (Lead-3) with a pretrained model (BART) for news summarisation, using ROUGE scores, human ratings and error analysis.

## 1. Problem

**Question:** Does a pretrained model (BART) write better news summaries than a simple Lead-3 baseline, measured by ROUGE scores and human ratings?

**Why it matters:** People do not have time to read full news articles, so good automatic summaries are useful. But a summary that matches a reference by word overlap is not always a good summary, so it is important to check with more than one method.

## 2. Data

- **Dataset:** CNN/DailyMail, version 3.0.0, from Hugging Face: https://huggingface.co/datasets/abisee/cnn_dailymail
- **What is in it:** news articles from CNN and the Daily Mail, each with highlights written by the news site. These highlights are used as the human reference summary.
- **Sample used:** 50 randomly chosen articles from the test split (random seed 42).
- **Human rating subset:** 20 of those 50 articles (40 summaries in total: 20 Lead-3 and 20 BART), also chosen with seed 42.
- **Privacy:** only public data was used. No private or personal records. The dataset itself is not included in this repository. The notebook downloads it automatically.

## 3. Methods

| Method | What it does |
|---|---|
| **Lead-3 (baseline)** | Takes the first 3 sentences of the article as the summary. |
| **BART** (`facebook/bart-large-cnn`) | Pretrained model that writes a summary. Settings: max_length=130, min_length=30, num_beams=4, input cut at 1024 tokens. |

**Why Lead-3 as baseline:** it is very simple, and in news the first sentences usually hold the main facts, so it is a strong baseline that is hard to beat.

## 4. Evaluation

**ROUGE (automatic):** compares a summary with the human reference by counting matching words.
- ROUGE-1: matching single words
- ROUGE-2: matching word pairs
- ROUGE-L: longest matching run of words in order
- I report the F1 score (balance of precision and recall), averaged over the 50 articles.

**Human ratings:** 3 raters (friends, not experts) each rated all 40 summaries from 1 (very bad) to 5 (very good) on three questions:
- **Relevance:** does the summary keep the same main points as the reference?
- **Fluency:** is it easy to read, with correct English?
- **Accuracy:** are the facts, names and numbers correct?

Rows were shuffled and the method column was hidden, so raters did not know which method wrote which summary. Raters compared each summary with the reference summary (not the full article). After all ratings were collected, I matched them to the methods using a private answer key (`answer_key.csv`). The analysis is in `analyze_ratings.ipynb`.

**Error analysis:** I read the 10 BART summaries with the lowest ROUGE-L and grouped the problems. The table is also in the main notebook (`error_table`).

## 5. Results

### ROUGE (average over 50 articles)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead-3 (baseline) | 0.3845 | 0.1502 | 0.2370 |
| BART | 0.4191 | 0.1756 | 0.2796 |
| Difference | +0.0346 | +0.0254 | +0.0426 |

BART scored higher than the baseline on all three ROUGE metrics (about +9% on ROUGE-1, +17% on ROUGE-2 and +18% on ROUGE-L). Re-running the notebook with the same seed gave the same numbers.

### Human ratings (average of 3 raters, 1 to 5)

| Method | Relevance | Fluency | Accuracy |
|---|---|---|---|
| Lead-3 (baseline) | 4.63 | 4.93 | 4.58 |
| BART | 4.45 | 4.93 | 4.43 |

- Lead-3 scored slightly higher than BART on relevance and accuracy. Fluency was equal.
- I compared the two methods article by article (average of 3 raters). For relevance, BART was better on 6 articles, worse on 9 and tied on 5. A Wilcoxon test gave p = 0.24 for relevance and p = 0.72 for accuracy, so **the difference is not statistically significant**. Human ratings show no clear winner.
- **Raters agreed well.** For accuracy, pairs of raters gave exactly the same score 72% to 92% of the time. For relevance it was 72% to 75%. For fluency it was 88% to 98%, but almost every score was 5, so there is little spread. Nearly all pairs of ratings were within 1 point of each other.
- BART got a low accuracy score (1 or 2) on 4 of 60 ratings. Lead-3 never got a score below 3.

### ROUGE and humans disagree

ROUGE favoured BART, while humans rated Lead-3 slightly higher (not significantly). A likely reason is that Lead-3 copies real sentences from the article, so its facts and English are almost always correct. BART writes more of its own wording, which helps ROUGE only if the words match the reference, and which can sometimes introduce errors. ROUGE counts matching words but does not check facts or meaning, so both measures are needed.

## 6. Error analysis

I read the 10 BART summaries with the lowest ROUGE-L (0.144 to 0.202). One summary can have more than one problem.

| Error type | Count (of 10) | Examples |
|---|---|---|
| Different focus from the reference | 8 | Miliband/SNP, Corinthian, Hillsborough, Cassey Ho, Obama, NYC train, Peru mummy, Memphis boy |
| Key point or result missing | 4 | Miliband (Sturgeon's support deal), Corinthian (student debts should be forgiven), Barcelona (never says they reached the semi-final), Mayweather's chef (never says what the "weaknesses" are) |
| Minor or unsuitable detail | 3 | Obama golfing, Barcelona offside goal, offensive tweet quotes in the Memphis summary |
| Copies sentences from the article | 4 | Obama, NYC train, Mayweather's chef, Memphis boy |
| Repetition | 0 | None found |

### Examples

- **Miliband/SNP (ROUGE-L 0.144):** BART focuses on Miliband's speech and the poll numbers. It leaves out the main story: Sturgeon offering to support Miliband, her demand for full fiscal autonomy, and the warned £7 billion gap.
- **Corinthian (0.152):** BART focuses on different points than the reference and misses that student debts should be forgiven. BART also names a different author (Gloria Borger) than the reference (David Wheeler).
- **Barcelona (0.190):** BART describes the goals but never says Barcelona reached the Champions League semi-finals, which is the main result.
- **Mayweather's chef (0.198):** BART says the chef "revealed his weaknesses" but never says what they are (Twizzlers, noodles, fried hotdogs).
- **Hillsborough (0.159):** The summary is accurate and readable, but it leaves out the phone number, website and Home Office inquiry, so ROUGE is low.

### What I learned

- A low ROUGE-L does not always mean a bad summary. Several of the lowest-scoring summaries (Hillsborough, Cassey Ho, NYC train, Peru mummy) are readable and mostly accurate. They score low because BART chose different details or different words than the human writer.
- The real problems are missing main points or results.
- BART copied sentences from the article in at least 4 of the 10 cases. This may explain why humans found BART and Lead-3 so similar.

## 7. Limitations

- Small sample: 50 articles for ROUGE, only 20 articles (40 summaries) for human ratings.
- Only 3 raters, who are not experts.
- Raters compared summaries with the short reference, not the full article. A true fact in a summary that is missing from the reference might have been marked down.
- Scores are very high for both methods (most are 4 or 5), so it is hard to see differences. Fluency has almost no spread.
- The error analysis used the 10 lowest ROUGE-L scores, which is not a random sample of BART summaries.
- I could only check the first part of each article when judging the facts in the error analysis.
- Only English news from one dataset was tested.
- BART cuts articles longer than 1024 tokens.

## 8. How to run

1. Install Python and Jupyter Notebook.
2. Install the libraries: `pip install -r requirements.txt`
3. Open `project23_summariser.ipynb` and run all cells from top to bottom. This creates `answer_key.csv` and `human_rating_sheet_for_raters.xlsx`.
4. Give the rater sheet to raters. When you have the filled files (`rater1.xlsx`, `rater2.xlsx`, `rater3.xlsx`), open `analyze_ratings.ipynb` and run all cells. This creates `human_rating_results.csv`.

**Reproducibility:** random seed = 42, N = 50 articles. Library versions are in `requirements.txt`. The first run downloads the dataset and the model (about 1.5 GB), so internet is needed. Note that running step 3 again overwrites `answer_key.csv`, so keep a backup of the original if you reuse the existing rater files.

## 9. Files

- `project23_summariser.ipynb`: main code (data, Lead-3, BART, ROUGE, rating sheet, error analysis)
- `analyze_ratings.ipynb`: analysis of the human ratings
- `human_rating_sheet_for_raters.xlsx`: the file given to raters
- `rater1.xlsx`, `rater2.xlsx`, `rater3.xlsx`: the filled ratings from the 3 raters
- `answer_key.csv`: matches each rated row to Lead-3 or BART
- `human_rating_results.csv`: average human ratings per method
- `requirements.txt`: library versions
- `README.md`: this file
