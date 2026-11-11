# Week 3: Break the Spam Filter

AWS AI/ML Week 3 hands-on task. I trained the simple spam detector from the slides, tried to fool it with 5 messages, explained why it failed, and then tried to improve it.

## Dataset

[SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 5,574 labelled messages (4,827 ham, 747 spam). The data is not stored in this repo. The notebook downloads it the first time it runs.

Some messages contain a `"` character, so I read the file with `quoting=csv.QUOTE_NONE`. Without it, pandas joins neighbour messages and loses 2 rows.

## Method

- Removed 403 duplicate rows (5,171 messages left), so the same message cannot be in train and test.
- Split: train 70% (3,619), validation 15% (776), test 15% (776), stratified, `random_state=42`.
- Baseline: TF-IDF + Multinomial Naive Bayes (the 10-line model from slide 6).
- Improved model: I compared 3 feature types and 4 alpha values on the validation set. I chose character pieces (`char`, 2 to 4 letters) with alpha 0.01. My rule: validation precision at least 0.98, then the highest recall.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (slide 6) | 0.9472 | 1.0000 | 0.5816 | 0.7355 |
| Improved (char, alpha 0.01) | 0.9897 | 0.9787 | 0.9388 | 0.9583 |

A model that always says "ham" gets 0.8737 accuracy, so accuracy alone is not enough. The baseline caught 57 of 98 spam messages and missed 41. The improved model caught 92 and missed 6, but it marked 2 good messages as spam.

## Breaking it

| My message (short) | Baseline said | Improved said | Should be | Why it failed |
|---|---|---|---|---|
| T1 Look-alike (Cyrillic) letters in "call", "claim", "prize" | ham (0.124) | spam | spam | The model saw cаll, clаim and prіze as new words and ignored them. The other words lean ham. The char model fixed it. |
| T2 Friendly message with a hidden spam link | ham (0.001) | ham | spam | It has no spam words. Only normal words like at, still, see, you. Both models failed. |
| T3 Polite business tone | ham (0.105) | ham | spam | Some spam words (cash, reply, gift, account), but many normal words lean ham and are stronger. Both models failed. |
| T4 Real message with spam words | ham (0.071) | ham | ham | Did not fool it. Ham words (come, me, when, if) were stronger than spam words (free, won, reply, stop). |
| T5 Real university notice | ham (0.017) | ham | ham | Did not fool it. Ham words (at, and, the, pay, office) were stronger than free and call. |

Baseline was correct on 2 of 5 tricks. Improved was correct on 3 of 5.

## What I learned

- A word-counting model only knows the words it saw in training. New spelling, strange letters or polite wording can fool it.
- Accuracy can fool me because most messages are ham. I must also check precision and recall.
- Character pieces help with spelling tricks, but not with meaning. For meaning, I want to try embeddings or an LLM next.

## Run it

```bash
python -m venv venv
source venv/Scripts/activate    # Git Bash on Windows
pip install -r requirements.txt
```
Then open `spam_filter.ipynb` in Colab or VS Code and run all cells.
