# BankToken 🏦

A domain-specific BPE tokenizer trained on financial corpora, implemented from scratch in pure Python.

BankToken achieves **7.7% lower token fertility than GPT-2** and **9.3% lower than BERT** on held-out financial text — using a vocabulary 3x smaller — by training exclusively on financial language where terminology like *EBITDA*, *cryptocurrency*, and *macroeconomic* appear as single tokens instead of being fragmented into 3.

---

## Why this exists

General-purpose tokenizers like GPT-2 and BERT are trained on broad web corpora — Reddit, Wikipedia, books. They have never seen *EBITDA* or *collateralised* frequently enough to merge them into single tokens. Every extra token a model processes costs compute in the attention mechanism (O(n²) complexity). A tokenizer that produces fewer tokens for your domain means shorter sequences, cheaper inference, and a model that can fit more context into the same window.

This project demonstrates that **domain-specific training data compensates for vocabulary size** — a 16,000-token finance vocabulary outperforms a 50,257-token general vocabulary on financial text.

---

## Results

Benchmark on 10 held-out financial sentences not seen during training:

| Tokenizer | Vocab Size | Avg Fertility ↓ | vs BankToken |
|-----------|-----------|----------------|--------------|
| GPT-2     | 50,257    | 1.362          | +8.3% more tokens |
| BERT      | 30,522    | 1.385          | +10.2% more tokens |
| **BankToken** | **16,000** | **1.257**  | baseline |

*Fertility = tokens produced / words in text. Lower is better.*

### Word-level breakdown

| Word | GPT-2 | BERT | BankToken |
|------|-------|------|-----------|
| EBITDA | `E\|BIT\|DA` (3) | `e\|##bit\|##da` (3) | `ebitda</w>` **(1)** |
| cryptocurrency | `crypt\|oc\|urrency` (3) | `crypt\|##oc\|##ur\|##ren\|##cy` (5) | `cryptocurrency</w>` **(1)** |
| macroeconomic | `mac\|ro\|economic` (3) | `macro\|##economic` (2) | `macroeconomic</w>` **(1)** |
| recessionary | `re\|cession\|ary` (3) | `recession\|##ary` (2) | `recession\|ary</w>` (2) |
| collateralised | `coll\|ateral\|ised` (3) | `collateral\|##ised` (2) | `col\|lat\|er\|al\|ised</w>` (5) |

GPT-2 requires 5536 merge operations before learning `ebitda` as a unit. BankToken reaches it because the entire training budget is focused on financial text — the algorithm encounters *ebitda* frequently enough for it to become a merge target.

**Honest tradeoff:** Common English words like *quantitative* split into more tokens in BankToken (3) than GPT-2 (2). A domain tokenizer trades general vocabulary coverage for domain-specific compression.

---

## Implementation

Built from scratch in ~200 lines of pure Python. No tokenizer libraries used — the goal was to understand the algorithm, not wrap an existing one.

### Algorithm

Standard BPE (Sennrich et al., 2016):

1. Represent each word as a character sequence with an end-of-word marker: `low` → `('l', 'o', 'w', '</w>')`
2. Count frequency of every adjacent character pair across the corpus, weighted by word frequency
3. Merge the most frequent pair into a new token: `('l','o')` → `lo`
4. Repeat until vocabulary size is reached
5. Save the ordered merge table — this is the tokenizer

Encoding at inference replays the merge table greedily in training order. There is no model, no weights — encoding is a deterministic lookup algorithm.

**Merge criterion:** `max(pairs, key=lambda p: (pairs[p], p))` — primary sort by frequency, secondary sort alphabetically for deterministic tie-breaking across runs.



### File structure

```
banktoken/
├── tokenizer.py     # BPE core: get_vocab, get_stats, merge_vocab, train, encode, decode
├── train.py         # Data loading, cleaning, training entry point
├── benchmark.py     # Fertility rate comparison vs GPT-2 and BERT
├── demo.py          # Three-line usage example
├── banktoken.json   # Trained tokenizer (vocab + merge table)
└── requirements.txt
```

### Key design decisions

**`</w>` end-of-word marker** distinguishes word-final subwords from word-internal ones. The token `ing</w>` (word-final) and `ing` (internal, as in `rings`) can have different merge histories. GPT-2 uses a `Ġ` space prefix instead — different convention, same idea.

**Whitespace pre-tokenization** means BPE merges never cross word boundaries. A phrase like `basis points` cannot become a single token. This is a deliberate simplification — production tokenizers use regex-based pre-tokenization to handle contractions, punctuation attachment, and financial patterns like `$50bn` and `Q3'24`.

**Lowercase normalisation** reduces vocabulary fragmentation at the cost of losing case signal. `EBITDA` and `ebitda` map to the same tokens. A production finance tokenizer would preserve case for tickers and acronyms.

---

## Data

Trained on ~20,000 sentences sampled from:

| Dataset | Source | Size | Content |
|---------|--------|------|---------|
| [FinGPT/fingpt-sentiment-train](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) | HuggingFace | ~70k pairs | Financial news with sentiment labels |

Only the `input` field (financial sentences) was used. Single-word output labels (`positive`, `negative`, `neutral`) were excluded — including them artificially inflates character-pair frequencies for sentiment vocabulary at the expense of financial terminology.

All data is loaded via `datasets.load_dataset()` — no manual downloads required.

---

## Usage

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python train.py
# Trains on financial corpus, saves banktoken.json
```

### Encode / Decode

```python
from tokenizer import load, encode, decode

token2id, merges = load('banktoken.json')

ids = encode("EBITDA growth exceeded analyst estimates", token2id, merges)
print(ids)        # [423, 892, 1205, 3847, 2901]

text = decode(ids, token2id)
print(text)       # "ebitda growth exceeded analyst estimates"
```

### Benchmark

```bash
python benchmark.py
# Compares fertility rate against GPT-2 and BERT on held-out sentences
```

---

## Simplifications vs production tokenizers

This is a pedagogical implementation. Compared to GPT-2's tiktoken or HuggingFace's Rust-based tokenizers:

| Aspect | BankToken | Production |
|--------|-----------|------------|
| Base units | Unicode characters | Raw bytes (256 base tokens) |
| OOV handling | `<unk>` token | Impossible — all bytes in vocab |
| Pre-tokenization | Whitespace split | Regex (contractions, punctuation, numbers) |
| Case | Lowercased | Case-sensitive |
| Serialisation | JSON | Optimised binary (loads in ms) |
| Multilingual | English only | Any Unicode via byte fallback |

The most impactful missing feature is **byte-level base vocabulary**. Our tokenizer produces `<unk>` for any character not seen during training — the Bitcoin symbol `₿`, any emoji, non-Latin scripts. GPT-2's byte-level BPE makes `<unk>` impossible: every string of any language decomposes into bytes 0–255, all of which are always in vocabulary. This is why all modern autoregressive LLMs (GPT-4, LLaMA, Mistral) use byte-level BPE.

---

## References

- Sennrich, R., Haddow, B., & Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/). ACL 2016. — *Original BPE paper adapted for NLP*
- Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). — *GPT-2: introduced byte-level BPE*
- Devlin, J. et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers](https://aclanthology.org/N19-1423/). NAACL 2019. — *WordPiece tokenization*
- Kudo, T. (2018). [Subword Regularization](https://aclanthology.org/P18-1007/). ACL 2018. — *Unigram LM tokenization*
- Rust, P. et al. (2021). [How Good is Your Tokenizer?](https://aclanthology.org/2021.acl-long.300/) ACL 2021. — *Token fertility as an evaluation metric*
- Bostrom, K. & Durrett, G. (2020). [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://aclanthology.org/2020.findings-emnlp.414/). Findings of EMNLP 2020.

---

## What I learned

The most instructive part of this project was debugging a silent failure. The training pipeline produced plausible-looking output — tokens, merges, a vocabulary file — but the benchmark showed fertility 3x worse than GPT-2. Adding a step-by-step merge trace revealed two bugs:

1. A malformed regex `r"^a-z0-9"` (missing square brackets) that silently failed to strip punctuation — characters outside the training vocabulary were passed to the encoder, breaking merge matching
2. A comparison operator checking `word_tokens[i] == pair[1]` instead of `word_tokens[i+1] == pair[1]` — the merge condition only fired when both characters in a pair were identical (`tt`, `ee`, `oo`), silently skipping every other merge

Both bugs produced output that looked reasonable on the surface. Only the diagnostic trace made them visible. This is the most common failure mode in ML pipelines: the code runs, the numbers come out, but the numbers are wrong.
