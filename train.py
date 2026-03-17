from datasets import load_dataset
from tokenizer import train, save
import re
import random

def load_financial_corpus() -> list[str]:
    corpus = []

    # Dataset: FinGPT sentiment — formal financial sentences
    print("Loading FinGPT sentiment...")
    ds3 = load_dataset("FinGPT/fingpt-sentiment-train")
    for row in ds3['train']:
        corpus.append(row['input'])
    print(f"  → {len(corpus)} sentences so far")

    return corpus


def clean_corpus(corpus: list[str]) -> list[str]:
    """
    Light cleaning — remove noise that would waste vocabulary slots.
    We keep financial numbers like "q3", "2024" because they are
    meaningful in this domain.
    """
    cleaned = []
    for sentence in corpus:
        # remove URLs
        sentence = re.sub(r"http\S+|www\S+", "", sentence)
        # remove @mentions and #hashtags (twitter noise)
        sentence = re.sub(r"[@#]\S+", "", sentence)
        # remove multiple spaces
        sentence = re.sub(r"\s+", " ", sentence).strip()
        # skip very short sentences
        if len(sentence.split()) >= 4:
            cleaned.append(sentence)

    print(f"Cleaned corpus: {len(cleaned)} sentences remaining")
    return cleaned


def main():
    VOCAB_SIZE = 16000
    SAVE_PATH  = "banktoken.json"
    MAX_SENTENCES = 20000
    # load and clean
    corpus  = load_financial_corpus()
    corpus  = clean_corpus(corpus)

    # sample if corpus is larger than needed
    if len(corpus) > MAX_SENTENCES:
        random.seed(42)        # seed for reproducibility
        corpus = random.sample(corpus, MAX_SENTENCES)
        print(f"Sampled {MAX_SENTENCES} sentences for training")

    # quick stats before training
    total_words = sum(len(s.split()) for s in corpus)
    total_chars = sum(len(s) for s in corpus)
    print(f"\nCorpus stats:")
    print(f"  Sentences : {len(corpus):,}")
    print(f"  Words     : {total_words:,}")
    print(f"  Characters: {total_chars:,}")

    # train
    print(f"\nTraining BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    token2id, merges = train(corpus, vocab_size=VOCAB_SIZE)

    # save
    save(token2id, merges, SAVE_PATH)
    print(f"\nDone. Tokenizer saved to '{SAVE_PATH}'")


if __name__ == '__main__':
    main()

