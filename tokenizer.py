import re
import re
import json
from collections import Counter, defaultdict


def get_vocab(corpus: list[str]) -> dict[tuple, int]:
    vocab = Counter()

    for sentence in corpus:
        words = sentence.lower().split()

        for word in words:
            word = re.sub(r"[^a-z0-9]", "", word)
            if word:
                char_tuple = tuple(word) + ("</w>",)
                vocab[char_tuple] += 1

    return dict(vocab)


def get_stats(vocab: dict[tuple, int]) -> dict[tuple, int]:
    """
    Count frequency of every adjacent pair across all words
    """
    pairs = Counter()
    for word_tuple, freq in vocab.items():
        for pair in zip(word_tuple, word_tuple[1:]):
            pairs[pair] += freq

    return pairs


def merge_vocab(pair: tuple, vocab: dict[tuple, int]) -> dict[tuple, int]:
    """
    Merge all occurances of 'pair' in vocabulary
    """
    new_vocab = {}
    merged = "".join(pair)
    for word_tuple, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if (
                i < len(word_tuple) - 1
                and word_tuple[i] == pair[0]
                and word_tuple[i + 1] == pair[1]
            ):
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


def train(corpus: list[str], vocab_size: int) -> tuple[dict, list]:
    """
    Train BPE on corpus.

    Returns:
    token2id: dict mapping token string -> integer id
    merges: ordered list of merge rules
    """

    vocab = get_vocab(corpus)

    base_tokens = set()
    for word_tuple in vocab:
        for char in word_tuple:
            base_tokens.add(char)

    ## Giving special token first ids
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    all_tokens = special_tokens + sorted(base_tokens)
    merges = []

    if vocab_size <= len(all_tokens):
        raise ValueError(
            f"vocab_size ({vocab_size}) must be greater than the base "
            f"vocabulary size ({len(all_tokens)}). "
            f"Try a value above {len(all_tokens)}."
        )

    ## merge until we hit the vocabulary size
    num_merges = vocab_size - len(all_tokens)

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            print(f"No more pairs to merge at step{i}. Stopping...")
            break

        best_pair = max(pairs, key=lambda p: (pairs[p], p))

        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        all_tokens.append("".join(best_pair))

        if (i + 1) % 100 == 0:
            print(f"  Merge {i+1}/{num_merges}: {best_pair} → {''.join(best_pair)}")

    token2id = {token: idx for idx, token in enumerate(all_tokens)}
    print(f"Training complete. Final vocab size: {len(token2id)}")
    return token2id, merges


def encode(text: str, token2id: dict, merges: list) -> list[int]:
    """
    Encode a string into a list of token ids
    """
    unk_id = token2id.get("<unk>", 1)

    words = text.lower().split()
    all_ids = []

    for word in words:
        word = re.sub(r"[^a-z0-9]", "", word)
        if not word:
            continue

        word_tokens = list(word) + ["</w>"]

        ## apply each merge rule in order
        for pair in merges:
            merged = "".join(pair)
            i = 0
            new_tokens = []
            while i < len(word_tokens):
                if (
                    i < len(word_tokens) - 1
                    and word_tokens[i] == pair[0]
                    and word_tokens[i+1] == pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_tokens

        ## convert tokens to ids
        all_ids.extend([token2id.get(t, unk_id) for t in word_tokens])

    return all_ids


def decode(ids: list[int], token2id: dict) -> str:
    """
    Decode a list of token ids back to str
    """
    id2token = {idx: token for token, idx in token2id.items()}
    tokens = [id2token.get(i, "<unk>") for i in ids]

    text = " ".join(tokens)
    text = text.replace("</w>", " ").replace("</w>", "")
    return text.strip()


def save(token2id: dict, merges: list, path: str):
    data = {"token2id": token2id, "merges": [list(pair) for pair in merges]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved tokenizer to {path}")


def load(path: str) -> tuple[dict, list]:
    with open(path) as f:
        data = json.load(f)
    token2id = data["token2id"]
    merges = [tuple(pair) for pair in data["merges"]]
    return token2id, merges


# sentences = [
#     "hey there! how's it going?",
#     "It is going good!, I am planning to apply for a credit card",
#     "Do you have anything in mind?",
# ]
# token2id, merges = train(sentences, 10)
# print(f"merges: \n {merges}")
# print(f"token2id: \n {token2id}")
