from transformers import GPT2Tokenizer, BertTokenizer
from tokenizer import load, encode

TEST_SENTENCES = [
    "The company reported strong EBITDA growth despite macroeconomic headwinds.",
    "Q3 earnings per share exceeded analyst consensus estimates by 12 basis points.",
    "The Federal Reserve signaled a hawkish stance amid persistent inflationary pressure.",
    "Collateralised debt obligations were at the centre of the 2008 financial crisis.",
    "The portfolio rebalancing strategy involves overweighting equities and underweighting bonds.",
    "Amortisation of intangible assets reduced the reported net income significantly.",
    "The yield curve inversion historically precedes recessionary economic conditions.",
    "Quantitative easing measures were implemented to stimulate liquidity in credit markets.",
    "The derivatives desk hedged its exposure using interest rate swap contracts.",
    "Regulatory capital requirements under Basel III mandate higher tier-one capital ratios.",
]


def fertility_rate(tokens: list, text: str) -> float:
    """
    Fertility = tokens produced / words in original text.
    We count words by whitespace splitting — consistent across all tokenizers.
    """
    word_count = len(text.split())
    return len(tokens) / word_count


def benchmark_sentence(sentence: str,
                        gpt2_tok,
                        bert_tok,
                        bank_token2id: dict,
                        bank_merges: list) -> dict:
    """
    Run all three tokenizers on one sentence.
    Returns a dict of results for that sentence.
    """
    # GPT-2 tokenization
    gpt2_tokens = gpt2_tok.encode(sentence)

    # BERT tokenization
    bert_tokens = bert_tok.encode(sentence, add_special_tokens=False)

    # BankToken tokenization
    bank_tokens = encode(sentence, bank_token2id, bank_merges)

    return {
        'sentence'      : sentence,
        'words'         : len(sentence.split()),
        'gpt2_tokens'   : len(gpt2_tokens),
        'bert_tokens'   : len(bert_tokens),
        'bank_tokens'   : len(bank_tokens),
        'gpt2_fertility': fertility_rate(gpt2_tokens, sentence),
        'bert_fertility': fertility_rate(bert_tokens, sentence),
        'bank_fertility': fertility_rate(bank_tokens, sentence),
    }


def show_tokenization_breakdown(token2id: dict, merges: list,
                                 gpt2_tok, bert_tok):
    """
    For a set of finance-specific terms, show exactly how each
    tokenizer splits them. This makes the fertility numbers concrete.
    """
    words = [
        "EBITDA",
        "collateralised",
        "amortisation",
        "quantitative",
        "recessionary",
        "overcollateralisation",   # extreme case — very rare word
        "cryptocurrency",
        "macroeconomic",
    ]

    print("\n" + "="*65)
    print(f"{'Word':<25} {'GPT-2':<18} {'BERT':<18} {'BankToken'}")
    print("="*65)

    for word in words:
        # GPT-2: convert ids back to string tokens
        gpt2_ids    = gpt2_tok.encode(word)
        gpt2_pieces = gpt2_tok.convert_ids_to_tokens(gpt2_ids)

        # BERT: convert ids back to string tokens  
        bert_ids    = bert_tok.encode(word, add_special_tokens=False)
        bert_pieces = bert_tok.convert_ids_to_tokens(bert_ids)

        # BankToken: map ids back to tokens
        id2token    = {v: k for k, v in token2id.items()}
        bank_ids    = encode(word, token2id, merges)
        bank_pieces = [id2token.get(i, '<unk>') for i in bank_ids]

        # format as "piece|piece|piece"
        gpt2_str = '|'.join(gpt2_pieces)
        bert_str = '|'.join(bert_pieces)
        bank_str = '|'.join(bank_pieces)

        print(f"{word:<25} {gpt2_str:<18} {bert_str:<18} {bank_str}")


def main():
    print("Loading tokenizers...")

    ## load GPT-2 and BERT from HuggingFace
    gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2')
    bert_tok  = BertTokenizer.from_pretrained('bert-base-uncased')

    ## load our trained BankToken
    try:
        token2id, merges = load('banktoken.json')
        print(f"BankToken loaded — vocab size: {len(token2id)}")
    except FileNotFoundError:
        print("ERROR: banktoken.json not found. Run train.py first.")
        return

    ## per-sentence benchmark
    print("\nRunning benchmark on held-out financial sentences...")
    results = []
    for sentence in TEST_SENTENCES:
        r = benchmark_sentence(sentence, gpt2_tok, bert_tok, token2id, merges)
        results.append(r)

    ## aggregate stats
    avg_gpt2 = sum(r['gpt2_fertility'] for r in results) / len(results)
    avg_bert = sum(r['bert_fertility'] for r in results) / len(results)
    avg_bank = sum(r['bank_fertility'] for r in results) / len(results)

    gpt2_improvement = ((avg_gpt2 - avg_bank) / avg_gpt2) * 100
    bert_improvement = ((avg_bert - avg_bank) / avg_bert) * 100

    ## print results table
    print("\n" + "="*75)
    print(f"{'Sentence':<48} {'GPT2':>6} {'BERT':>6} {'Bank':>6}")
    print("="*75)
    for r in results:
        ## truncate sentence for display
        s = r['sentence'][:46] + '..' if len(r['sentence']) > 46 else r['sentence']
        print(f"{s:<48} {r['gpt2_fertility']:>6.2f} "
              f"{r['bert_fertility']:>6.2f} {r['bank_fertility']:>6.2f}")

    print("="*75)
    print(f"{'AVERAGE FERTILITY RATE':<48} {avg_gpt2:>6.2f} "
          f"{avg_bert:>6.2f} {avg_bank:>6.2f}")
    print(f"\nBankToken vs GPT-2 : {gpt2_improvement:+.1f}% fewer tokens")
    print(f"BankToken vs BERT  : {bert_improvement:+.1f}% fewer tokens")

    # --- word-level breakdown ---
    show_tokenization_breakdown(token2id, merges, gpt2_tok, bert_tok)

    # --- save results for README ---
    print("\n\nREADME snippet (copy this into your README.md):")
    print("-"*50)
    print(f"| Tokenizer | Avg Fertility | vs BankToken |")
    print(f"|-----------|--------------|--------------|")
    print(f"| GPT-2     | {avg_gpt2:.3f}         | {gpt2_improvement:+.1f}%        |")
    print(f"| BERT      | {avg_bert:.3f}         | {bert_improvement:+.1f}%        |")
    print(f"| BankToken | {avg_bank:.3f}         | baseline     |")


if __name__ == '__main__':
    main()