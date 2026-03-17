from tokenizer import load, encode, decode

token2id, merges = load('banktoken.json')

sentences = [
    "EBITDA growth exceeded analyst estimates by 12 basis points",
    "The Federal Reserve signaled quantitative tightening",
    "Collateralised debt obligations drove the 2008 financial crisis",
]

for sentence in sentences:
    ids = encode(sentence, token2id, merges)
    print(f"Input : {sentence}")
    print(f"Tokens: {ids}")
    print(f"Decoded: {decode(ids, token2id)}")
    print()