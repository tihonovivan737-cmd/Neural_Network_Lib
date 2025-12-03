import regex as re
import json
from collections import defaultdict


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def bpe(token, merge_rules):
    bpe_ranks = {pair: idx for idx, pair in enumerate(merge_rules)}
    word = list(token)
    pairs = get_pairs(word)

    while pairs:
        bigram = min(pairs, key=lambda p: bpe_ranks.get(p, float('inf')))
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break
            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
        if len(word) == 1:
            break
        pairs = get_pairs(word)
    return word


def initialize_vocab(texts, lowercase=True):
    vocab = defaultdict(int)
    byte_encoder = bytes_to_unicode()
    pat = re.compile(
        r"""(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
    for text in texts:
        if lowercase:
            text = text.lower()
        for match in pat.findall(text):
            if match:
                byte_match = ''.join(byte_encoder[b] for b in match.encode('utf-8'))
                formatted = ' '.join(list(byte_match)) + ' '
                vocab[formatted] += 1
    return vocab


def merge_vocab(pair, vocab):
    pattern = re.compile(rf"(?<!\S){re.escape(' '.join(pair))}(?!\S)")
    merged_token = "".join(pair)
    return {pattern.sub(merged_token, word): freq for word, freq in vocab.items()}


def get_bigram_counts(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        tokens = word.strip().split()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += freq
    return pairs


def save_tokenizer(merge_rules, file_path):
    merges = [{"pair": list(pair), "merged": "".join(pair)} for pair in merge_rules]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(merges, f, ensure_ascii=False, indent=4)


def train_bpe(input_data, lowercase=True, vocab_size=1000, file_path=None, save_interval=100):

    if isinstance(input_data, str):
        with open(input_data, 'r', encoding='utf-8') as f:
            texts = [f.read()]
    elif isinstance(input_data, list):
        texts = input_data
    else:
        raise ValueError("input_data должен быть строкой или списком строк.")

    vocab = initialize_vocab(texts, lowercase=lowercase)
    current_vocab = vocab.copy()
    merge_rules = []

    print("Начинаем обучение токенизатора...")

    for step in range(vocab_size):
        pairs = get_bigram_counts(current_vocab)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        current_vocab = merge_vocab(best_pair, current_vocab)
        merge_rules.append(best_pair)

        if (step + 1) % 10 == 0:
            tokens = {token for word in current_vocab for token in word.split()}
            print(f"Прогресс: {step + 1}/{vocab_size}, токенов в словаре: {len(tokens)}")

        if file_path and (step + 1) % save_interval == 0:
            save_tokenizer(merge_rules, file_path)

    if file_path:
        save_tokenizer(merge_rules, file_path)

    tokens = {token for word in current_vocab for token in word.split()}
    print("Обучение завершено.")

    return current_vocab, merge_rules, tokens


def load_tokenizer(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        merges = json.load(f)
    merge_rules = [tuple(merge["pair"]) for merge in merges]

    byte_encoder = bytes_to_unicode()
    encoder = {byte_encoder[b]: b for b in range(256)}
    for idx, pair in enumerate(merge_rules, start=256):
        merged = ''.join(pair)
        encoder[merged] = idx
    decoder = {v: k for k, v in encoder.items()}

    return merge_rules, encoder, decoder


def encode(text, merge_rules, encoder, lowercase=True):
    byte_encoder = bytes_to_unicode()
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    if lowercase:
        text = text.lower()
    ids = []
    for match in pat.findall(text):
        if not match:
            continue
        byte_match = ''.join(byte_encoder[b] for b in match.encode('utf-8'))
        bpe_tokens = bpe(byte_match, merge_rules)
        ids.extend(encoder[tok] for tok in bpe_tokens)
    return ids


def decode(ids, decoder):
    unicode_to_byte = {v: k for k, v in bytes_to_unicode().items()}
    text = ''.join(decoder[id_] for id_ in ids)
    byte_list = [unicode_to_byte[char] for char in text]
    return bytes(byte_list).decode('utf-8', errors='replace')
