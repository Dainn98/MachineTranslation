
from collections import Counter
from config import VOCAB_SIZE, MIN_FREQ, MAX_SEQ_LEN

class SimpleTokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE, min_freq=MIN_FREQ, lower=True):
        self.lower = lower
        self.min_freq = min_freq
        self.vocab_size = vocab_size

        self.PAD = "<pad>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"
        self.UNK = "<unk>"

        self.word2id = {}
        self.id2word = {}

    def norm(self, text):
        return text.lower().strip().split()

    def fit(self, texts):
        freq = Counter()
        for t in texts:
            freq.update(self.norm(t))

        vocab_words = [w for w, f in freq.items() if f >= self.min_freq]
        vocab_words = vocab_words[: self.vocab_size]

        vocab = [self.PAD, self.BOS, self.EOS, self.UNK] + vocab_words
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text, max_len=MAX_SEQ_LEN):
        ids = [self.word2id.get(w, self.word2id[self.UNK]) for w in self.norm(text)]
        ids = ids[:max_len]
        return [self.word2id[self.BOS]] + ids + [self.word2id[self.EOS]]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.id2word.get(int(i), self.UNK)
            if w not in [self.PAD, self.BOS, self.EOS]:
                words.append(w)
        return " ".join(words)

    def vocab_size_(self):
        return len(self.word2id)

    def pad_id(self):
        return self.word2id[self.PAD]
