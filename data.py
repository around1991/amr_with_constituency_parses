import os
import pickle
import gzip
import multiprocessing

from collections import Counter, defaultdict

import numpy as np


def _add_extra_symbols(vocab_list):
    vocab_list.insert(0, '<PAD>')
    vocab_list.append('<GO>')
    vocab_list.append('<EOS>')
    vocab_list.append('<UNK>')


def make_pointers(amr, text, banlist={}):
    amr_locations = defaultdict(list)
    text_locations = defaultdict(list)

    text_pointers = []
    amr_pointers = []

    for i, token in enumerate(amr):
        if token not in banlist:
            amr_locations[token].append(i)

    for i, token in enumerate(text):
        if token not in banlist:
            text_locations[token].append(i)
        amr_pointer = [0 for _ in range(len(amr))]
        for index in amr_locations[token]:
            amr_pointer[index] = 1
        amr_pointers.append(amr_pointer)

    for token in amr:
        text_pointer = [0 for _ in range(len(text))]
        for index in text_locations[token]:
            text_pointer[index] = 1
        text_pointers.append(text_pointer)

    # amr_pointers are pointers into the amr, have size text_len x amr_len
    # vice_versa for text_pointers
    # Be a bit careful of what we return - we don't want to match the first <GO>
    # to anything
    return np.array(amr_pointers[1:]), np.array(text_pointers[1:])


class DataLoader(object):
    def __init__(self, data_dir, batch_size,
                 max_text_len=50, max_amr_len=80, max_parse_len=150,
                 vocabs=None, additional_counts=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.length = None
        self.max_text_len = max_text_len
        self.max_amr_len = max_amr_len
        self.max_parse_len = max_parse_len
        self.buf = []
        self.max_buf_len = 50000
        self.min_buf_len = self.max_buf_len / 2

        self.min_token_count = 1

        self.training = True

        if not vocabs:
            self.build_vocabs(additional_counts)
        else:
            self.extract_vocabs(vocabs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __len__(self):
        if not self.length:
            # Check for cached length
            if os.path.exists(self.data_dir + 'length'):
                with open(self.data_dir + 'length', 'r') as f:
                    self.length = int(f.read().strip())
            else:
                print("reading length")
                length = 0
                # Iterate over the files in data_dir and count how many words
                # are in each file (assume whitespace tokenized)
                for batch, _, _, _, _ in self:
                    if (length + len(batch) % 10000) < (length % 10000):
                        print("current length: {}".format(length + len(batch)))
                    length += len(batch)

                # Cache the length
                with open(self.data_dir + 'length', 'w') as f:
                    f.write(str(length))
                self.length = length
        return self.length

    def count_helper(self, filename):
        print("processing file: {}".format(filename))
        global_counter = Counter()
        with open(self.data_dir + filename, 'r') as f:
            for i, line in enumerate(f):
                if i % 10000 == 0:
                    print("reading line {}".format(i))
                global_counter += Counter(
                    line.strip().split())

        return global_counter

    def build_vocabs(self, additional_counts):
        if not os.path.exists(self.data_dir + 'cache.pkl'):
            self.text_counts = Counter()
            self.amr_counts = Counter()
            self.parse_counts = Counter()
            files = [filename for filename in os.listdir(self.data_dir)
                     if 'cache' not in filename or 'length' not in filename]
            pool = multiprocessing.Pool(10)
            all_counts = pool.map(self.count_helper, files)
            for filename, count in zip(files, all_counts):
                if 'amr' in filename:
                    self.amr_counts += count
                elif 'parse' in filename:
                    self.parse_counts += count
                elif 'txt' in filename:
                    self.text_counts += count
            with open(self.data_dir + 'cache.pkl', 'wb') as f:
                pickle.dump(self.text_counts, f)
                pickle.dump(self.amr_counts, f)
                pickle.dump(self.parse_counts, f)
        else:
            with open(self.data_dir + 'cache.pkl', 'rb') as f:
                self.text_counts = pickle.load(f)
                self.amr_counts = pickle.load(f)
                self.parse_counts = pickle.load(f)

        if additional_counts:
            with open(additional_counts, 'rb') as f:
                additional_counts = pickle.load(f)

        words = set([x for x, y in self.text_counts.items()
                     if y >= self.min_token_count])

        # Now turn these into IDs
        self.text_totoken = list(words)
        self.amr_totoken = [amr for amr in self.amr_counts]
        self.parse_totoken = [parse for parse in self.parse_counts]
        self.text_totoken.sort(key=lambda x: (self.text_counts[x], x))
        self.parse_totoken.sort(key=lambda x: (self.parse_counts[x], x))
        self.amr_totoken.sort(key=lambda x: (self.amr_counts[x], x))
        print("text vocab size: {}".format(len(self.text_totoken)))
        print("amr vocab size: {}".format(len(self.amr_totoken)))
        _add_extra_symbols(self.text_totoken)
        _add_extra_symbols(self.amr_totoken)
        _add_extra_symbols(self.parse_totoken)

        self.text_toid = {word: val
                          for val, word in enumerate(self.text_totoken)}
        self.amr_toid = {amr: val for val, amr in enumerate(self.amr_totoken)}
        self.parse_toid = {parse: val
                           for val, parse in enumerate(self.parse_totoken)}

    def extract_vocabs(self, vocabs):
        self.text_totoken = vocabs[0]
        self.amr_totoken = vocabs[1]
        self.parse_totoken = vocabs[2]
        self.text_toid = vocabs[3]
        self.amr_toid = vocabs[4]
        self.parse_toid = vocabs[5]
        self.text_counts = vocabs[6]
        self.amr_counts = vocabs[7]
        self.parse_counts = vocabs[8]

    def package_vocabs(self):
        return (self.text_totoken, self.amr_totoken, self.parse_totoken,
                self.text_toid, self.amr_toid, self.parse_toid,
                self.text_counts, self.amr_counts, self.parse_counts)

    @property
    def text_vocab_size(self):
        return len(self.text_totoken)

    @property
    def amr_vocab_size(self):
        return len(self.amr_totoken)

    @property
    def parse_vocab_size(self):
        return len(self.parse_totoken)

    def load_vectors(self, vector_file):
        with open(vector_file, 'rb') as f:
            vecs = pickle.load(f)
            totokens = pickle.load(f)

        toids = {totokens[i]: i for i in range(len(totokens))}

        indices = list(map(lambda x: toids[x] if x in toids else -1,
                           self.text_totoken))

        vectors = list(map(
            lambda x: vecs[x] if x != -1 else np.random.normal(
                scale=0.1, size=300),
            indices))

        del vecs
        del totokens

        return np.vstack(vectors)

    def to_ids(self, tokens, toid, UNK=0.0, counts=None):
        if UNK and self.training:
            def helper(x):
                if x in {'<GO>', '<EOS>', '(', ')'}:
                    return toid[x]
                elif x[0] in {':', '(', ')'}:
                    return toid[x]
                elif counts and counts[x] > 1:
                    return toid[x]
                elif np.random.binomial(1, UNK) == 1:
                    return toid[x]
                else:
                    return toid['<UNK>']
        else:
            def helper(x):
                if x in toid:
                    return toid[x]
                else:
                    return toid['<UNK>']
        # Handle UNK=True: replace words with count 1 with UNK 50% of the time
        ids = list(map(helper, tokens))

        return ids

    def process_buffer(self):
        assert self.buf

        batch = self.buf[:self.batch_size]
        self.buf = self.buf[self.batch_size:]
        text_batch, amr_batch, parse_batch = zip(*batch)
        max_text_len = min(max(len(line) for line in text_batch),
                           self.max_text_len)
        if all(amr_batch):
            max_amr_len = min(max(len(line) for line in amr_batch),
                              self.max_amr_len)
            amr_ids_batch = np.zeros((len(text_batch), max_amr_len),
                                     dtype=np.int64)
            amr_text_batch = []
        if all(parse_batch):
            max_parse_len = min(max(len(line) for line in parse_batch),
                                self.max_parse_len)
            parse_ids_batch = np.zeros((len(text_batch), max_parse_len),
                                       dtype=np.int64)

        text_ids_batch = np.zeros((len(text_batch), max_text_len),
                                  dtype=np.int64)
        text_text_batch = []

        amr_pointers_batch = np.zeros(
            (len(text_batch), max_text_len-1, max_amr_len), dtype=np.int64)
        text_pointers_batch = np.zeros(
            (len(text_batch), max_amr_len-1, max_text_len), dtype=np.int64)

        for i, (text, amr, parse) in enumerate(
                zip(text_batch, amr_batch, parse_batch)):
            # Convert the tokens into numerical ids
            if amr is not None:
                amr_ids = self.to_ids(amr, self.amr_toid, UNK=0.9)
                truncated_amr_ids = amr_ids[:max_amr_len]
                amr_ids_batch[i, :len(truncated_amr_ids)] = truncated_amr_ids
                amr_text_batch.append(amr)
            else:
                amr_ids_batch = None
                amr_text_batch = []

            if parse is not None:
                parse_ids = self.to_ids(parse, self.parse_toid, UNK=0.9)
                truncated_parse_ids = parse_ids[:max_parse_len]
                parse_ids_batch[i, :len(truncated_parse_ids)] = truncated_parse_ids
            else:
                parse_ids_batch = None

            text_ids = self.to_ids(text, self.text_toid, UNK=0.5,
                                   counts=self.text_counts)
            truncated_text_ids = text_ids[:max_text_len]
            text_ids_batch[i, :len(truncated_text_ids)] = truncated_text_ids
            text_text_batch.append(text)
            amr_pointers, text_pointers = make_pointers(
                amr, text, banlist={'(', ')', '<EOS>'})

            amr_pointers_batch[
                i, :len(truncated_text_ids)-1, :len(truncated_amr_ids)] = \
                amr_pointers[:len(truncated_text_ids)-1, :len(truncated_amr_ids)]

            text_pointers_batch[
                i, :len(truncated_amr_ids)-1, :len(truncated_text_ids)] = \
                text_pointers[:len(truncated_amr_ids)-1, :len(truncated_text_ids)]

        return (amr_ids_batch, parse_ids_batch, text_ids_batch,
                amr_text_batch, text_text_batch,
                amr_pointers_batch, text_pointers_batch)

    def __iter__(self):
        self.buf = []
        files = set(x.split('.')[0] for x in os.listdir(self.data_dir))
        if 'length' in files:
            files.remove('length')
        if 'cache' in files:
            files.remove('cache')
        files = sorted(list(files))
        for filename in files:
            if os.path.exists(self.data_dir + filename + '.parse'):
                with open(self.data_dir + filename + '.txt.anonymized') as text, \
                        open(self.data_dir + filename + '.amr.anonymized') as amr, \
                        open(self.data_dir + filename + '.parse') as parse:
                    for text_line, amr_line, parse_line in zip(text, amr, parse):
                        # Add the lines to the respective batches
                        self.buf.append(
                            (['<GO>'] + text_line.split() + ['<EOS>'],
                             ['<GO>'] + amr_line.split() + ['<EOS>'],
                             ['<GO>'] + parse_line.split() + ['<EOS>']))
                        if len(self.buf) > self.max_buf_len:
                            while len(self.buf) > self.min_buf_len:
                                batches = self.process_buffer()

                                yield batches
            else:
                assert os.path.exists(self.data_dir + filename + '.txt.anonymized.gz')
                with gzip.open(self.data_dir + filename + '.txt.anonymized.gz') as text, \
                        gzip.open(self.data_dir + filename + '.parse.gz') as parse:
                    for text_line, parse_line in zip(text, parse):
                        self.buf.append(
                            (['<GO>'] + text_line.decode('utf-8').split() + ['<EOS>'],
                             None,
                             ['<GO>'] + parse_line.decode('utf-8').split() + ['<EOS>']))
                        if len(self.buf) > self.max_buf_len:
                            while len(self.buf) > self.min_buf_len:
                                batches = self.process_buffer()

                                yield batches

        # Might have some unconsumed batches. Yield them
        while self.buf:
            batches = self.process_buffer()

            yield batches


if __name__ == '__main__':
    with open('/home/kc391/synsem/data/big_dev/cache.pkl', 'rb') as f:
        additional_counts = pickle.load(f)
    test_iter = DataLoader('/home/kc391/synsem/data/ldc_train/', 50,
                           additional_counts=additional_counts)
    for item in test_iter:
        print(item)
