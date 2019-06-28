from collections import OrderedDict


day_entity_re = 'day.*_date-entity_(\d+)'
month_entity_re = 'month.*_date-entity_(\d+)'
year_entity_re = 'year.*_date-entity_(\d+)'

complex_num_re = '.+_num_(\d+)'


class Alignment(object):
    def __init__(self, anon_tokens):
        # anon_tokens a list of the anonymized tokens in the sentence
        temp_alignments = {}
        for token in anon_tokens:
            token, _, text, _, _ = token.split('|||')
            if token not in temp_alignments:
                temp_alignments[token] = text
        # Make sure we replace things like temporal-quantity_num_0 before num_0
        self.alignments = OrderedDict(
            sorted(
                temp_alignments.items(), key=lambda x: (len(x[0]), x[1]),
                reverse=True)
        )

    def deanonymize_sentence(self, sentence):
        # Tokenize sentence
        sentence = sentence.split()
        new_sent = []
        for token in sentence:
            if token[:6] == 'month_':
                token = 'month_date-entity_' + token.split('_')[-1]
            elif token[:4] == 'day_':
                token = 'day_date-entity_' + token.split('_')[-1]
            elif token[:5] == 'year_':
                token = 'year_date-entity_' + token.split('_')[-1]
            for possible_match, replacement in self.alignments.items():
                token = token.replace(possible_match, replacement).lower()
            new_sent.append(token)

        return ' '.join(new_sent)


def deanonymize_sentence_raw(raw_sent, raw_alignment):
    if raw_alignment is not '':
        alignment = Alignment((x.strip() for x in raw_alignment.split('#')))
        new_sent = alignment.deanonymize_sentence(raw_sent)
        return new_sent
    else:
        return raw_sent


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_filename')
    parser.add_argument('--alignments')

    args = parser.parse_args()

    with open(args.input_filename, 'r') as f, open(args.alignments, 'r') as g:
        for sent, alignment in zip(f, g):
            sent, alignment = sent.strip(), alignment.strip()
            print(deanonymize_sentence_raw(sent, alignment))
