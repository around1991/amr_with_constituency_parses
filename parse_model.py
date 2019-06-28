import numpy as np


class ParseTree(object):
    def __init__(self, string):
        # Decompose the string into a list of constituent-span tuples.
        idx = 0
        stack = []
        self.constituents = []
        self.string = string
        self.terminals = []
        for token in string.split():
            if '(' in token:
                stack.append((token[1:], idx))
            elif ')' in token:
                if stack:
                    const = stack.pop()
                    if const[0] == token[1:]:
                        label = const[0]
                    else:
                        label = 'NULL'
                    self.constituents.append((label, (const[1], idx)))
            else:
                self.terminals.append(token)
                idx += 1

    def project_spans_through_alignment(self, alignment):
        new_consts = []
        for label, span in self.constituents:
            # Project the span through the alignment: the left is the least
            # index, the right is the highest matching index

            # x.index gives the least index matching value, and we can do
            # len(x) - 1 - x[::-1].index to get the highest
            new_span = (
                alignment.index(span[0]),
                len(alignment) - 1 - alignment[::-1].index(span[1])
            )
            new_consts.append((label, new_span))

        return new_consts


def overlap(this, other):
    # For each constituent in self, try to match it with a constituent in
    # other
    labelled_matches = 0
    unlabelled_matches = 0
    this_align, other_align = calculate_insert_delete_alignment(this, other)
    for const in this.project_spans_through_alignment(this_align):
        other_consts = other.project_spans_through_alignment(other_align)
        other_spans = [x[1] for x in other_consts]
        if const in other_consts:
            labelled_matches += 1
        if const[1] in other_spans:
            unlabelled_matches += 1

    return labelled_matches, unlabelled_matches


def score(a, b):
    terminal_classes = {
        'NN': 1,
        'NNS': 1,
        'NNP': 1,
        'NNPS': 1,
        'JJ': 2,
        'JJR': 2,
        'JJS': 2,
        'MD': 3,
        'VB': 3,
        'VBD': 3,
        'VBG': 3,
        'VBN': 3,
        'VBP': 3,
        'VBZ': 3,
        'RB': 4,
        'RBR': 4,
        'RBS': 4,
        'RP': 5,
        'TO': 5,
        'IN': 5
    }
    if a == b:
        return 2
    elif (a in terminal_classes and b in terminal_classes and
          terminal_classes[a] == terminal_classes[b]):
        return 1
    else:
        return 0


def calculate_insert_delete_alignment(prediction, gold):
    # Given a prediction and a gold, aligns the terminals of the prediction and
    # the gold by insert, delete and match operations
    gap_score = -1

    alignment_scores = np.zeros(
        (len(prediction.terminals) + 1, len(gold.terminals) + 1))

    for i in range(1, len(prediction.terminals) + 1):
        alignment_scores[i, 0] = i * gap_score

    for i in range(1, len(gold.terminals) + 1):
        alignment_scores[0, i] = i * gap_score

    for i in range(1, len(prediction.terminals) + 1):
        for j in range(1, len(gold.terminals) + 1):
            alignment_scores[i, j] = max(
                alignment_scores[i - 1, j - 1] + score(
                    prediction.terminals[i - 1], gold.terminals[j - 1]),
                alignment_scores[i - 1, j] + gap_score,
                alignment_scores[i, j - 1] + gap_score)

    # Now calculate the alignments
    alignment = []
    i = len(prediction.terminals)
    j = len(gold.terminals)

    while i > 0 and j > 0:
        alignment.append((i, j))
        if (alignment_scores[i - 1, j - 1] +
                score(prediction.terminals[i - 1], gold.terminals[j - 1]) ==
                alignment_scores[i, j]):
            i -= 1
            j -= 1
        elif (alignment_scores[i - 1, j] + gap_score ==
                alignment_scores[i, j]):
            i -= 1
        else:
            j -= 1

    while i > 0:
        alignment.append((i, j))
        i -= 1

    while j > 0:
        alignment.append((i, j))
        j -= 1

    alignment.append((0, 0))

    pred_alignments, gold_alignments = zip(*reversed(alignment))

    return pred_alignments, gold_alignments


def calculate_parse_f1(prediction, gold):
    labelled_matches, unlabelled_matches = overlap(prediction, gold)

    if prediction.constituents:
        labelled_precision = labelled_matches / len(prediction.constituents)
        unlabelled_precision = unlabelled_matches / len(prediction.constituents)
    else:
        labelled_precision = 0.0
        unlabelled_precision = 0.0

    labelled_recall = labelled_matches / len(gold.constituents)
    unlabelled_recall = unlabelled_matches / len(gold.constituents)

    if labelled_recall == 0.0 and labelled_precision == 0.0:
        labelled_f1 = 0.0
    else:
        labelled_f1 = 2 * (labelled_precision * labelled_recall) / (
            labelled_precision + labelled_recall)

    if unlabelled_recall == 0.0 and unlabelled_precision == 0.0:
        unlabelled_f1 = 0.0
    else:
        unlabelled_f1 = 2 * (unlabelled_precision * unlabelled_recall) / (
            unlabelled_precision + unlabelled_recall)

    return labelled_f1, unlabelled_f1


class ParseActionMasker(object):
    def __init__(self, toid):
        self.toid = toid
        self.totoken = [0 for _ in range(len(toid))]
        for key, val in self.toid.items():
            self.totoken[val] = key

        self.terminal_actions = {
            val for key, val in toid.items()
            if key[0] not in {'(', ')'} and key != '<EOS>'}
        self.open_actions = {
            val for key, val in toid.items() if '(' in key}
        self.end_action = {toid['<EOS>']}

        self.stack = []
        self.action_history = []
        self.added_to_stack = False

    def return_mask(self, action):
        # Updates the internal stack state and returns a mask of the permitted
        # structural actions at the current timestep

        # The only permitted structural action is closing the top item of the
        # stack

        # timestep is a batch_size x 1 sized tensor

        self.action_history.append(action)
        token = self.totoken[int(action)]
        if '(' in token:
            self.stack.append(token[1:])
            self.added_to_stack = True
        elif ')' in token:
            top = self.stack.pop()
            assert top == token[1:]
        # Work out what the permissible actions are
        if self.stack:
            # We can only close the top stack item, but can open anything
            permissible_actions = (
                {self.toid[')' + self.stack[-1]]} | self.open_actions |
                self.terminal_actions)
        else:
            # We can only open constituents
            permissible_actions = self.open_actions

        if self.added_to_stack:
            permissible_actions = permissible_actions | self.end_action

        permissible_actions = list(permissible_actions)

        mask = np.zeros(len(self.toid))
        mask[permissible_actions] = 1

        return np.array(mask)

    def return_series_of_masks(self, actions):
        # actions is a seq_len array
        assert not self.stack
        masks = np.array([self.return_mask(action) for action in actions])

        return masks
