import torch
import numpy as np


def trim_prediction(pred, end_symbol):
    terminated = (pred == end_symbol).nonzero()
    if terminated.numel() > 0:
        pred = pred[:int(terminated[0])]
    return pred


def realize_sentence(texts, dictionary, copy_history, pointers, input_sentences,
                     end_symbol):
    '''Takes model predictions and converts them into a list of list of tokens

    Returns:
        A list of list of tokens, one per input example

    Args:
        texts: numpy array of model predictions, starting with <GO> token
        dictionary: dictionary mapping ids->tokens
        copy_history: whether each word is copied or generated
        pointers: where the copy is from in the input
        input_sentence: the input tokens
        end_symbol: the end symbol generated'''

    out_texts = []

    for text_row, copy_row, pointer_row, sentence in zip(
            texts, copy_history, pointers, input_sentences):
        out_text_row = []

        def helper(copy, token, pointer):
            if copy == 1:
                return sentence[int(pointer)]
            else:
                return dictionary[int(token)]
        for text, copy, pointer in zip(text_row, copy_row, pointer_row):
            trimmed_text = trim_prediction(text, end_symbol)
            out_text_row.append(list(map(
                helper,
                copy.flatten(), trimmed_text.flatten(),
                pointer.flatten())))
        out_texts.append(out_text_row)

    return out_texts


def generate_sentence(amr, dictionary, model):
    '''Takes an input AMR (as a numpy array) and generates a text realization.

    Returns a list of list of strings'''

    outs = model.predict(torch.LongTensor(amr).cuda())

    return realize_sentence(outs, dictionary, model.text_model.end_symbol)


def paraphrase_sentence(sentence, to_id, dictionary, model):
    '''Takes an input sentence (as a tokenized list) and paraphrases it.'''

    def helper(token):
        if token in to_id:
            return to_id[token]
        else:
            return to_id['<UNK>']
    sent_ids = np.array(list(map(helper, sentence)))

    outs = model.text_paraphrase(torch.LongTensor([sent_ids]).cuda(),
                                 150, 200, 100)
    rv = realize_sentence(outs, dictionary, model.text_model.end_symbol)

    return rv
