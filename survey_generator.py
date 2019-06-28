import torch

from nltk.tokenize.treebank import TreebankWordDetokenizer
from alignments import deanonymize_sentence_raw
from model import AMR2Parse2Text, AMR2Text
from data import DataLoader
from generate import realize_sentence

MC_QUESTION = '''\
[[Question:MC:MultipleAnswer]]
Reference: {ref}

[[Choices]]
{choice0}
{choice1}
{choice2}
'''

QUESTION_BLOCK = '''\
[[Block]]

{question0}

{question1}

{question2}

{question3}

{question4}

{question5}

{question6}

{question7}

{question8}

{question9}

[[PageBreak]]
'''


class Question(object):
    def __init__(self, ref, choices):
        self.ref = ref
        self.choices = choices

    def __str__(self):
        ref_tuple = (('ref', self.ref),)
        choices_tuple = (
            ('choice{}'.format(i), choice)
            for i, choice in enumerate(self.choices)
        )

        format_dict = dict(ref_tuple + tuple(choices_tuple))

        return MC_QUESTION.format(**format_dict)


def create_question_block(list_of_questions):
    questions_dict = dict(
        ('question{}'.format(i), str(question))
        for i, question in enumerate(list_of_questions)
    )

    return QUESTION_BLOCK.format(**questions_dict)

if __name__ == '__main__':
    data_iterator = DataLoader(
        '/home/kc391/synsem/data/ldc_train/', 2)
    dev_iterator = DataLoader(
        '/home/kc391/synsem/data/ldc2015e86_dev/', 1,
        vocabs=data_iterator.package_vocabs())
    data_iterator.eval()
    dev_iterator.eval()

    parse_model = AMR2Parse2Text(500, 300, data_iterator.amr_vocab_size,
                                 data_iterator.parse_vocab_size,
                                 data_iterator.text_vocab_size,
                                 encoder_dropout=0.5,
                                 input_dropout=0.5, output_dropout=0.5,
                                 recurrent_dropout=0.3, split_gpu=False,
                                 share_parameters=True)
    parse_model.load_state_dict(
        torch.load('../models/forward_model_ldc2017t10_'
                   'refactored_enc_rec_dropout'))
    parse_model.cuda()
    parse_model.eval()

    base_model = AMR2Text(500, 300, data_iterator.amr_vocab_size,
                          data_iterator.parse_vocab_size,
                          data_iterator.text_vocab_size,
                          encoder_dropout=0.4, input_dropout=0.4,
                          output_dropout=0.4, recurrent_dropout=0.3)
    base_model.load_state_dict(
        torch.load('../models/amr_to_text_ldc2017t10'))
    base_model.cuda()
    base_model.eval()

    parse_questions = []
    base_questions = []

    with open('test_bleu/dev.alignments', 'r') as f, \
            open('paraphrase_survey.txt', 'w') as g, \
            open('test_bleu/dev.txt', 'r') as h:
        g.write('[[AdvancedFormat]]\n')
        detok = TreebankWordDetokenizer()
        blocks = 0
        for alignment, data_stuff, ref in (
                filter(lambda x: 15 < len(x[1][0][0]) < 60,
                       zip(f, dev_iterator, h))):
            if input(data_stuff[4]) == 'y':
                _, texts, _, copy, pointers = parse_model.predict(
                    torch.LongTensor(data_stuff[0]).cuda(), data_stuff[3],
                    data_iterator.parse_toid,
                    data_iterator.text_toid, num_parses=3,
                    one_best=False, mask=True, sample=True, temp=0.3)

                realized_sentences = realize_sentence(
                    texts[:, :, 1:].cpu().numpy(),
                    data_iterator.text_totoken,
                    copy.cpu().numpy(), pointers.cpu().numpy(),
                    data_stuff[3], base_model.end_symbol.cpu())[0]

                # Clean up the realised sentences: deanon and detokenise
                clean_sentences = [
                    detok.detokenize(
                        deanonymize_sentence_raw(
                            ' '.join(sent), alignment.strip()).split())
                    for sent in realized_sentences]

                question = Question(ref, clean_sentences)
                parse_questions.append(question)
                texts, _, copy, pointers = base_model.predict(
                    [torch.LongTensor(data_stuff[0]).cuda()], data_stuff[3],
                    data_iterator.text_toid, max_out_len=100, num_parses=3,
                    one_best=False, mask=True, sample=True, temp=0.3, nbest=3)

                realized_sentences = realize_sentence(
                    texts[:, :, 1:].cpu().numpy(),
                    data_iterator.text_totoken,
                    copy.cpu().numpy(), pointers.cpu().numpy(),
                    data_stuff[3], base_model.end_symbol.cpu())[0]

                # Clean up the realised sentences: deanon and detokenise
                clean_sentences = [
                    detok.detokenize(
                        deanonymize_sentence_raw(
                            ' '.join(sent), alignment.strip()).split())
                    for sent in realized_sentences]

                question = Question(ref, clean_sentences)
                base_questions.append(question)

            if len(base_questions) == 10:
                g.write(create_question_block(parse_questions))
                g.write(create_question_block(base_questions))
                parse_questions = []
                base_questions = []
                blocks += 1
                if blocks == 5:
                    break
