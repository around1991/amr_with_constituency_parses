import time
import tqdm
import subprocess

import warnings
warnings.filterwarnings('error', category=UserWarning)

import torch
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

from seq2seq import (Encoder, LSTMAttentionDecoder, SequenceLoss,
                     repeat_batch_items)
from data import DataLoader
from generate import realize_sentence
from parse_model import ParseTree, calculate_parse_f1, ParseActionMasker


instr = ('/home/kc391/synsem/src/test_bleu/evaluate.sh {} {} {}')


def add_noise(param, standard_deviation):
    normal_dist = torch.distributions.normal.Normal(0.0, standard_deviation)
    sample = normal_dist.sample(param.size())
    sample = sample.to(param.device)
    param.data.add_(sample)


class EMA(object):
    def __init__(self, smoothing=0.95):
        self.value = None
        self.smoothing = smoothing

    def push(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = (
                self.smoothing * self.value + (1 - self.smoothing) * value)

        return self.value


class AbstractEncoderDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AbstractEncoderDecoder, self).__init__()
        self._create_modules(*args, **kwargs)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=5)

    def encode(self, inputs):
        masks = [x != 0 for x in inputs]
        embeds = [embed(inp) for embed, inp in zip(
            self.input_embeddings, inputs)]
        encodes = [encoder(embed) for encoder, embed in zip(
            self.encoders, embeds)]

        return masks, embeds, encodes

    def calculate_losses_on_batch(self, amr_batch, parse_batch, text_batch,
                                  amr_pointers, text_pointers, device=0):
        amr_batch = torch.LongTensor(amr_batch).cuda(device)
        parse_batch = torch.LongTensor(parse_batch).cuda(device)
        text_batch = torch.LongTensor(text_batch).cuda(device)
        amr_pointers = torch.LongTensor(amr_pointers).cuda(device)
        text_pointers = torch.LongTensor(text_pointers).cuda(device)

        # Make predictions on the batches
        outs = self(
            amr_batch, parse_batch, text_batch, amr_pointers, text_pointers)

        mean_outs = [x.mean(0) for x in outs]

        return mean_outs

    def calculate_loss_with_copy(self, output_embeds, encodes, masks, targets,
                                 pointers):
        raw_loss, attention, copy = self.decoder(
            output_embeds, encodes, masks)

        raw_loss = F.log_softmax(raw_loss, dim=-1)
        # We should not predict UNK if there is a token in the AMR that matches
        # the ground truth token in the text
        vocab_losses, mask, predict_unk = self.loss_fn(
            raw_loss, targets)

        vocab_mask = 1 - (predict_unk & (pointers.sum(-1) != 0))

        vocab_prob = (1 - copy) * torch.exp(-vocab_losses * vocab_mask.float())

        pointer_prob = copy * (attention * pointers.float()).sum(-1)

        text_loss = -torch.log(vocab_prob + pointer_prob).masked_fill_(
            1 - mask, 0)

        # Normalise text_loss per word
        loss_per_word = text_loss.sum() / mask.sum().float()

        return loss_per_word

    def train_iteration(self, iterator, writer, step, labels, GAMMA=1e-5):
        self.train()
        t = tqdm.tqdm(total=len(iterator), ncols=120, smoothing=0.05)
        for (amr_batch, parse_batch, text_batch, _, _,
                amr_pointers, text_pointers) in iterator:
            outs = self.calculate_losses_on_batch(
                amr_batch, parse_batch, text_batch, amr_pointers, text_pointers)

            step += len(text_batch)

            self.optimizer.zero_grad()
            total_loss = outs[0]
            total_loss.backward()
            self.optimizer.step()

            assert len(outs[1:]) == len(labels)
            value_dict = {}
            for label, value in zip(labels, outs[1:]):
                value_numpy = value.data.cpu().numpy()
                writer.add_scalar(label, value_numpy, step)
                value_dict[label] = value_numpy

            t.update(len(text_batch))
            t.set_postfix(**value_dict)
        t.close()

        return step

    def train_loop(self, exp_name, epochs, iterator, train_labels, dev_labels,
                   dev_iterator=None, monitor='max', patience=5,
                   step=0, epoch_num=1, save=True, writer=None,
                   unsup_iterator=None):
        if not writer:
            writer = SummaryWriter('separate_parse_text_logs/' + exp_name)
        patience_counter = 0
        if patience is None:
            patience = epochs
        if monitor == 'max':
            best_dev = 0
        elif monitor == 'min':
            best_dev = float('inf')

        # Reset the scheduler based on the chosen mode
        self.scheduler.mode = monitor
        self.scheduler.patience = patience
        self.scheduler._init_is_better(
            mode=self.scheduler.mode, threshold=self.scheduler.threshold,
            threshold_mode=self.scheduler.threshold_mode)

        for epoch_num in range(epoch_num, epochs + epoch_num):
            step = self.train_iteration(
                iterator, writer, step, train_labels)

            dev_outs = self.evaluate_loop(dev_iterator, exp_name=exp_name)

            for label, value in zip(dev_labels, dev_outs):
                writer.add_scalar(label, np.array([value]), epoch_num)

            self.scheduler.step(dev_outs[0])

            if ((monitor == 'max' and dev_outs[0] > best_dev) or
                    (monitor == 'min' and dev_outs[0] < best_dev)):
                if save:
                    torch.save(self.state_dict(), '../models/' + exp_name)
                patience_counter = 0
                best_dev = dev_outs[0]
            else:
                patience_counter += 1

            if patience_counter > patience:
                patience_counter = 0

        return step, epoch_num + 1, best_dev

    def evaluate_loop(self, data_iterator, **kwargs):
        self.eval()
        per_batch_outs = []
        with torch.no_grad():
            for amr_batch, parse_batch, text_batch, _, _ in data_iterator:
                eval_batch_outs = self.calculate_losses_on_batch(
                    amr_batch, parse_batch, text_batch)
                per_batch_outs.append(
                    [x.view(-1).data.cpu().numpy() for x in eval_batch_outs[1:]]
                )
        per_batch_outs = np.vstack(per_batch_outs)

        return per_batch_outs.mean(0).tolist()

    def predict(self, inputs, input_sentences, toid, max_out_len,
                action_masker=None, sample=False, **kwargs):
        embedded_inputs = list(embedding(inp) for embedding, inp in zip(
            self.input_embeddings, inputs))

        encoded_inputs = list(encoder(embed) for encoder, embed in zip(
            self.encoders, embedded_inputs))

        masks = [(inp != 0) for inp in inputs]

        if sample:
            pred_fn = self.decoder.predict
        else:
            pred_fn = self.decoder.beam_predict

        (predicted_outs, log_probs, copy_history,
         pointers) = pred_fn(
             encoded_inputs, masks, self.start_symbol, self.end_symbol,
             max_out_len, self.output_embeddings, input_sentences, toid,
             action_masker=action_masker, **kwargs)

        return predicted_outs, log_probs, copy_history, pointers


class AMR2Parse(AbstractEncoderDecoder):
    def _create_modules(self, hidden_size, embed_size, amr_vocab_size,
                        parse_vocab_size, text_vocab_size,
                        encoder_dropout, input_dropout,
                        recurrent_dropout, output_dropout, from_text=False):
        self.from_text = from_text
        if self.from_text:
            self.amr_embeddings = nn.Embedding(
                text_vocab_size, embed_size, padding_idx=0)
        else:
            self.amr_embeddings = nn.Embedding(
                amr_vocab_size, embed_size, padding_idx=0)
        self.input_embeddings = nn.ModuleList([self.amr_embeddings])
        self.amr_embeddings = None

        self.output_embeddings = nn.Embedding(
            parse_vocab_size, embed_size, padding_idx=0)

        self.register_buffer(
            'start_symbol', torch.LongTensor([parse_vocab_size - 3]))
        self.register_buffer(
            'end_symbol', torch.LongTensor([parse_vocab_size - 2]))

        self.amr_encoder = nn.Sequential(
            Encoder(embed_size, hidden_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout),
            Encoder(hidden_size, hidden_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout))
        self.encoders = nn.ModuleList([self.amr_encoder])
        self.amr_encoder = None

        self.decoder = LSTMAttentionDecoder(
            embed_size, hidden_size, parse_vocab_size, 1,
            self.output_embeddings.weight, input_dropout, recurrent_dropout,
            output_dropout)

        self.loss_fn = SequenceLoss(
            ignore_index=0, reduce_dim=None, unk_index=-1)

    def calculate_loss(self, output_embeds, encodes, masks, targets, pointers,
                       action_masks=None):
        parse_predict, _, _ = self.decoder(
            output_embeds, encodes, masks)

        if action_masks is not None:
            action_masks = torch.ByteTensor(action_masks).to(
                parse_predict.device)
            parse_predict.masked_fill_(1 - action_masks, -float('inf'))
        parse_predict = F.log_softmax(parse_predict, dim=2)
        parse_loss, mask, _ = self.loss_fn(
            parse_predict, targets)

        parse_loss = parse_loss.masked_fill_(1 - mask, 0)
        parse_loss = parse_loss.sum() / mask.sum().float()

        return parse_loss

    def forward(self, amr_batch, parse_batch, text_batch,
                amr_pointers, text_pointers):

        batch_size = parse_batch.shape[0]
        if self.from_text:
            inputs = [text_batch]
        else:
            inputs = [amr_batch]
        masks, _, encodes = self.encode(inputs)

        parse_embed = self.output_embeddings(parse_batch)

        maskers = [
            ParseActionMasker(data_iterator.parse_toid) for __ in range(
                batch_size)]
        action_masks = np.array([
            masker.return_series_of_masks(parse) for masker, parse in zip(
                maskers, parse_batch.cpu().numpy())])[:, :-1]

        parse_loss = self.calculate_loss(
            parse_embed, encodes, masks, parse_batch[:, 1:].contiguous(), None,
            action_masks)

        return parse_loss, parse_loss.mean(0)

    def evaluate_loop(self, data_iterator, exp_name):
        self.eval()
        losses = []
        labelled_f1s = []
        unlabelled_f1s = []

        with torch.no_grad():
            for (amr_batch, parse_batch, text_batch, amr_sentences, _,
                    amr_pointers, text_pointers) in data_iterator:
                _, parse_loss = self.calculate_losses_on_batch(
                    amr_batch, parse_batch, text_batch,
                    amr_pointers, text_pointers)

                if self.from_text:
                    input_batch = torch.LongTensor(text_batch).cuda()
                else:
                    input_batch = torch.LongTensor(amr_batch).cuda()

                parses, _, _, _ = self.predict(
                    [input_batch], amr_sentences, data_iterator.parse_toid, 200,
                    action_masker=ParseActionMasker, max_beam_size=2,
                    sample=False)

                losses.append(parse_loss.view(-1).cpu().numpy())
                parse_string = realize_sentence(
                    parses[:, :, 1:].cpu().numpy(),
                    data_iterator.parse_totoken,
                    np.zeros_like(parses[:, :].cpu().numpy()),
                    np.zeros_like(parses[:, :].cpu().numpy()),
                    amr_sentences, self.end_symbol.cpu())
                parse_string = [x[0] for x in parse_string]

                gold_parse_string = realize_sentence(
                    parse_batch[:, None, 1:],
                    data_iterator.parse_totoken,
                    np.zeros_like(parse_batch[:, None, :]),
                    np.zeros_like(parse_batch[:, None, :]),
                    amr_sentences, self.end_symbol.cpu())
                gold_parse_string = [x[0] for x in gold_parse_string]

                parse_trees = list(
                    map(ParseTree,
                        map(lambda x: ' '.join(x), parse_string)))
                gold_trees = list(
                    map(ParseTree,
                        map(lambda x: ' '.join(x), gold_parse_string)))

                batch_lf1, batch_uf1 = list(
                    zip(*map(
                        calculate_parse_f1, parse_trees, gold_trees)))

                labelled_f1s.extend(batch_lf1)
                unlabelled_f1s.extend(batch_uf1)

            average_parse_loss = np.average(losses)
            average_lf1 = np.average(labelled_f1s)
            average_uf1 = np.average(unlabelled_f1s)

            return average_parse_loss, average_lf1, average_uf1


class AMRParse2Text(AbstractEncoderDecoder):
    def _create_modules(self, hidden_size, embed_size, amr_vocab_size,
                        parse_vocab_size, text_vocab_size, encoder_dropout,
                        input_dropout, recurrent_dropout, output_dropout,
                        word_vecs, use_copy):
        self.amr_embeddings = nn.Embedding(
            amr_vocab_size, embed_size, padding_idx=0)
        self.parse_embeddings = nn.Embedding(
            parse_vocab_size, embed_size, padding_idx=0)
        self.input_embeddings = nn.ModuleList(
            [self.amr_embeddings, self.parse_embeddings])

        self.amr_encoder = nn.Sequential(
            Encoder(embed_size, hidden_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout),
            Encoder(hidden_size, hidden_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout))
        self.parse_encoder = nn.Sequential(
            Encoder(embed_size, hidden_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout),
            Encoder(hidden_size, hidden_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout))
        self.encoders = nn.ModuleList([self.amr_encoder, self.parse_encoder])

        self.output_embeddings = nn.Embedding(
            text_vocab_size, embed_size, padding_idx=0)

        if word_vecs is not None:
            assert word_vecs.shape[0] == text_vocab_size
            self.output_embeddings.weight = nn.Parameter(
                torch.FloatTensor(word_vecs))

        self.register_buffer(
            'start_symbol', torch.LongTensor([text_vocab_size - 3]))
        self.register_buffer(
            'end_symbol', torch.LongTensor([text_vocab_size - 2]))

        self.decoder = LSTMAttentionDecoder(
            embed_size, hidden_size, text_vocab_size, 2,
            self.output_embeddings.weight, input_dropout, recurrent_dropout,
            output_dropout, use_copy=use_copy)

        self.loss_fn = SequenceLoss(
            ignore_index=0, reduce_dim=None, unk_index=text_vocab_size - 1)

        self.LAMBDA = 0.0

    def forward(self, amr_batch, parse_batch, text_batch,
                amr_pointers, text_pointers):
        inputs = [amr_batch, parse_batch]

        masks, embeds, encodes = self.encode(inputs)
        embed = self.output_embeddings(text_batch)

        loss = self.calculate_loss_with_copy(
            embed, encodes, masks, text_batch[:, 1:].contiguous(), amr_pointers)

        return loss, loss

    def evaluate_loop(self, data_iterator, eval_bleu, exp_name, fast_eval):
        self.eval()
        bleu_scores = []
        losses = []

        with torch.no_grad():
            with open(exp_name, 'w') as f:
                for (amr_batch, parse_batch, text_batch,
                        amr_sentences, _,
                        amr_pointers, text_pointers) in data_iterator:
                    _, text_loss = self.calculate_losses_on_batch(
                        amr_batch, parse_batch, text_batch,
                        amr_pointers, text_pointers)

                    losses.append(text_loss.view(-1).data.cpu().numpy())

                    if eval_bleu:
                        texts, _, copy_history, pointers = self.predict(
                            [torch.LongTensor(amr_batch).cuda(),
                             torch.LongTensor(parse_batch).cuda()],
                            amr_sentences, data_iterator.text_toid, 100)
                        token_lists = realize_sentence(
                            texts.cpu().numpy(), data_iterator.text_totoken,
                            copy_history.cpu().numpy(), pointers.cpu().numpy(),
                            amr_sentences, self.end_symbol.cpu())
                        for token_list in token_lists:
                            f.write(' '.join(token_list) + '\n')

            with open('{}.metrics'.format(exp_name)) as f:
                for line in f:
                    if 'BLEU' in line.strip():
                        score = float(line.split(',')[0].split('=')[1])
                        bleu_scores.append(score)

        average_text_loss = np.average(np.concatenate(losses))
        average_bleu_score = np.average(bleu_scores)

        return average_bleu_score, average_text_loss


class AMR2Parse2Text(AbstractEncoderDecoder):
    def _create_modules(self, hidden_size, embed_size, amr_vocab_size,
                        parse_vocab_size, text_vocab_size, encoder_dropout=0.0,
                        input_dropout=0.0, recurrent_dropout=0.0,
                        output_dropout=0.0, split_gpu=False, word_vecs=None,
                        use_copy=True, share_parameters=True):

        # The full model consists of a parse model and a text model
        self.parse_model = AMR2Parse(
            hidden_size, embed_size, amr_vocab_size, parse_vocab_size,
            text_vocab_size, encoder_dropout, input_dropout, recurrent_dropout,
            output_dropout)
        self.text_model = AMRParse2Text(
            hidden_size, embed_size, amr_vocab_size, parse_vocab_size,
            text_vocab_size, encoder_dropout, input_dropout, recurrent_dropout,
            output_dropout, word_vecs, use_copy)

        if share_parameters:
            # We should share the AMR and parse embeddings and AMR encoder
            # between the two models
            self.parse_model.input_embeddings = nn.ModuleList([
                self.text_model.input_embeddings[0]])
            self.parse_model.encoders = nn.ModuleList([
                self.text_model.encoders[0]])
            self.parse_model.output_embeddings = (
                self.text_model.input_embeddings[1])
            self.parse_model.decoder._to_logits.weight = (
                self.parse_model.output_embeddings.weight)

        self.amr_parser = QText2AMR(
            hidden_size, embed_size, amr_vocab_size, parse_vocab_size,
            text_vocab_size, encoder_dropout, input_dropout, recurrent_dropout,
            output_dropout, word_vecs)

        self.split_gpu = split_gpu
        if self.split_gpu:
            self.amr_parser.to(2)
            self.parse_model.to(1)
            self.text_model.to(0)

    def load_from_pretrained(self, parse_model_path, text_model_path,
                             parser_path=None):

        self.parse_model.load_state_dict(torch.load(parse_model_path))
        self.text_model.load_state_dict(torch.load(text_model_path))

        if parser_path:
            self.amr_parser.load_state_dict(torch.load(parser_path))

    def calculate_loss(self, output_embeds, encodes, masks, targets, pointers,
                       action_masks=None):
        # Output_embeds is a list, as are targets
        # Calculate parse loss
        parse_loss = self.parse_model.calculate_loss(
            output_embeds[0], encodes[:1], masks[:1], targets[0], None,
            action_masks)

        text_loss = self.text_model.calculate_loss_with_copy(
            output_embeds[1], encodes, masks, targets[1], pointers)

        return parse_loss + text_loss, parse_loss, text_loss

    def forward(self, amr_batch, parse_batch, text_batch,
                amr_pointers, text_pointers):
        batch_size = amr_batch.shape[0]
        # We expect to see full (amr, parse, text) batches. Complain if we don't
        if parse_batch is None:
            assert False  # should never happen

        inputs = [amr_batch, parse_batch]
        masks, embeds, encodes = self.text_model.encode(inputs)

        parse_embed = embeds[1]
        text_embed = self.text_model.output_embeddings(text_batch)

        maskers = [
            ParseActionMasker(data_iterator.parse_toid) for __ in range(
                batch_size)]
        action_masks = np.array([
            masker.return_series_of_masks(parse) for masker, parse in zip(
                maskers, parse_batch.cpu().numpy())])[:, :-1]

        output_embeds = [parse_embed, text_embed]
        targets = [
            parse_batch[:, 1:].contiguous(), text_batch[:, 1:].contiguous()]

        overall_loss, parse_loss, text_loss = self.calculate_loss(
            output_embeds, encodes, masks, targets, amr_pointers, action_masks)
        return (overall_loss, parse_loss, text_loss)

    def predict(self, amr_batch, input_sentences, parse_toid, text_toid,
                num_parses=5, gold_parse=None,
                max_parse_len=200, max_text_len=100, mask=True, one_best=True,
                **kwargs):
        # Predict a parse using the sampling method
        batch_size = amr_batch.size()[0]
        action_masker = ParseActionMasker if mask else None
        if gold_parse is None:
            predicted_parses, parse_log_probs, _, _ = self.parse_model.predict(
                [amr_batch], None, parse_toid, max_parse_len, nbest=num_parses,
                max_beam_size=num_parses, action_masker=action_masker, **kwargs)
            # Now flatten predicted_parses
        else:
            predicted_parses = gold_parse
            parse_log_probs = torch.zeros(batch_size)

        flat_predicted_parses = predicted_parses.view(
            -1, predicted_parses.size()[-1])
        if np.ndim(predicted_parses) == 2:
            predicted_parses = predicted_parses.unsqueeze(1)

        # Repeat the AMRs num_parses times. Don't forget to repeat the sentences
        # too!
        repeated_amr_batch = repeat_batch_items(amr_batch, num_parses)
        new_sentences = [
            item for item in input_sentences for i in range(num_parses)]

        # Be a bit clever here: we don't really care about low probability text
        # sequences, so just return the best hypothesis found by beam search
        (predicted_texts, text_log_probs,
         copy_history, pointers) = self.text_model.predict(
            [repeated_amr_batch, flat_predicted_parses],
             new_sentences, text_toid, max_text_len, max_beam_size=2,
             sample=False)

        # Reshape predicted_texts and the text scores to be the same size as
        # predicted_parses (we only return 1 text per parse so this is fine)
        predicted_texts = predicted_texts.view(batch_size, num_parses, -1)
        copy_history = copy_history.view(batch_size, num_parses, -1)
        pointers = pointers.view(batch_size, num_parses, -1)
        text_log_probs = text_log_probs.view(*parse_log_probs.size())

        # Find the jointly best scoring hypothesis for each batch item
        if gold_parse is None:
            total_log_prob = text_log_probs + parse_log_probs
            if one_best:
                best_scores, best_hyps = total_log_prob.max(1)

                best_hyps = best_hyps.squeeze()
                best_parse = predicted_parses[
                    torch.arange(batch_size), best_hyps, :].unsqueeze(1)
                best_text = predicted_texts[
                    torch.arange(batch_size), best_hyps, :].unsqueeze(1)
                pointers = pointers[
                    torch.arange(batch_size), best_hyps, :].unsqueeze(1)
                copy_history = copy_history[
                    torch.arange(batch_size), best_hyps, :].unsqueeze(1)
            else:
                best_parse, best_text, best_scores = (
                    predicted_parses, predicted_texts, total_log_prob)
        else:
            best_parse = gold_parse
            best_scores = text_log_probs
            best_text = predicted_texts.squeeze()
            pointers = pointers.squeeze()
            copy_history = copy_history.squeeze()

        return best_parse, best_text, best_scores, copy_history, pointers

    def evaluate_loop(self, data_iterator, exp_name=None, **kwargs):
        evaluate_parse = False
        self.eval()
        parse_losses = []
        text_losses = []
        bleu_scores = []
        labelled_f1s = []
        unlabelled_f1s = []
        parses = []
        if 'num_parses' in kwargs:
            num_parses = kwargs['num_parses']
        else:
            num_parses = 1
        with torch.no_grad():
            with open(exp_name, 'w') as f:
                for (amr_batch, parse_batch, text_batch,
                     amr_sentences, _,
                     amr_pointers, text_pointers) in data_iterator:
                    _, parse_loss, text_loss = self.calculate_losses_on_batch(
                        amr_batch, parse_batch, text_batch,
                        amr_pointers, text_pointers)

                    parse_losses.append(parse_loss.view(-1).data.cpu().numpy())
                    text_losses.append(text_loss.view(-1).data.cpu().numpy())

                    parses, texts, _, copy, pointers = self.predict(
                        torch.LongTensor(amr_batch).cuda(),
                        amr_sentences, data_iterator.parse_toid,
                        data_iterator.text_toid, mask=True,
                        num_parses=num_parses)

                    token_lists = realize_sentence(
                        texts[:, :, 1:].cpu().numpy(),
                        data_iterator.text_totoken,
                        copy.cpu().numpy(), pointers.cpu().numpy(),
                        amr_sentences, self.text_model.end_symbol.cpu())
                    for token_list in token_lists:
                        f.write(
                            '||'.join(' '.join(x) for x in token_list) + '\n')

                    # Now compute the parse accuracy
                    if evaluate_parse:
                        parse_string = realize_sentence(
                            parses[:, :, 1:].cpu().numpy(),
                            data_iterator.parse_totoken,
                            np.zeros_like(parses[:, None, :].cpu().numpy()),
                            np.zeros_like(parses[:, None, :].cpu().numpy()),
                            amr_sentences, self.parse_model.end_symbol.cpu())
                        parse_string = [x[0] for x in parse_string]

                        gold_parse_string = realize_sentence(
                            parse_batch[:, None, 1:],
                            data_iterator.parse_totoken,
                            np.zeros_like(parse_batch[:, None, :]),
                            np.zeros_like(parse_batch[:, None, :]),
                            amr_sentences, self.parse_model.end_symbol.cpu())
                        gold_parse_string = [x[0] for x in parse_string]

                        parse_trees = list(
                            map(ParseTree,
                                map(lambda x: ' '.join(x), parse_string)))
                        gold_trees = list(
                            map(ParseTree,
                                map(lambda x: ' '.join(x), gold_parse_string)))

                        batch_lf1, batch_uf1 = list(
                            zip(*map(
                                calculate_parse_f1, parse_trees, gold_trees)))

                        labelled_f1s.extend(batch_lf1)
                        unlabelled_f1s.extend(batch_uf1)

            subprocess.call(instr.format(exp_name,
                                         'test_bleu/dev.txt',
                                         'test_bleu/dev.alignments'),
                            shell=True)

            with open('{}.metrics'.format(exp_name)) as f:
                for line in f:
                    if 'BLEU' in line.strip():
                        score = float(line.split(',')[0].split('=')[1])
                        bleu_scores.append(score)

        average_parse_loss = np.average(np.concatenate(parse_losses))
        average_text_loss = np.average(np.concatenate(text_losses))
        average_bleu_score = np.average(bleu_scores)
        average_lf1 = np.average(labelled_f1s)
        average_uf1 = np.average(unlabelled_f1s)

        return (average_bleu_score, average_parse_loss, average_text_loss,
                average_lf1, average_uf1)


class QText2AMR(AbstractEncoderDecoder):
    # Need to be able to train this module, and sample from it and get an
    # associated model score
    def __init__(self, hidden_size, embed_size, amr_vocab_size,
                 parse_vocab_size, text_vocab_size, encoder_dropout=0.0,
                 input_dropout=0.0, recurrent_dropout=0.0, output_dropout=0.0,
                 word_vecs=None):
        super(QText2AMR, self).__init__(
            hidden_size, embed_size, amr_vocab_size, parse_vocab_size,
            text_vocab_size, encoder_dropout, input_dropout, recurrent_dropout,
            output_dropout, word_vecs)

    def _create_modules(self, hidden_size, embed_size, amr_vocab_size,
                        parse_vocab_size, text_vocab_size, encoder_dropout,
                        input_dropout, recurrent_dropout, output_dropout,
                        word_vecs):

        self.text_embeddings = nn.Embedding(
            text_vocab_size, embed_size, padding_idx=0)
        if word_vecs is not None:
            assert word_vecs.shape[0] == text_vocab_size
            self.text_embeddings.weight = nn.Parameter(
                torch.FloatTensor(word_vecs))
        self.input_embeddings = nn.ModuleList([self.text_embeddings])

        self.text_encoder = nn.Sequential(
            Encoder(embed_size, encoder_dropout, encoder_dropout, hidden_size),
            Encoder(hidden_size, encoder_dropout, encoder_dropout, hidden_size)
        )
        self.encoders = nn.ModuleList([self.text_encoder])

        self.output_embeddings = nn.Embedding(
            amr_vocab_size, embed_size, padding_idx=0)

        self.decoder = LSTMAttentionDecoder(
            embed_size, hidden_size, amr_vocab_size, 1,
            self.output_embeddings.weight, input_dropout, recurrent_dropout,
            output_dropout)

        self.register_buffer(
            'start_symbol', torch.LongTensor([amr_vocab_size - 3]))
        self.register_buffer(
            'end_symbol', torch.LongTensor([amr_vocab_size - 2]))

    def forward(self, amr_batch, parse_batch, text_batch, predict_unk=True):
        # If we call forward, we're training on supervised data

        text_mask = (text_batch != 0)
        text_embed = self.text_embeddings(text_batch)
        text_encode = self.text_encoder(text_embed)

        amr_embed = self.output_embeddings(amr_batch)

        amr_predict = self.decoder(
            amr_embed, [text_encode], [text_mask], predict_unk=predict_unk)

        amr_loss = self.loss_fn(amr_predict, amr_batch)

        return amr_loss, amr_loss.mean(0)


class AMR2Text(AbstractEncoderDecoder):
    def _create_modules(self, hidden_size, embed_size, amr_vocab_size,
                        parse_vocab_size, text_vocab_size, encoder_dropout=0.0,
                        input_dropout=0.0, recurrent_dropout=0.0,
                        output_dropout=0.0, word_vecs=None, use_copy=True):

        self.input_embeddings = nn.ModuleList(
            [nn.Embedding(amr_vocab_size, embed_size, padding_idx=0)])
        self.encoders = nn.ModuleList([nn.Sequential(
            Encoder(embed_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout, hidden_size),
            Encoder(hidden_size, encoder_dropout, encoder_dropout,
                    recurrent_dropout, hidden_size))])

        self.output_embeddings = nn.Embedding(
            text_vocab_size, embed_size, padding_idx=0)

        if word_vecs is not None:
            assert word_vecs.shape[0] == text_vocab_size
            self.output_embeddings.weight = nn.Parameter(
                torch.FloatTensor(word_vecs))

        self.register_buffer(
            'start_symbol', torch.LongTensor([text_vocab_size - 3]))
        self.register_buffer(
            'end_symbol', torch.LongTensor([text_vocab_size - 2]))

        self.decoder = LSTMAttentionDecoder(
            embed_size, hidden_size, text_vocab_size, 1,
            self.output_embeddings.weight, input_dropout, recurrent_dropout,
            output_dropout, use_copy=use_copy)

        self.loss_fn = SequenceLoss(
            ignore_index=0, reduce_dim=None, unk_index=text_vocab_size - 1)

    def forward(self, amr_batch, parse_batch, text_batch,
                amr_pointers, text_pointers):
        inputs = [amr_batch]
        masks, embeds, encodes = self.encode(inputs)

        embed = self.output_embeddings(text_batch)
        loss = self.calculate_loss_with_copy(
            embed, encodes, masks, text_batch[:, 1:].contiguous(), amr_pointers)

        return loss, loss

    def evaluate_loop(self, data_iterator, exp_name=None, **kwargs):
        self.eval()
        bleu_scores = []
        losses = []

        with torch.no_grad():
            with open(exp_name, 'w') as f:
                for (amr_batch, parse_batch, text_batch,
                        amr_sentences, _,
                        amr_pointers, text_pointers) in data_iterator:
                    _, text_loss = self.calculate_losses_on_batch(
                        amr_batch, parse_batch, text_batch,
                        amr_pointers, text_pointers)

                    losses.append(text_loss.view(-1).data.cpu().numpy())

                    texts, _, copy_history, pointers = self.predict(
                        [torch.LongTensor(amr_batch).cuda()],
                        amr_sentences, data_iterator.text_toid, 100)
                    token_lists = realize_sentence(
                        texts[:, 1:].unsqueeze(1).cpu().numpy(),
                        data_iterator.text_totoken,
                        copy_history.unsqueeze(1).cpu().numpy(),
                        pointers.unsqueeze(1).cpu().numpy(),
                        amr_sentences, self.end_symbol.cpu())
                    for token_list in token_lists:
                        f.write('||'.join(' '.join(x) for x in token_list) + '\n')

        subprocess.call(instr.format(exp_name, 'test_bleu/test.txt', 'test_bleu/test.alignments'), shell=True)

        with open('{}.metrics'.format(exp_name)) as f:
            for line in f:
                if 'BLEU' in line.strip():
                    score = float(line.split(',')[0].split('=')[1])
                    bleu_scores.append(score)

        average_text_loss = np.average(np.concatenate(losses))
        average_bleu_score = np.average(bleu_scores)

        return average_bleu_score, average_text_loss


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size,
                 input_dropout=0.5, recurrent_dropout=0.3, output_dropout=0.5):
        super(LanguageModel, self).__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(
            self.vocab_size, self.embed_size, padding_idx=0).cuda()

        self.decoder_1 = LSTMAttentionDecoder(
            embed_size, hidden_size, embed_size, 0,
            None, input_dropout, recurrent_dropout,
            output_dropout).cuda()
        self.decoder_2 = LSTMAttentionDecoder(
            embed_size, hidden_size, vocab_size, 0,
            self.embeddings.weight, input_dropout, recurrent_dropout,
            output_dropout).cuda()

        self.loss_fn = SequenceLoss(ignore_index=0, reduce_dim=1).cuda()
        self.optimizer = torch.optim.Adam(lr=5e-4, params=self.parameters())

    def forward(self, batch, predict_unk=True):
        batch_embed = self.embeddings(batch)
        decode_1 = self.decoder_1(batch_embed, [], [])
        logits = self.decoder_2(decode_1, [], [], predict_unk=predict_unk)

        loss = self.loss_fn(logits, batch)

        return loss, logits

    def calculate_losses_on_batch(self, batch):
        batch = torch.LongTensor(batch)
        batch = batch.cuda()

        loss = self(batch)

        return loss

    def train_epoch(self, iterator, writer, step):
        self.train()
        t = tqdm.tqdm(total=len(data_iterator), ncols=120, smoothing=0.05)
        for (amr_batch, _, _, _, _) in iterator:
            outs = self.calculate_losses_on_batch(amr_batch)
            step += len(amr_batch)

            self.optimizer.zero_grad()
            total_loss = outs[0]
            self.optimizer.step()
            self.optimizer.zero_grad()

            writer.add_scalar(
                'amr prior loss', np.array([total_loss.data.cpu().numpy()]),
                step)

            t.update(len(amr_batch))
            t.set_postfix(amr_prior_loss=total_loss.data.cpu().numpy()[0])
        t.close()

        return step

    def evaluate_loop(self, iterator):
        self.eval()
        losses = []
        for amr_batch, _, _, _, _ in iterator:
            loss = self.calculate_losses_on_batch(amr_batch)
            losses.append(loss.data.cpu().numpy())

        average_loss = np.average(np.concatenate(losses))

        return average_loss

    def train_loop(self, exp_name, epochs, data_iterator, dev_iterator,
                   patience=None):
        writer = SummaryWriter('tensorboard_logs/' + exp_name)
        patience_counter = 0
        step = 0
        best_dev = float('inf')
        if patience is None:
            patience = epochs

        for epoch_num in range(epochs):
            start = time.time()
            step = self.train_epoch(data_iterator, writer, step)
            end = time.time()

            print("epoch {} finished in {} seconds".format(
                epoch_num, end-start))

            dev_out = self.evaluate_loop(dev_iterator)

            writer.add_scalar('dev amr prior', np.array([dev_out]), epoch_num)

            if dev_out < best_dev:
                torch.save(self.state_dict(), '../models/' + exp_name)
                patience_counter = 0
                best_dev = dev_out
            else:
                patience_counter += 1

            if patience_counter > patience:
                break

        return best_dev


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_forward', action='store_true',
                            default=False)
    arg_parser.add_argument('--train_backward', action='store_true',
                            default=False)
    arg_parser.add_argument('--train_amr_prior', action='store_true',
                            default=False)
    arg_parser.add_argument('--train_amr_text', action='store_true',
                            default=False)
    arg_parser.add_argument('--train_amr_parse', action='store_true',
                            default=False)
    arg_parser.add_argument('--train_amr_parse_text', action='store_true',
                            default=False)
    arg_parser.add_argument('--continue_train', action='store_true',
                            default=False)
    arg_parser.add_argument('--test', action='store_true', default=False)
    arg_parser.add_argument('--evaluate', action='store_true', default=False)
    arg_parser.add_argument('--unsup_GAMMA', type=float, default=0.0)
    arg_parser.add_argument('--GAMMA', type=float, default=0.0)

    args = arg_parser.parse_args()

    data_iterator = DataLoader(
        '/home/kc391/synsem/data/ldc_train/', 40)

    unsup_iterator = DataLoader(
        '/home/kc391/synsem/data/nyt_200k/', 50)

    vecs = data_iterator.load_vectors(
        '/local/filespace-2/kc391/Mikolov/mikolov.pkl')

    dev_iterator = DataLoader('/home/kc391/synsem/data/ldc2015e86_dev/', 50,
                              vocabs=data_iterator.package_vocabs(),
                              max_amr_len=150, max_parse_len=200)
    dev_iterator.eval()
    test_iterator = DataLoader('/home/kc391/synsem/data/ldc2015e86_test/', 50,
                               vocabs=data_iterator.package_vocabs(),
                               max_amr_len=150, max_parse_len=200)
    test_iterator.eval()

    if args.train_forward:
        model = AMR2Parse2Text(500, 300, data_iterator.amr_vocab_size,
                               data_iterator.parse_vocab_size,
                               data_iterator.text_vocab_size,
                               encoder_dropout=0.5,
                               input_dropout=0.5, output_dropout=0.5,
                               recurrent_dropout=0.3, split_gpu=False,
                               word_vecs=vecs)
        model.cuda()
        model.train_loop(
            'forward_model_ldc2015e86_refactored_enc_rec_dropout',
            200, data_iterator, ['parse_loss', 'text_loss'],
            ['dev_bleu', 'dev_parse', 'dev_text', 'dev_lf1', 'dev_uf1'],
            dev_iterator,
            monitor='max')
    elif args.train_amr_parse:
        model = AMR2Parse(500, 300, data_iterator.amr_vocab_size,
                          data_iterator.parse_vocab_size,
                          data_iterator.text_vocab_size, encoder_dropout=0.4,
                          input_dropout=0.5, recurrent_dropout=0.3,
                          output_dropout=0.5, from_text=True)
        model.cuda()
        model.train_loop(
            'text2parse_ldc2017t10',
            200, data_iterator, ['parse_loss'],
            ['dev_parse', 'dev_lf1', 'dev_uf1'],
            dev_iterator=dev_iterator, monitor='min')
    elif args.train_amr_parse_text:
        model = AMRParse2Text(500, 300, data_iterator.amr_vocab_size,
                              data_iterator.parse_vocab_size,
                              data_iterator.text_vocab_size,
                              encoder_dropout=0.4, input_dropout=0.5,
                              recurrent_dropout=0.3, output_dropout=0.5,
                              word_vecs=vecs, use_copy=False)
        model.cuda()
        model.train_loop(
            'amrparse2text_ldc2015e86_enc_0.4_inp_0.5_rec_0.3_out_0.5_layer_norm_encoder_no_copy',
            500, data_iterator, ['text_loss'], ['dev_bleu', 'dev_text'],
            dev_iterator=dev_iterator, monitor='max')
    elif args.train_amr_text:
        model = AMR2Text(500, 300, data_iterator.amr_vocab_size,
                         data_iterator.parse_vocab_size,
                         data_iterator.text_vocab_size,
                         encoder_dropout=0.4,
                         input_dropout=0.4, output_dropout=0.4,
                         recurrent_dropout=0.3, word_vecs=vecs)
        model.cuda()
        model.train_loop('amr_to_text_ldc2015e86_with_copy_enc_rec_dropout',
                         200, data_iterator, ['text loss'],
                         ['dev bleu', 'dev text'], dev_iterator, monitor='max')
    elif args.train_backward:
        model = QText2AMR(500, 300, data_iterator.amr_vocab_size,
                          data_iterator.parse_vocab_size,
                          data_iterator.text_vocab_size,
                          encoder_dropout=0.4,
                          input_dropout=0.5, output_dropout=0.5,
                          recurrent_dropout=0.3, word_vecs=vecs)
        model.cuda()
        model.train_loop('text2amr_ldc2015e86_enc_0.4_inp_0.5_rec_0.3_out_0.5',
                         500, data_iterator,
                         ['amr loss'], ['dev amr'], dev_iterator,
                         monitor='min')
    elif args.train_amr_prior:
        model = LanguageModel(data_iterator.amr_vocab_size, 500, 300,
                              0.5, 0.4, 0.5)
        model.train_loop('amr_prior_inp_0.5_rec_0.3_out_0.5',
                         200, data_iterator, dev_iterator)
    elif args.evaluate:
        model = AMR2Parse2Text(500, 300, data_iterator.amr_vocab_size,
                               data_iterator.parse_vocab_size,
                               data_iterator.text_vocab_size,
                               encoder_dropout=0.5,
                               input_dropout=0.6, output_dropout=0.7,
                               recurrent_dropout=0.3, word_vecs=vecs,
                               use_copy=True)
        model.load_state_dict(
            torch.load('../models/forward_model_ldc2017t10_refactored_enc_rec_dropout'), strict=True)
        model.cuda()
        outs = model.evaluate_loop(
            dev_iterator, exp_name='amr_to_text_fixed', num_parses=2)
        print('bleu: {}'.format(outs[0]))

    elif args.continue_train:
        pass

    elif args.predict:
        model = AMR2Parse2Text(300, 300, data_iterator.amr_vocab_size,
                               data_iterator.parse_vocab_size,
                               data_iterator.text_vocab_size,
                               encoder_dropout=0.5,
                               input_dropout=0.6, output_dropout=0.7,
                               recurrent_dropout=0.3, split_gpu=False)
        model.eval()
        model.load_state_dict(torch.load('../models/semi-sup'),
                              strict=False)

        for amr, _, _ in dev_iterator:
            pass
