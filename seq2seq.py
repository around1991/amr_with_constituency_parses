import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from copy import deepcopy

Module = torch.jit.ScriptModule
trace = torch.jit.trace


def backward_hook(module, grad_input, grad_output):
    if any(x is not None and torch.isnan(x).any() for x in grad_output):
        import pdb
        pdb.set_trace()


def repeat_batch_items(batch, ntimes):
    '''Repeats each element in the first dimension ntimes'''
    return batch.unsqueeze(1).expand(-1, ntimes, *batch.size()[1:]).reshape(
        -1, *batch.size()[1:])


def kl_from_log_probs(q, p):
    '''Calculates KL(q || p) along last dim of p, q'''
    q_prob = torch.exp(q)
    log_diff = q - p
    kl = -(q_prob * log_diff).sum(-1)
    return kl


def sequence_kl_from_log_probs(q, p, mask):
    kls = kl_from_log_probs(q, p) * mask.float()

    normalised_kls = kls.sum(1) / mask.sum(1).float()

    return normalised_kls


def trim_predicted_batch(batch, start_symbol, end_symbol, pad_symbol):
    batch_numpy = batch.cpu().data.numpy()

    for i, pred in enumerate(batch_numpy):
        ended = False
        for j, token in enumerate(pred):
            if not ended:
                if token == end_symbol:
                    ended = True
            else:
                batch_numpy[i, j] = pad_symbol

    start_symbol_batch = np.ones((batch_numpy.shape[0], 1)) * start_symbol
    batch_numpy = np.hstack((start_symbol_batch, batch_numpy))
    batch_trimmed = Variable(torch.LongTensor(batch_numpy))
    device = batch.get_device()
    batch_trimmed = batch_trimmed.cuda(device)

    return batch_trimmed


@torch.jit.script
def compute_attention_context(key, precon_stack, stack, mask):
    attention_logits = torch.matmul(precon_stack, key.unsqueeze(2)).squeeze(2)

    inverse_mask = (mask == 0)
    attention_logits = attention_logits.masked_fill_(inverse_mask, -1e10)

    attention_weights = F.softmax(attention_logits, 1)

    context = torch.sum(stack * attention_weights.unsqueeze(2), dim=1)

    return context, attention_weights


def gumbel_ST_sample(logits, temp=1.0, hard=False):
    noise = torch.rand(logits.size())
    eps = 1e-8
    gumbel_noise = Variable(-torch.log(-torch.log(noise + eps) + eps))

    gumbel_sample = F.softmax((logits + gumbel_noise) / temp, dim=-1)

    if hard:
        one_hot_sample = (gumbel_sample == gumbel_sample.max(1)[0].unsqueeze(1))
        rv = (one_hot_sample.float() - gumbel_sample).detach() + gumbel_sample
    else:
        rv = gumbel_sample

    return rv


def replace_pointers_with_word_ids(pointers, sentences, toid):
    # pointers is an N x ... sized array of ints, sentences is a list of length
    # N
    output = []
    for pointer, sentence in zip(pointers, sentences):
        def helper(idx):
            if sentence[int(idx)] in toid:
                return toid[sentence[int(idx)]]
            else:
                return toid['<UNK>']
        word_ids = torch.LongTensor(
            list(map(helper, pointer.cpu().view(-1)))).reshape(*pointer.shape)
        output.append(word_ids)

    return torch.stack(output).to(pointers.device)


def merge_leading_dims(tensor, ndims=2):
    return tensor.reshape(-1, *tensor.shape[2:])


@torch.jit.script
def apply_dropout_mask(dropout_mask, activations, dropout_p):
    # type: (Tensor, Tensor, float) -> Tensor
    dropped_activations = (
        (activations * dropout_mask) / (1 - dropout_p)
    )

    return dropped_activations


def select_beam_item(store, index):
    # Assume batch dimension is first dimension, beam item is second dim
    return torch.gather(
        store, 1, index.unsqueeze(-1).expand(-1, -1, store.shape[-1]))


def normalize_gen_copy_scores(gen_scores, copy_prob, pointers, action_mask=None,
                              predict_unk=False):
    if action_mask is not None:
        gen_scores.masked_fill_(1 - action_mask, -float('inf'))
    norm_gen_scores = (1 - copy_prob) * F.softmax(gen_scores, dim=-1)
    if not predict_unk:
        norm_gen_scores = norm_gen_scores[:, :-1].contiguous()
    vocab_size = norm_gen_scores.shape[-1]

    norm_copy_scores = copy_prob * pointers

    scores = torch.cat(
        [torch.log(norm_gen_scores), torch.log(norm_copy_scores)], dim=-1)

    return scores, vocab_size


def process_combined_samples(samples, vocab_size, toid, sentences):
    copy_action_taken = (samples >= vocab_size)
    index_of_copy = torch.clamp(samples - vocab_size, min=0)

    if copy_action_taken.any():
        copied_word_ids = replace_pointers_with_word_ids(
            index_of_copy, sentences, toid)
        word_token = torch.where(
            copy_action_taken, copied_word_ids, samples)
    else:
        word_token = samples

    assert not ((word_token == vocab_size) & (1 - copy_action_taken)).any()

    return word_token, copy_action_taken, index_of_copy


class SequenceLoss(nn.Module):
    def __init__(self, ignore_index=-1, reduce_dim=None, unk_index=None):
        super(SequenceLoss, self).__init__()
        self.reduce_dim = reduce_dim
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, predictions, targets):
        # first column of targets is <GO>, don't predict this
        mask = (targets != self.ignore_index)
        predict_unk = (targets == self.unk_index)
        if self.reduce_dim is None:
            return (F.nll_loss(predictions.view(-1, predictions.size()[-1]),
                               targets.view(-1),
                               ignore_index=self.ignore_index,
                               reduction='none').view(*targets.shape),
                    mask, predict_unk)
        else:
            losses = F.nll_loss(predictions.view(-1, predictions.size()[-1]),
                                targets.view(-1),
                                ignore_index=self.ignore_index,
                                reduction='none').view(targets.size())
            return (losses.sum(self.reduce_dim) /
                    mask.float().sum(self.reduce_dim))


class LayerNormLSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.ln_ih = trace(
            nn.LayerNorm(4 * hidden_size), torch.rand(1, 4 * hidden_size))
        self.ln_hh = trace(
            nn.LayerNorm(4 * hidden_size), torch.rand(1, 4 * hidden_size))

    @torch.jit.script_method
    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx = hidden[0], hidden[1]
        gates = (self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) +
                 self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh)))

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class LayerNormLSTM(Module):
    __constants__ = ['recurrent_dropout']

    def __init__(self, input_size, hidden_size, recurrent_dropout=0.0):
        super(LayerNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout

        self.forward_cell = LayerNormLSTMCell(
            self.input_size, self.hidden_size)
        self.backward_cell = LayerNormLSTMCell(
            self.input_size, self.hidden_size)

        self._forward_init = nn.Parameter(
            torch.FloatTensor(2, self.hidden_size))
        self._backward_init = nn.Parameter(
            torch.FloatTensor(2, self.hidden_size))

        self._initialise()

    def _initialise(self):
        for x in self.parameters():
            if x.dim() == 1:
                x.data.normal_(0.0, 0.05)
            else:
                nn.init.xavier_uniform_(x)

    @torch.jit.script_method
    def init_hidden(self, batch_size):
        # type: (int) -> Tensor
        forward_hidden = self._forward_init[0].unsqueeze(0).expand(
            batch_size, -1)
        forward_cell = self._forward_init[1].unsqueeze(0).expand(
            batch_size, -1)
        forward_init = torch.stack((forward_hidden, forward_cell), 0)

        backward_hidden = self._backward_init[0].unsqueeze(0).expand(
            batch_size, -1)
        backward_cell = self._backward_init[1].unsqueeze(0).expand(
            batch_size, -1)
        backward_init = torch.stack((backward_hidden, backward_cell), 0)

        return torch.cat((forward_init, backward_init), 0)

    @torch.jit.script_method
    def _lstm_recurrence(self, inputs, hidden, cell_state, forward_dir):
        # type: (Tensor, Tensor, Tensor, bool) -> Tensor
        outputs = []

        if self.training and self.recurrent_dropout > 0.0:
            dropout_mask = torch.bernoulli(
                torch.rand(hidden[0].size()).fill_(
                    1 - self.recurrent_dropout))
        else:
            dropout_mask = torch.ones_like(hidden)
        dropout_mask = dropout_mask.to(inputs.device, dtype=dropout_mask.dtype)
        for i in range(inputs.shape[0]):
            hidden = apply_dropout_mask(
                dropout_mask, hidden, self.recurrent_dropout)
            if forward_dir:
                hidden, cell_state = self.forward_cell(
                    inputs[i], (hidden, cell_state))
            else:
                hidden, cell_state = self.backward_cell(
                    inputs[i], (hidden, cell_state))
            outputs.append(hidden)

        rv = torch.stack(outputs, dim=1)
        return rv

    @torch.jit.script_method
    def forward(self, input):
        batch_size = input.shape[0]

        a = self.init_hidden(batch_size)
        forward_hidden, forward_cell, backward_hidden, backward_cell = (
            a[0], a[1], a[2], a[3])

        # Permute the input so that the time axis is dimension 0
        time_first = input.permute(1, 0, 2)

        # Apply forwards LSTM cell
        forward_outputs = self._lstm_recurrence(
            time_first, forward_hidden, forward_cell, True)
        # Apply backwards LSTM cell
        backward_outputs = self._lstm_recurrence(
            torch.flip(time_first, (0,)), backward_hidden, backward_cell, False)

        # Reverse the backwards state and concatenate the two

        final_state = torch.cat([forward_outputs, backward_outputs], dim=-1)

        return final_state, final_state[:, -1, :]


class InputDropout(nn.Module):
    '''Replaces words with UNKs'''
    def __init__(self, dropout_p, replacement_id):
        super(InputDropout, self).__init__()
        self.dropout_p = dropout_p
        self.replacement_id = replacement_id

    def forward(self, batch):
        if self.training and self.dropout_p > 0:
            # Probabilities to dropout out channel
            probs = (batch != 0).float() * self.dropout_p
            mask = torch.bernoulli(probs).byte()
            out = batch.clone().masked_fill_(mask, self.replacement_id)
        else:
            out = batch

        return out


class LSTMAttentionDecoder(Module):
    '''Decoder language model with attention'''

    __constants__ = ['recurrent_dropout', 'use_copy', 'attention_heads',
                     '_attention_context_mats', 'hidden_size']

    def __init__(self, input_size, hidden_size, output_size,
                 attention_heads, output_embeds=None,
                 input_dropout=0.0, recurrent_dropout=0.0, output_dropout=0.0,
                 use_copy=False):
        super(LSTMAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention_heads = attention_heads

        self.use_copy = use_copy
        # The copy decision depends on the context vectors, the current
        # input and the decoder state
        self.copy_decision = trace(
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(
                    (self.input_size +
                        (self.attention_heads + 1) * self.hidden_size),
                    self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.Tanh(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()),
            torch.rand(
                2, self.input_size + (
                    self.attention_heads + 1) * self.hidden_size),
            check_trace=False
        )

        self.input_dropout = trace(
            nn.Dropout(input_dropout),
            torch.rand(1, self.input_size),
            check_trace=False)
        self.recurrent_dropout = recurrent_dropout
        self.output_dropout = trace(
            nn.Dropout(output_dropout),
            torch.rand(1, self.output_size),
            check_trace=False)

        self._attention_context_mats = nn.ModuleList([
            trace(
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                    nn.LayerNorm(self.hidden_size)),
                torch.rand(1, self.hidden_size))
            for _ in range(self.attention_heads)])

        self._hidden_initializer = nn.Parameter(
            torch.FloatTensor(self.hidden_size))
        self._cell_initializer = nn.Parameter(
            torch.FloatTensor(self.hidden_size))

        self._lstm_cell = LayerNormLSTMCell(self.hidden_size, self.hidden_size)

        self._merge_context_input = trace(
            nn.Sequential(
                nn.Linear((
                    self.attention_heads * self.hidden_size) + self.input_size,
                    self.hidden_size),
                nn.LayerNorm(self.hidden_size)),
            torch.rand(
                1, (self.attention_heads * self.hidden_size) + self.input_size)
        )

        self._merge_context_hidden = trace(
            nn.Sequential(
                nn.Linear(
                    (self.attention_heads * self.hidden_size) + self.hidden_size,
                    self.hidden_size),
                nn.LayerNorm(self.hidden_size)),
            torch.rand(
                1, (self.attention_heads * self.hidden_size) + self.hidden_size)
        )

        output_projection = nn.Linear(self.hidden_size, self.input_size)

        _to_logits = nn.Linear(
            self.hidden_size, self.output_size, bias=True)

        if output_embeds is not None:
            _to_logits.weight = output_embeds

        self._to_logits = trace(
            nn.Sequential(output_projection, _to_logits),
            torch.rand(1, 2, self.hidden_size))

        self._initialise()

    def _initialise(self):
        for x in self.parameters():
            if x.dim() == 1:
                x.data.normal_(0.0, 0.05)
            else:
                nn.init.xavier_uniform_(x)

    @torch.jit.script_method
    def init_hidden(self, batch_size):
        # type: (int) -> Tuple[Tensor, Tensor]
        hidden_init = self._hidden_initializer.unsqueeze(0).expand(
            batch_size, self.hidden_size)
        cell_init = self._cell_initializer.unsqueeze(0).expand(
            batch_size, self.hidden_size)
        hidden = (hidden_init, cell_init)

        return hidden

    @torch.jit.script_method
    def precondition_stacks(self, stacks):
        # type: (List[Tensor]) -> List[Tensor]
        stacks_bilinear = []
        i = 0
        for module in self._attention_context_mats:
            stack_bn = module(stacks[i])
            stacks_bilinear.append(stack_bn)
            i += 1

        return stacks_bilinear

    @torch.jit.script_method
    def _step(self, input, hidden, stacks, stacks_bilinear, masks):
        # type: (Tensor, Tuple[Tensor, Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tuple[Tuple[Tensor, Tuple[Tensor, Tensor]], List[Tensor], Tensor]

        # Compute attention weighted contexts
        contexts = []
        attention_weights = []
        for i in range(len(stacks)):
            contexts_and_logits = compute_attention_context(
                hidden[0], stacks_bilinear[i], stacks[i], masks[i])
            contexts.append(contexts_and_logits[0])
            attention_weights.append(contexts_and_logits[1])

        # Decide whether or not to copy from attention or generate from the
        # vocabulary
        if self.use_copy:
            copy_prob = self.copy_decision(
                torch.cat(contexts + [input] + [hidden[0]], -1))
        else:
            copy_prob = torch.zeros([input.shape[0], 1]).to(
                input.device, dtype=input.dtype)

        # Merge contexts with the input to the LSTM cell
        context_input_concat = torch.cat(contexts + [input], dim=1)
        new_input = torch.tanh(
            self._merge_context_input(context_input_concat))

        dropped_input = self.input_dropout(new_input)
        new_hidden = self._lstm_cell(dropped_input, hidden)

        # Merge contexts with the output of the LSTM cell
        context_hidden_concat = torch.cat(
            contexts + [new_hidden[0]], dim=1)
        out = torch.tanh(
            self._merge_context_hidden(context_hidden_concat))

        return ((out, new_hidden), attention_weights, copy_prob)

    # @torch.jit.script_method
    def forward(self, embeds, stacks, masks, predict_unk=True):
        # type: (Tensor, List[Tensor], List[Tensor], bool) -> Tuple[Tensor, Tensor, Tensor]
        # Pre-multiply the stack vectors by the bilinear matrix to save
        # computation
        assert len(stacks) == self.attention_heads
        batch_size = embeds.size(0)
        outs = []
        attentions = []
        copy_probs = []
        hidden = self.init_hidden(batch_size)
        stacks_bilinear = self.precondition_stacks(stacks)

        if self.training and self.recurrent_dropout > 0.0:
            dropout_mask = torch.bernoulli(
                torch.rand(hidden[0].size()).fill_(
                    1 - self.recurrent_dropout))
        else:
            dropout_mask = torch.ones_like(hidden[0])
        dropout_mask = dropout_mask.to(embeds.device, dtype=dropout_mask.dtype)

        time_first = embeds.permute(1, 0, 2)[:-1]

        for i in range(time_first.shape[0]):
            embed = time_first[i]
            dropped_hidden = apply_dropout_mask(
                dropout_mask, hidden[0], self.recurrent_dropout)
            new_hidden = (dropped_hidden, hidden[1])
            # Calculate the attention weight over each element of the stack
            a = self._step(embed, new_hidden, stacks, stacks_bilinear, masks)
            b, attention_weights, copy = a
            out, hidden = b
            outs.append(out)
            if len(attention_weights) > 0:
                attention_row = attention_weights[0]
            else:
                attention_row = torch.zeros_like(embed)
            attentions.append(attention_row)
            copy_probs.append(copy)

        outs = torch.stack(outs, 1)
        attentions = torch.stack(attentions, 1)
        copy_probs = torch.stack(copy_probs, 1)

        dropped_outs = self.output_dropout(outs)

        logits = self._to_logits(dropped_outs)
        if not predict_unk:
            logits = logits[:, :, :-1]

        return (logits, attentions, copy_probs.squeeze())

    def beam_predict(self, stacks, masks, start_symbol, end_symbol, nsteps,
                     embeds, sentences, toid, max_beam_size=5, alpha=0.0,
                     predict_unk=False, nbest=1, action_masker=None, **kwargs):
        assert len(stacks) == self.attention_heads
        batch_size = stacks[0].size()[0]
        hidden = self.init_hidden(batch_size)
        stacks_bilinear = self.precondition_stacks(stacks)

        completed_hypotheses = [[] for _ in range(batch_size)]

        beam_items = start_symbol.repeat(batch_size).unsqueeze(1).unsqueeze(1)
        beam_hidden = (x.unsqueeze(1) for x in hidden)
        beam_scores = Variable(torch.zeros(batch_size)).cuda().unsqueeze(-1)
        best_score = -1e5 * torch.ones_like(beam_scores)
        if action_masker is not None:
            maskers = [[action_masker(toid)] for _ in range(batch_size)]

        copy_history = torch.ByteTensor([]).cuda()
        pointers = torch.LongTensor([]).cuda()
        hyp_complete = torch.zeros_like(beam_items).squeeze(2)
        beam_ended = torch.zeros_like(beam_items).squeeze()

        for _ in range(nsteps):
            # Extend each item in the current beam
            beam_size = int(beam_items.size(1))
            current_input = beam_items[:, :, -1]
            # Update the masks as well
            if action_masker is not None:
                all_masks = []
                for i, row in enumerate(current_input.cpu().numpy()):
                    mask_row = []
                    for j, value in enumerate(row):
                        mask_row.append(maskers[i][j].return_mask(value))
                    all_masks.append(mask_row)
                action_mask_np = np.array(all_masks)
                action_mask = torch.ByteTensor(action_mask_np).view(
                    -1, action_mask_np.shape[-1]).to(stacks[0].device)
            else:
                action_mask = None
            (out, new_hidden), attn, copy_probs = self._step(
                embeds(current_input.contiguous().view(-1)),
                tuple(merge_leading_dims(x, 2) for x in beam_hidden),
                [repeat_batch_items(x, beam_size) for x in stacks],
                [repeat_batch_items(x, beam_size) for x in stacks_bilinear],
                [repeat_batch_items(x, beam_size) for x in masks])

            # Resize the outputs so that beam_size is the second dimension
            beam_hidden = tuple(
                x.view(batch_size, beam_size, x.shape[-1]) for x in new_hidden)

            logits = self._to_logits(out)

            attn[0][:, 0] = torch.zeros_like(attn[0][:, 0])

            total_scores, vocab_size = normalize_gen_copy_scores(
                logits, copy_probs, attn, action_mask, predict_unk)
            total_scores = total_scores.view(
                batch_size, beam_items.size()[1], -1)
            total_scores = (
                total_scores - 1e10 * hyp_complete.unsqueeze(-1).float())
            continuation_scores = total_scores + beam_scores.unsqueeze(-1)

            # Merge scores for all beam items together
            continuation_scores = continuation_scores.view(batch_size, -1)

            # Compute the top scoring extensions, and work out which beam_item
            # and actual token id they correspond to
            beam_scores, raw_indices = torch.topk(
                continuation_scores, max_beam_size, dim=-1)

            beam_item, index_of_cont = (
                raw_indices / total_scores.size()[-1],
                raw_indices % total_scores.size()[-1])

            # beam_item is a batch_size x max_beam_size array. Update action
            # maskers
            if action_masker is not None:
                new_maskers = [[] for _ in range(batch_size)]
                for i, batch_item in enumerate(beam_item.cpu().numpy()):
                    for j, item in enumerate(batch_item):
                        assert len(new_maskers[i]) == j
                        new_maskers[i].append(deepcopy(maskers[i][int(item)]))
                maskers = new_maskers

            (word_token, copy_action_taken,
             index_of_copy) = process_combined_samples(
                 index_of_cont, vocab_size, toid, sentences)

            # Extend the beam items with the highest scoring continuations
            selected_prefixes = select_beam_item(
                beam_items, beam_item)

            beam_items = torch.cat(
                [selected_prefixes, word_token.unsqueeze(-1)], -1)

            # Record whether we copied, and which index we copied from
            if copy_history.numel():
                copy_history = select_beam_item(copy_history, beam_item)
            copy_history = torch.cat(
                [copy_history, copy_action_taken.unsqueeze(-1)], dim=-1)

            if pointers.numel():
                pointers = select_beam_item(pointers, beam_item)
            pointers = torch.cat(
                [pointers, index_of_copy.unsqueeze(-1)], -1)

            # Update our coverage for each beam item based on the continuation
            # we selected
            beam_hidden = tuple(
                select_beam_item(x, beam_item) for x in beam_hidden)

            # Find if any hypotheses are completed: if so, add them to the
            # store and do not consider their continuations at the next step
            hyp_complete = (word_token == end_symbol)

            if hyp_complete.any():
                for x in hyp_complete.nonzero():
                    batch_item, beam_item = x[0], x[1]
                    if (beam_ended[batch_item] == 1).all():
                        continue
                    beam_item_score = beam_scores[batch_item, beam_item]
                    completed_hypotheses[int(batch_item)].append(
                        (beam_items[batch_item, beam_item],
                         beam_scores[batch_item, beam_item],
                         copy_history[batch_item, beam_item],
                         pointers[batch_item, beam_item])
                    )
                    if (beam_item_score > best_score[batch_item]).all():
                        best_score[batch_item] = beam_item_score

            beam_ended = hyp_complete.prod(1) | beam_ended
            if beam_ended.prod():
                break

        # If we haven't completed a hypothesis, grab the highest scoring
        # hypothesis currently on the beam, and return at least nbest hypotheses
        uncomplete = [i for i in range(len(completed_hypotheses))
                      if len(completed_hypotheses[i]) < nbest]

        for i in uncomplete:
            j = 1
            while len(completed_hypotheses[i]) < nbest:
                completed_hypotheses[i].append(
                    (beam_items[i, -j], beam_scores[i, -j],
                     copy_history[i, -j], pointers[i, -j]))
                j += 1

        # Sort the completed hypotheses by score and return the highest scoring
        sorted_hyps = [sorted(
            x, reverse=True,
            key=lambda y: (float(y[1]) / (y[0].numel() - 1) ** alpha))
            for x in completed_hypotheses]

        # Pack the sequences into a batch_size x nbest x seq_len array

        best_hyps = [x[0] for y in sorted_hyps for x in y[:nbest]]
        scores = [x[1] for y in sorted_hyps for x in y[:nbest]]
        copy_history = [x[2] for y in sorted_hyps for x in y[:nbest]]
        pointers = [x[3] for y in sorted_hyps for x in y[:nbest]]

        max_len = max(x.numel() for x in best_hyps)

        device = stacks[0].get_device()
        sequences = torch.zeros(batch_size, nbest, max_len).long().cuda(device)
        copies = torch.zeros(batch_size, nbest, max_len).long().cuda(device)
        pointers_store = torch.zeros(batch_size, nbest, max_len).long().cuda()

        assert len(best_hyps) == (batch_size * nbest)
        for index, (sequence, copy, pointers) in enumerate(
                zip(best_hyps, copy_history, pointers)):
            batch_element, best_num = index // nbest, index % nbest
            leng = sequence.numel() - 1
            sequences[
                batch_element, best_num, :sequence.numel()] = sequence.view(-1)
            copies[batch_element, best_num, :leng] = copy.view(-1)
            pointers_store[batch_element, best_num, :leng] = pointers.view(-1)

        return (sequences,
                torch.stack(scores).view(batch_size, nbest),
                copies, pointers_store)

    def predict(self, stacks, masks, start_symbol, end_symbol, nsteps, embeds,
                sentences, toid, predict_unk=False, alpha=0.0, temp=1.0,
                sequence_probs=False, batch_size=None, nbest=1,
                action_masker=None, **kwargs):
        '''Samples the best action sequence with temperature temp'''
        if stacks:
            batch_size = stacks[0].size(0)
        else:
            assert batch_size is not None

        hidden = self.init_hidden(batch_size * nbest)
        stacks_bilinear = self.precondition_stacks(stacks)

        # Now repeat stacks and stacks_bilinear nbest times
        stacks = [repeat_batch_items(stack, nbest) for stack in stacks]
        stacks_bilinear = [repeat_batch_items(stack, nbest) for stack in
                           stacks_bilinear]
        masks = [repeat_batch_items(mask, nbest) for mask in masks]

        wordid = start_symbol.repeat(batch_size * nbest)
        embed = embeds(wordid)
        samples = []
        copies = []
        pointers_store = []
        sample_log_probs = []
        ended = torch.zeros_like(start_symbol).repeat(batch_size * nbest).byte()
        if action_masker is not None:
            maskers = [action_masker(toid) for _ in range(batch_size * nbest)]

        for _ in range(nsteps):
            (out, hidden), attn, copy_probs = self._step(
                embed, hidden, stacks, stacks_bilinear, masks)

            logits = self._to_logits(out)
            if action_masker is not None:
                all_masks = torch.ByteTensor([
                    masker.return_mask(value) for masker, value in zip(
                        maskers, wordid.view(-1).cpu().numpy())]).to(
                            wordid.device)
            else:
                all_masks = None

            predict_logits = logits / temp

            all_logits, vocab_size = normalize_gen_copy_scores(
                predict_logits, copy_probs, attn, all_masks, predict_unk)

            dist = torch.distributions.categorical.Categorical(
                logits=all_logits)
            sample = dist.sample()
            log_prob = dist.log_prob(sample)

            wordid, copy, pointers = process_combined_samples(
                sample.view(batch_size, nbest), vocab_size, toid, sentences)

            samples.append(wordid)
            copies.append(copy.view(batch_size, nbest))
            pointers_store.append(pointers.view(batch_size, nbest))
            sample_log_probs.append(log_prob)

            embed = embeds(wordid.view(-1))

            ended = ended | (wordid.view(-1) == end_symbol)
            if ended.all():
                break

        # Combine the samples and the log_probs into a single tensor, mask the
        # log_probs and find the sequence probability.

        # This gets a bit tricky because we have an additional <GO> symbol at
        # the start of tensor_samples, which effectively has probability 1 and
        # shouldn't contribute to the log probability.
        tensor_samples = trim_predicted_batch(
            torch.stack(samples, -1).squeeze(),
            start_symbol.cpu().data.numpy(),
            end_symbol.cpu().data.numpy(), 0)

        mask = (tensor_samples[:, 1:] != 0)
        inverse_mask = (mask == 0)
        masked_sample_log_probs = torch.stack(
            sample_log_probs, 1).masked_fill_(inverse_mask, 0)
        masked_log_probs = masked_sample_log_probs * mask.float()
        # Sequence log probs are per word
        if sequence_probs:
            sequence_log_probs = masked_log_probs.sum(1)
            sequence_log_probs /= (mask.sum(1) - 1).float() ** alpha
        else:
            sequence_log_probs = masked_log_probs.sum(1) / mask.sum(1).float()

        # Reshape the samples and the scores into batch_size x nbest x length
        tensor_samples = tensor_samples.reshape(batch_size, nbest, -1)
        sequence_log_probs = sequence_log_probs.reshape(batch_size, nbest)
        copies = torch.stack(copies, dim=-1)
        pointers = torch.stack(pointers_store, dim=-1)

        return tensor_samples, sequence_log_probs, copies, pointers


class Encoder(Module):
    '''
    Takes a variable-length sequence and encodes it into a sequence of vectors
    '''

    __constants = ['input_dropout', 'output_dropout', 'recurrent_dropout',
                   'lstm_hidden', 'hidden_size']

    def __init__(self, input_size, hidden_size, input_dropout=0.0,
                 output_dropout=0.0, recurrent_dropout=0.0):
        super(Encoder, self).__init__()

        self.input_dropout = trace(
            nn.Dropout(input_dropout),
            torch.rand(1, 2, input_size),
            check_trace=False)
        self.encoder = LayerNormLSTM(
            input_size, hidden_size, recurrent_dropout)
        self.projection = trace(
            nn.Linear(2*hidden_size, hidden_size),
            torch.rand(1, 2, 2*hidden_size))
        self.output_dropout = trace(
            nn.Dropout(output_dropout),
            torch.rand(1, 2, hidden_size),
            check_trace=False
        )

        self._initialise()

    def _initialise(self):
        for x in self.parameters():
            if x.dim() == 1:
                x.data.normal_(0.0, 0.05)
            else:
                nn.init.xavier_uniform_(x)

    # @torch.jit.script_method
    def forward(self, input_embed):
        dropped_input = self.input_dropout(input_embed)
        encoding, _ = self.encoder(dropped_input)

        proj_encoding = self.projection(encoding)
        dropped_proj = self.output_dropout(proj_encoding)

        return torch.tanh(dropped_proj)


if __name__ == '__main__':
    test_logits = Variable(torch.rand(4, 10))
    test_logit_params = Variable(torch.rand(4, 10), requires_grad=True)
    sample = gumbel_ST_sample(test_logits * test_logit_params,
                              hard=True)

    sample_cat = sample.max(1)[1]

    test_embeds = Variable(torch.rand(10, 5), requires_grad=True)

    out = torch.mm(sample, test_embeds.detach()).sum()
    # out = test_embeds[sample_cat].sum()

    grad = torch.autograd.grad(out, test_logit_params)

    print(grad)
