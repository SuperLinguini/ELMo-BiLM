# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import collections
import time
import math
import mxnet as mx
from mxnet import gluon, autograd, init, nd
from mxnet.gluon import nn, Block
from base import get_rnn_cell
# from mxnet.gluon import data, text

import gluonnlp as nlp
# from gluonnlp.models.language_model import StandardRNN, AWDRNN

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=750,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_h', type=float, default=0.3,
                    help='dropout applied to hidden layer (0 = no dropout)')
parser.add_argument('--dropout_i', type=float, default=0.65,
                    help='dropout applied to input layer (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.5,
                    help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--gctype', type=str, default='none',
                    help='type of gradient compression to use, \
                          takes `2bit` or `none` for now.')
parser.add_argument('--gcthreshold', type=float, default=0.5,
                    help='threshold for 2bit gradient compression')
parser.add_argument('--eval_only', action='store_true',
                    help='Whether to only evaluate the trained model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. (the result of multi-gpu training might be slightly different compared to single-gpu training, still need to be finalized)')
args = parser.parse_args()




class ElmoLSTM(gluon.Block):
    def __init__(self, mode, num_layers, input_size, hidden_size, dropout, weight_dropout, bidirectional=True):
        super(ElmoLSTM, self).__init__()

        self.num_layers = num_layers

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size

        with self.name_scope():
            for layer_index in range(num_layers):
                forward_layer = get_rnn_cell(mode, 1, lstm_input_size, hidden_size, dropout, weight_dropout)
                backward_layer = get_rnn_cell(mode, 1, lstm_input_size, hidden_size, dropout, weight_dropout)

                self.register_child(forward_layer)
                self.register_child(backward_layer)

                forward_layers.append(forward_layer)
                backward_layers.append(backward_layer)

                lstm_input_size = hidden_size
            self.forward_layers = forward_layers
            self.backward_layers = backward_layers

    def begin_state(self, *args, **kwargs):
        return [forward_layer.begin_state(*args, **kwargs) for forward_layer in self.forward_layers], [backward_layer.begin_state(*args, **kwargs) for backward_layer in self.backward_layers]

    def forward(self, inputs, states_forward=None, states_backward=None):
        if not (states_forward and states_backward):
            states_forward, states_backward = self.begin_state(batch_size=inputs.shape[1])

        outputs_forward = []
        out_states_forward = []
        outputs_backward = []
        out_states_backward = []

        for j in range(self.num_layers):
            outputs_forward.append([])
            for i in range(inputs.shape[0]):
                if j == 0:
                    output, states_forward[j] = self.forward_layers[j](inputs[i], states_forward[j])
                else:
                    output, states_forward[j] = self.forward_layers[j](outputs_forward[j-1][i], states_forward[j])
                outputs_forward[j].append(output)
            out_states_forward.append(states_forward[j])

            outputs_backward.append([None] * inputs.shape[0])
            for i in reversed(range(inputs.shape[0])):
                if j == 0:
                    output, states_backward[j] = self.backward_layers[j](inputs[i], states_backward[j])
                else:
                    output, states_backward[j] = self.backward_layers[j](outputs_backward[j-1][i], states_backward[j])
                outputs_backward[j][i] = output
            out_states_backward.append(states_backward[j])

        for i in range(self.num_layers):
            outputs_forward[i] = mx.nd.stack(*outputs_forward[i])
            outputs_backward[i] = mx.nd.stack(*outputs_backward[i])

        return outputs_forward, out_states_forward, outputs_backward, out_states_backward


class StandardRNN(Block):
    """Standard RNN language model.

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout rate to use for encoder output.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    """
    def __init__(self, mode, vocab_size, embed_size, hidden_size,
                 num_layers, tie_weights=False, dropout=0.5, **kwargs):
        if tie_weights:
            assert embed_size == hidden_size, "Embedding dimension must be equal to " \
                                              "hidden dimension in order to tie weights. " \
                                              "Got: emb: {}, hid: {}.".format(embed_size,
                                                                              hidden_size)
        super(StandardRNN, self).__init__(**kwargs)
        self._mode = mode
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        self._vocab_size = vocab_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(nn.Embedding(self._vocab_size, self._embed_size,
                                       weight_initializer=init.Uniform(0.1)))
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout))
        return embedding

    def _get_encoder(self):
        return ElmoLSTM(self._mode, self._num_layers, self._embed_size,
                              self._hidden_size, self._dropout, 0)

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            output.add(nn.Dropout(self._dropout))
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, states_forward=None, states_backward=None): # pylint: disable=arguments-differ
        embedded_inputs = self.embedding(inputs)
        if not (states_forward and states_backward):
            states_forward, states_backward = self.begin_state(batch_size=inputs.shape[1])
        outputs_forward, out_states_forward, outputs_backward, out_states_backward = self.encoder(embedded_inputs, states_forward, states_backward)

        # out2 = mx.nd.empty((len(outputs_forward[-1]), outputs_forward[-1][0].shape[0], self._vocab_size))
        # out = []
        # for i in range(len(outputs_forward[-1])):
        #     # out[i, :, :] = self.decoder(mx.nd.concat(outputs_forward[-1][0], outputs_backward[-1][0], dim=1))
        #     out.append(self.decoder(mx.nd.concat(outputs_forward[-1][0], outputs_backward[-1][0], dim=1)))
        # out2 = mx.nd.stack(*out)
        out = self.decoder(mx.nd.concat(outputs_forward[-1], outputs_backward[-1], dim=2))
        return out, states_forward, states_backward


###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == "" else \
          [mx.gpu(int(i)) for i in args.gpus.split(',')]

args.batch_size *= len(context)

dataset_name = 'wikitext-2'
train_dataset, val_dataset, test_dataset = [nlp.data.WikiText2(segment=segment,
                                                               bos=None, eos='<eos>',
                                                               skip_empty=False)
                                            for segment in ['train', 'val', 'test']]

vocab = nlp.Vocab(nlp.data.Counter(train_dataset[0]), padding_token=None, bos_token=None)

train_data, val_data, test_data = [x.bptt_batchify(vocab, args.bptt, args.batch_size,
                                                   last_batch='keep')
                                   for x in [train_dataset, val_dataset, test_dataset]]


###############################################################################
# Build the model
###############################################################################

if args.weight_dropout:
    model = AWDRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                   args.tied, args.dropout, args.weight_dropout, args.dropout_h, args.dropout_i)
else:
    model = StandardRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                      args.tied, args.dropout)

model.initialize(mx.init.Xavier(), ctx=context)


compression_params = None if args.gctype == 'none' else {'type': args.gctype, 'threshold': args.gcthreshold}
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args.lr,
                         'momentum': 0,
                         'wd': 0},
                        compression_params=compression_params)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def detach(hidden):
    if isinstance(hidden, list):
        hidden = [detach(i) for i in hidden]
    elif isinstance(hidden, tuple):
        hidden = tuple(detach(i) for i in hidden)
    else:
        hidden = hidden.detach()
    return hidden

def evaluate(model, data_source, ctx):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(batch_size=args.batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output_tuple = model(data, *hidden)
        output = output_tuple[0]
        hidden = output_tuple[1:]
        L = loss(mx.nd.reshape(output, (-3, -1)), mx.nd.reshape(target, (-1,)))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def get_ppl(cur_loss):
    try:
        ppl = math.exp(cur_loss)
    except:
        ppl = float('inf')
    return ppl

def train():
    best_val = float("Inf")
    start_train_time = time.time()
    parameters = model.collect_params().values()
    for epoch in range(args.epochs):
        total_L, n_total = 0.0, 0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()

        hiddens = [model.begin_state(batch_size=args.batch_size//len(context), func=mx.nd.zeros, ctx=ctx) for ctx in context]
        for i, (data, target) in enumerate(train_data):
            data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
            target_list = gluon.utils.split_and_load(target, context, batch_axis=1, even_split=True)
            hiddens = detach(hiddens)
            L = 0
            Ls = []
            with autograd.record():
                for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                    output_tuple = model(X, *h)
                    output = output_tuple[0]
                    h = output_tuple[1:]
                    batch_L = loss(mx.nd.reshape(output, (-3, -1)), mx.nd.reshape(y, (-1,)))
                    L = L + batch_L.as_in_context(context[0]) / X.size
                    Ls.append(batch_L)
                    hiddens[j] = h

            L.backward()
            grads = [p.grad(x.context) for p in parameters for x in data_list]
            gluon.utils.clip_global_norm(grads, args.clip)

            trainer.step(1)

            total_L += sum([mx.nd.sum(l).asscalar() for l in Ls])
            n_total += data.size

            if i % args.log_interval == 0 and i > 0:
                cur_L = total_L / n_total
                ppl = get_ppl(cur_L)
                print('[Epoch %d Batch %d/%d] loss %.2f, ppl %.2f, throughput %.2f samples/s' % (
                    epoch, i, len(train_data), cur_L, ppl, args.batch_size * args.log_interval / (time.time() - start_log_interval_time)))
                total_L, n_total = 0.0, 0
                start_log_interval_time = time.time()

        mx.nd.waitall()

        print('[Epoch %d] throughput %.2f samples/s' % (
            epoch, (args.batch_size * len(train_data)) / (time.time() - start_epoch_time)))
        val_L = evaluate(model, val_data, context[0])
        ppl = get_ppl(val_L)
        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f' % (
            epoch, time.time() - start_epoch_time, val_L, ppl))

        if val_L < best_val:
            best_val = val_L
            test_L = evaluate(model, test_data, context[0])
            model.collect_params().save(args.save)
            ppl = get_ppl(test_L)
            print('test loss %.2f, test ppl %.2f' % (test_L, ppl))
        else:
            args.lr = args.lr * 0.25
            print('Learning rate now %f' % (args.lr))
            trainer.set_learning_rate(args.lr)

    print('Total training throughput %.2f samples/s' % (
            (args.batch_size * len(train_data) * args.epochs) / (time.time() - start_train_time)))

if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
    model.collect_params().load(args.save, context)
    val_L = evaluate(model, val_data, context[0])
    test_L = evaluate(model, test_data, context[0])
    print('Best validation loss %.2f, test ppl %.2f' % (val_L, math.exp(val_L)))
    print('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))
    print('Total time cost %.2fs' % (time.time() - start_pipeline_time))