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
import time
import math
import mxnet as mx
from mxnet import gluon, autograd, init, nd
from mxnet.gluon import nn

from data import UnicodeCharsVocabulary, WikiText2Character
from model import ElmoBiLM, elmo_options as options

import gluonnlp as nlp
# from gluonnlp.model import StandardRNN, AWDRNN

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--lr_update_interval', type=int, default=30,
                    help='lr udpate interval')
parser.add_argument('--lr_update_factor', type=float, default=0.1,
                    help='lr udpate factor')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_h', type=float, default=0.2,
                    help='dropout applied to hidden layer (0 = no dropout)')
parser.add_argument('--dropout_i', type=float, default=0.65,
                    help='dropout applied to input layer (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0,
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
parser.add_argument('--char_embedding', action='store_true', help='Whether to use character embeddings or word embeddings')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(i)) for i in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

if args.char_embedding:
    max_word_length = 50
    train_dataset, val_dataset, test_dataset = [WikiText2Character(segment=segment,
                                                                   bos='<bos>', eos='<eos>',
                                                                   skip_empty=False)
                                                for segment in ['train', 'val', 'test']]
    vocab = UnicodeCharsVocabulary(nlp.data.Counter(train_dataset[0]), max_word_length)

    train_data, val_data, test_data = [x.batchify(vocab, args.batch_size, max_word_length, load='train_data' if x is train_dataset else None)
        for x in [train_dataset, val_dataset, test_dataset]]
else:
    train_dataset, val_dataset, test_dataset = [nlp.data.WikiText2(segment=segment,
                                                                   bos=None, eos='<eos>',
                                                                   skip_empty=False)
                                                for segment in ['train', 'val', 'test']]

    vocab = nlp.Vocab(nlp.data.Counter(train_dataset[0]), padding_token=None, bos_token=None)

    train_data, val_data, test_data = [x.batchify(vocab, args.batch_size)
        for x in [train_dataset, val_dataset, test_dataset]]

def get_batch_char_embedding(data_source, index, seq_len=None):
    i = index + 1
    seq_len = min(seq_len if seq_len else 35, len(data_source[0]) - 1 - i)
    data = data_source[0][i:i+seq_len]
    forward_target = data_source[1][i+1:i+1+seq_len]
    backward_target = data_source[1][i-1:i-1+seq_len]
    return data, (forward_target, backward_target)

def get_batch_word_embedding(data_source, i, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(data_source) - 1 - i)
    data = data_source[i:i + seq_len]
    target = data_source[i + 1:i + 1 + seq_len]
    return data, target

###############################################################################
# Build the model
###############################################################################


model = ElmoBiLM(args.model, len(vocab), args.emsize,
                 args.nhid, args.nlayers, args.tied, args.dropout)

model.initialize(init.Xavier(), ctx=context)
model.hybridize()

if args.char_embedding:
    model.set_highway_bias()


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
    for i in range(0, len(data_source[0]) - 1, args.bptt):
        if args.char_embedding:
            data, target = get_batch_char_embedding(data_source, i)
            data = data.as_in_context(ctx)
            target = target[0].as_in_context(ctx), target[1].as_in_context(ctx)
            output, h = model(data, *hidden)
        else:
            data, target = get_batch_word_embedding(data_source, i)
            data = data.as_in_context(ctx)
            target = target.as_in_context(ctx)
            output, h = model((data, target), *hidden)

        L = loss(mx.nd.reshape(output[0], (-3, -1)), mx.nd.reshape(target[0] if args.char_embedding else target, (-1,)))
        total_L += mx.nd.sum(L).asscalar()

        L = loss(mx.nd.reshape(output[1], (-3, -1)), mx.nd.reshape(target[1] if args.char_embedding else data, (-1,)))
        total_L += mx.nd.sum(L).asscalar()

        ntotal += 2 * L.size
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
        start_epoch_time = time.time()

        if args.char_embedding:
            train_char(epoch, parameters)
        else:
            train_word(epoch, parameters)

        mx.nd.waitall()

        print('[Epoch %d] throughput %.2f samples/s' % (
            epoch, (args.batch_size * len(train_data[0])) / (time.time() - start_epoch_time)))
        val_L = evaluate(model, val_data, context[0])
        ppl = get_ppl(val_L)
        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f' % (
            epoch, time.time() - start_epoch_time, val_L, ppl))

        if val_L < best_val:
            update_lr_epoch = 0
            best_val = val_L
            test_L = evaluate(model, test_data, context[0])
            model.collect_params().save(args.save)
            ppl = get_ppl(test_L)
            print('test loss %.2f, test ppl %.2f' % (test_L, ppl))
        else:
            update_lr_epoch += 1
            if update_lr_epoch % args.lr_update_interval == 0 and update_lr_epoch != 0:
                lr_scale = trainer.learning_rate * args.lr_update_factor
                print('Learning rate after interval update %f'%(lr_scale))
                trainer.set_learning_rate(lr_scale)
                update_lr_epoch = 0

    print('Total training throughput %.2f samples/s' % (
            (args.batch_size * len(train_data[0]) * args.epochs) / (time.time() - start_train_time)))

def train_char(epoch, parameters):
    total_L = 0.0
    start_log_interval_time = time.time()
    hiddens = [model.begin_state(batch_size=args.batch_size//len(context), func=mx.nd.zeros, ctx=ctx) for ctx in context]

    batch_i, i = 0, 0
    while i < len(train_data[0]) - 1 - 1:
        lr_batch_start = trainer.learning_rate
        trainer.set_learning_rate(lr_batch_start)
        data, target = get_batch_char_embedding(train_data, i, seq_len=args.bptt)

        data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
        forward_target_list = gluon.utils.split_and_load(target[0], context, batch_axis=1, even_split=True)
        backward_target_list = gluon.utils.split_and_load(target[1], context, batch_axis=1, even_split=True)
        hiddens = detach(hiddens)

        L = 0
        Ls = []
        with autograd.record():
            for j, (X, y_forward, y_backward, h) in enumerate(
                    zip(data_list, forward_target_list, backward_target_list, hiddens)):
                output, h = model(X, *h)
                batch_L = loss(mx.nd.reshape(output[0], (-3, -1)), mx.nd.reshape(y_forward, (-1,)))
                L = L + batch_L.as_in_context(context[0]) / y_forward.size
                Ls.append(batch_L / y_forward.size)

                batch_L = loss(mx.nd.reshape(output[1], (-3, -1)), mx.nd.reshape(y_backward, (-1,)))
                L = L + batch_L.as_in_context(context[0]) / y_backward.size
                Ls.append(batch_L / y_backward.size)
                hiddens[j] = h

        L.backward()
        grads = [p.grad(x.context) for p in parameters for x in data_list]
        gluon.utils.clip_global_norm(grads, args.clip)

        trainer.step(1)

        total_L += sum([mx.nd.sum(l).asscalar() for l in Ls])
        trainer.set_learning_rate(lr_batch_start)

        if batch_i % args.log_interval == 0 and batch_i > 0:
            cur_L = total_L / (args.log_interval * 2)
            print('[Epoch %d Batch %d/%d] loss %.2f, ppl %.2f, throughput %.2f samples/s, lr %.2f' % (
                epoch, batch_i, len(train_data[0]) // args.bptt, cur_L, math.exp(cur_L),
                args.batch_size * args.log_interval / (time.time() - start_log_interval_time), args.lr))
            total_L = 0.0
            start_log_interval_time = time.time()
        i += args.bptt
        batch_i += 1

def train_word(epoch, parameters):
    total_L = 0.0
    start_log_interval_time = time.time()
    hiddens = [model.begin_state(batch_size=args.batch_size//len(context), func=mx.nd.zeros, ctx=ctx) for ctx in context]

    batch_i, i = 0, 0
    while i < len(train_data) - 1 - 1:
        lr_batch_start = trainer.learning_rate
        trainer.set_learning_rate(lr_batch_start)
        data, target = get_batch_word_embedding(train_data, i, seq_len=args.bptt)

        data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
        target_list = gluon.utils.split_and_load(target, context, batch_axis=1, even_split=True)
        hiddens = detach(hiddens)

        L = 0
        Ls = []
        with autograd.record():
            for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                output, h = model((X, y), *h)
                batch_L = loss(mx.nd.reshape(output[0], (-3, -1)), mx.nd.reshape(y, (-1,)))
                L = L + batch_L.as_in_context(context[0]) / X.size
                Ls.append(batch_L / X.size)

                batch_L = loss(mx.nd.reshape(output[1], (-3, -1)), mx.nd.reshape(X, (-1,)))
                L = L + batch_L.as_in_context(context[0]) / X.size
                Ls.append(batch_L / X.size)
                hiddens[j] = h

        L.backward()
        grads = [p.grad(x.context) for p in parameters for x in data_list]
        gluon.utils.clip_global_norm(grads, args.clip)

        trainer.step(1)

        total_L += sum([mx.nd.sum(l).asscalar() for l in Ls])
        trainer.set_learning_rate(lr_batch_start)

        if batch_i % args.log_interval == 0 and batch_i > 0:
            cur_L = total_L / (args.log_interval * 2)
            print('[Epoch %d Batch %d/%d] loss %.2f, ppl %.2f, throughput %.2f samples/s, lr %.2f' % (
                epoch, batch_i, len(train_data) // args.bptt, cur_L, math.exp(cur_L),
                args.batch_size * args.log_interval / (time.time() - start_log_interval_time), args.lr))
            total_L = 0.0
            start_log_interval_time = time.time()
        i += args.bptt
        batch_i += 1

if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
    model.collect_params().load(args.save, context)
    val_L = evaluate(model, val_data, context[0])
    test_L = evaluate(model, test_data, context[0])
    print('Best validation loss %.2f, val ppl %.2f' % (val_L, math.exp(val_L)))
    print('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))
    print('Total time cost %.2fs' % (time.time() - start_pipeline_time))