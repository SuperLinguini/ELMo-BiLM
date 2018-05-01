import json
from mxnet import gluon, autograd, init, nd
from mxnet.ndarray.ndarray import NDArray
from mxnet.gluon import nn, Block
from overrides import overrides

class Highway(gluon.Block):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``nn.activations.Activation``, optional (default=``nn.Activation('relu')``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: nn.activations.Activation = nn.Activation('relu')) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim

        with self.name_scope():
            self._layers = []
            for i in range(num_layers):
                layer = nn.Dense(input_dim * 2, in_units=input_dim)
                self._layers.append(layer)
                setattr(self, 'layer_{}'.format(i), layer)

            self._activation = activation

    def set_bias(self):
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, to we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias.data()[self._input_dim:] = 1

    @overrides
    def forward(self, inputs: NDArray) -> NDArray:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = gate.sigmoid()
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input

class ElmoCharacterEncoder(gluon.Block):
    def __init__(self, options):
        super(ElmoCharacterEncoder, self).__init__()

        if isinstance(options, dict):
            self._options = options
        else:
            with open(options, 'r') as fin:
                self._options = json.load(fin)
        # self._weight_file = weight_file

        self.output_dim = self._options['lstm']['projection_dim']
        # self.requires_grad = requires_grad

        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        char_embed_dim = cnn_options['embedding']['dim']
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options['n_highway']

        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=cnn_options['n_characters'],
                                      output_dim=char_embed_dim)

            convolutions = []
            for i, (width, out_channels) in enumerate(filters):
                conv = nn.Conv1D(in_channels=char_embed_dim, channels=out_channels, kernel_size=width, use_bias=True)
                setattr(self, 'char_conv_{}'.format(i), conv)
            self.num_filters = len(filters)

            self.highways = Highway(n_filters, n_highway, activation=nn.Activation('relu'))

            self.projection = nn.Dense(in_units=n_filters, units=self.output_dim, use_bias=True)

    def set_highway_bias(self):
        self.highways.set_bias()

    def forward(self, inputs):
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']

        character_embedding = self.embedding(inputs.reshape(-1, max_chars_per_token))

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = nn.Activation('tanh')
        elif cnn_options['activation'] == 'relu':
            activation = nn.Activation('relu')
        else:
            raise ValueError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = nd.swapaxes(character_embedding, 1, 2)
        convs = []
        for i in range(self.num_filters):
            conv = getattr(self, 'char_conv_{}'.format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved = nd.max(convolved, axis=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = nd.concat(*convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self.highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self.projection(token_embedding)

        # reshape to (sequence_length, batch_size, embedding_dim)
        sequence_length, batch_size, _ = inputs.shape

        return token_embedding.reshape(sequence_length, batch_size, -1)

# cnn = ElmoCharacterEncoder(options, 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', True)
# cnn = ElmoCharacterEncoder(options)
# cnn.collect_params().initialize()
# cnn.set_highway_bias()
# train_data = nd.load('data')
# output = cnn(train_data)