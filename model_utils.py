import tensorflow as tf

EOS_SYMBOL = 38
GO_SYMBOL = 37


def make_training_labels(labels, batch_size):
    """
    Used for inference during training.
    Creates a copy of the labels with a GO symbol inserted at the beginning of each ICD-10 code and padding on the last
    dimension to include the additional EOS and GO symbols
    :param labels: the current batch's labels. Tensor of shape [batch_size, 4, 36]
    :param batch_size: the current batch's size
    :return: the labels padded with the GO_SYMBOL. Tensor pf shape [batch_size, 5, 38]
    """
    go_symbol = tf.one_hot([36] * batch_size, depth=38)
    go_symbol = tf.expand_dims(go_symbol, axis=1)

    train_labels = tf.pad(
        labels,
        paddings=[[0, 0], [0, 0], [0, 2]],
        mode='CONSTANT',
        constant_values=0
    )

    return tf.concat((go_symbol, train_labels), axis=1)


def from_rnn_encoder_to_decoder_states(encoder_states, params, mode):
    """
    takes in the encoder's LSTM states, and modifies their size if required to fit the decoder's state sizes
    :param encoder_states: the encoder's LSTM final states. A list of LSTMStateTuple
    :param mode: a tf.estimator.ModeKeys
    :param params: the encoder and decoder's hyperparameters. A dict with entries:
            - 'encoder': dict with entry 'units', list of layer sizes (int)
            - 'decoder': dict with entry 'units', list of layer sizes (int)
    :return: The decoder's LSTM initial states. A list of LSTMStateTuple
    """
    num_encoder_cell = len(params['encoder']['units'])
    num_decoder_cell = len(params['decoder']['units'])
    init = tf.contrib.layers.xavier_initializer(uniform=False)
    decoder_init_states = []

    for i in range(num_decoder_cell):

        if params['encoder']['units'][num_encoder_cell - num_decoder_cell + i] == params['decoder']['units'][i]:
            decoder_init_states.append(encoder_states[num_encoder_cell - num_decoder_cell + i])
        else:
            cell_state = tf.layers.dense(
                encoder_states[num_encoder_cell - num_decoder_cell + i][0],
                units=params['decoder']['units'][i],
                kernel_initializer=init,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                )
            )

            cell_out = tf.zeros(
                [
                    encoder_states[num_encoder_cell - num_decoder_cell + i][0].get_shape()[0],
                    params['decoder']['units'][i]
                ],
                dtype=tf.float32
            )

            decoder_init_states.append(tf.nn.rnn_cell.LSTMStateTuple(cell_state, cell_out))

    return tuple(decoder_init_states)


def from_tcn_encoder_to_rnn_decoder_states(encoder_state, params, mode):
    """
    takes in the encoder's state, and converts it into a tuple of LSTMStateTuple to feed the decoder's RNN
    :param encoder_state: the encoder's final state. A Tensor of size [batch_size, dim_state]
    :param mode: a tf.estimator.ModeKeys
    :param params: the decoder's hyperparameters. A dict with entry:
            - 'decoder': dict with entry 'units', list of layer sizes (int)
    :return: The decoder's LSTM initial states. A list of LSTMStateTuple
        """
    num_decoder_cell = len(params['decoder']['units'])
    init = tf.contrib.layers.xavier_initializer(uniform=False)
    decoder_init_states = []

    for i in range(num_decoder_cell):

        if type(encoder_state) == list:
            cell_state = []

            for j in range(len(encoder_state)):

                cell_state.append(tf.layers.dense(
                    encoder_state[j],
                    units=params['decoder']['units'][i],
                    kernel_initializer=init,
                    activity_regularizer=lambda x: tf.layers.batch_normalization(
                        inputs=x,
                        training=mode == tf.estimator.ModeKeys.TRAIN
                    )
                ))

            cell_state = tf.add_n(cell_state)

        else:
            cell_state = tf.layers.dense(
                encoder_state,
                units=params['decoder']['units'][i],
                kernel_initializer=init,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                )
            )

        cell_out = tf.zeros(
            [cell_state.get_shape()[0], params['decoder']['units'][i]],
            dtype=tf.float32
        )

        decoder_init_states.append(tf.nn.rnn_cell.LSTMStateTuple(cell_state, cell_out))

    return tuple(decoder_init_states)


def res_block(features, params, init, mode, name=None):
    """
    :param features: the input features to the residual block
    :param params: the block's hyperparameters. A dictionary with entries:
                    - 'units' a list of int defining the block layers' number of neurons
                    - 'drop_out' an int defining the dropout rate for every layer in the current block
    :param init: A neural layer initializer (typically tf.contrib.xavier_initializer)
    :param mode: A tf.estimator.ModeKeys
    :param name: A string defining the block's name
    :return: The residual block's output. A tensor of same rank as input, whose shape is identical except for the last
    dimension where it is defined as the last 'units' parameter
    """
    out = features
    input_rank = len(out.get_shape())

    with tf.name_scope('dense_block'):

        for i in range(len(params['units'])):
            noise_shape = [tf.constant(1)] * (input_rank - 1)
            noise_shape.append(tf.constant(params['units'][i]))

            out = tf.layers.dense(
                out,
                units=params['units'][i],
                activation=tf.nn.leaky_relu,
                kernel_initializer=init,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                ),
                name=name + "_dense_" + str(i)
            )

            out = tf.layers.dropout(
                out,
                rate=params['drop_out'],
                noise_shape=noise_shape,
                training=mode == tf.estimator.ModeKeys.TRAIN
            )

    with tf.name_scope('residuals'):
        if params['units'][-1] == features.get_shape()[-1]:
            residuals = features
        else:
            residuals = tf.layers.dense(
                features,
                units=params['units'][-1],
                kernel_initializer=init,
                name=name + '_residuals'
            )

    return out + residuals


def causal_conv1d(inputs,
                  filters,
                  kernel_size,
                  dilation_rate,
                  activation=None,
                  kernel_initializer=None,
                  activity_regularizer=None,
                  name=None):
    """
    A dilated 1d convolutional layer padded to enable causal convolutions.
    :param inputs: the input tensor of shape [batch_size, dim_1, num_channels] or
            [batch_size, dim_1, dim_2, num_channels]. Either way, the convolution operation will be performed on axis
            dim_1. In case of rank 4 tensors, the operation performs 1d convolutions on tensors as if they were stacked
            along dim_2 axis.
    :param filters: An int denoting the number of filters for the conv operation
    :param kernel_size: An int denoting the kernel size for the conv operation
    :param dilation_rate: An int denoting the dilation rate for the conv operation
    :param activation: the network's activation function (applied post normalization)
    :param kernel_initializer: An initializer for the convolution kernel
    :param activity_regularizer: An activity regularizer layer (typically a batch_norm layer)
    :param name: A string, the name of the layer
    :return:
    """
    input_rank = len(inputs.get_shape())

    # depending on input rank, pads the input for causal convolution, chooses the right convolution op and adapts the
    # kernel size to the required format (both op remain 1d conv. The 2d one just executes it on a stack of tensor
    # in parallel
    if input_rank == 4:
        padding = (kernel_size - 1) * dilation_rate
        kernel_size = [kernel_size, 1]
        conv_fn = tf.layers.conv2d
        inputs = tf.pad(inputs, tf.constant([(0, 0), (1, 0), (0, 0), (0, 0)]) * padding)
    else:
        conv_fn = tf.layers.conv1d
        padding = (kernel_size - 1) * dilation_rate
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)

    out = conv_fn(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        activation=activation,
        kernel_initializer=kernel_initializer,
        activity_regularizer=activity_regularizer,
        name=name
    )

    return out


def temporal_block(inputs,
                   params,
                   dilation_rate,
                   kernel_initializer,
                   mode,
                   name=None):
    """
    Implements the temporal block as defined in (Bai et Al.)
    :param inputs: the input tensor of shape [batch_size, dim_1, num_channels] or
            [batch_size, dim_1, dim_2, num_channels]. Either way, the convolution operation will be performed on axis
            dim_1. In case of rank 4 tensors, the operation performs 1d convolutions on tensors as if they were stacked
            along dim_2 axis.
    :param params: a dict of parameters with following entries:
                - 'filters': A list int denoting the number of filters for each conv operation
                - 'kernel': A list of int denoting the kernel size for each conv operation
                - 'drop_out': The drop out probability for each layer
    :param dilation_rate: An int denoting the dilation rate for the conv operation
    :param kernel_initializer: An initializer for each convolution kernel
    :param mode: a tf.estimator.ModeKeys
    :param name: A string, the name of the layer
    :return:
    """
    input_rank = len(inputs.get_shape())

    # depending on input rank, pads the input for causal convolution, chooses the right convolution op and adapts the
    # kernel size to the required format (both op remain 1d conv. The 2d one just executes it on a stack of tensor
    # in parallel
    if input_rank == 4:
        conv_fn = tf.layers.conv2d
    else:
        conv_fn = tf.layers.conv1d

    out = inputs
    with tf.name_scope('dilated_convolutions'):
        for i in range(len(params['filters'])):
            out = causal_conv1d(
                out,
                filters=params['filters'][i],
                kernel_size=params['kernel'][i],
                dilation_rate=dilation_rate,
                activation=tf.nn.leaky_relu,
                kernel_initializer=kernel_initializer,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                ),
                name=name + '_conv_' + str(i)
            )

            noise_shape = [tf.constant(1)] * (input_rank-1)
            noise_shape.append(tf.constant(params['filters'][i]))

            out = tf.layers.dropout(
                out,
                rate=params['drop_out'],
                noise_shape=noise_shape,
                training=mode == tf.estimator.ModeKeys.TRAIN
            )

    with tf.name_scope('residuals'):
        if params['filters'][-1] == inputs.get_shape()[-1]:
            residuals = inputs
        else:
            residuals = conv_fn(
                inputs,
                filters=params['filters'][-1],
                kernel_size=1,
                kernel_initializer=kernel_initializer,
                name=name + '_residuals'
            )

    return out + residuals


def entry_net(features, mode, params, key):
    out = features
    init = tf.contrib.layers.xavier_initializer(uniform=False)

    with tf.name_scope(key + '_network'):
        for i in range(len(params)):
            out = res_block(
                out,
                params=params[i],
                init=init,
                mode=mode,
                name=key + '_dense_block_' + str(i)
            )

    with tf.name_scope(key + '_input_residual'):
        residual = tf.layers.dense(
            out,
            units=params[-1]['units'][0],
            kernel_initializer=init,
            activity_regularizer=lambda x: tf.layers.batch_normalization(
                x,
                training=mode == tf.estimator.ModeKeys.TRAIN
            ),
            name=key + '_residuals'
        )

    return out + residual


def date_causal_net(date, params, mode):
    """
    Processes each date code individually to convert it into a rich, dense representation using
    dilated causal convolutions
    :param date: the stacked dates represented by a dense tensor of shape [batch_size, 2, seq_length, 1]
    :param mode: a tf.estimator.ModeKeys
    :param params: the model's hyperparameters represented as a dict with entries:
                        - 'block': a list of dictionaries containing parameters for each residual block
                        - 'units': the network output's dimensionality for each ICD-10 code
                        - 'drop_out': the dropout probability for the sequence conversion layer
    :return: A sequence of output vectors of shape [batch_size, sequence_length, params['units']]
    """
    out = date

    with tf.name_scope('date_network'):
        init = tf.contrib.layers.xavier_initializer(uniform=False)

        with tf.name_scope('temporal_convolution_stage'):
            for i in range(len(params['block'])):
                out = temporal_block(
                    out,
                    params['block'][i],
                    dilation_rate=2 ** i,
                    kernel_initializer=init,
                    mode=mode,
                    name='date_net_temporal_block_' + str(i)
                )

            # skip connection from the inputs to the conv stage output
            out += tf.layers.conv2d(
                inputs=date,
                filters=params['block'][-1]['filters'][-1],
                kernel_size=1,
                padding='same',
                activation=None,
                kernel_initializer=init,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                )
            )

        with tf.name_scope('date_sequence_conversion'):
            out = tf.layers.conv2d(
                inputs=out,
                filters=params['units'],
                kernel_size=(out.get_shape()[1], 1),
                padding='valid',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                ),
                name='date_net_out_unsqueezed'
            )
            out = tf.layers.dropout(
                inputs=out,
                rate=params['drop_out'],
                training=mode == tf.estimator.ModeKeys.TRAIN
            )

    out = tf.squeeze(out, axis=1, name='date_net_out')

    return out


def cause_causal_net(cause, params, mode):
    """
    Processes each ICD-10 code individually to convert it into a rich, dense representation using
    dilated causal convolutions
    :param cause: the stacked ICD-10 codes represented by a one-hot tensor of shape [batch_size, 4, seq_length, 36]
    :param mode: a tf.estimator.ModeKeys
    :param params: the model's hyperparameters represented as a dict with entries:
                        - 'block': a list of dictionaries containing parameters for each residual block
                        - 'units': the network output's dimensionality for each ICD-10 code
                        - 'drop_out': the dropout probability for the sequence conversion layer
    :return: A sequence of output vectors of shape [batch_size, sequence_length, params['units']]
    """
    out = cause

    with tf.name_scope('cause_network'):
        init = tf.contrib.layers.xavier_initializer(uniform=False)

        with tf.name_scope('temporal_convolution_stage'):
            for i in range(len(params['block'])):
                out = temporal_block(
                    out,
                    params['block'][i],
                    dilation_rate=2 ** i,
                    kernel_initializer=init,
                    mode=mode,
                    name='cause_net_temporal_block_' + str(i)
                )

            # skip connection from the inputs to the conv stage output
            out += tf.layers.conv2d(
                inputs=cause,
                filters=params['block'][-1]['filters'][-1],
                kernel_size=1,
                padding='same',
                activation=None,
                kernel_initializer=init,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                )
            )

        with tf.name_scope('causal_sequence_conversion'):
            out = tf.layers.conv2d(
                inputs=out,
                filters=params['units'],
                kernel_size=(4, 1),
                padding='valid',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init,
                activity_regularizer=lambda x: tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                ),
                name='causal_net_out_unsqueezed'
            )
            out = tf.layers.dropout(
                inputs=out,
                rate=params['drop_out'],
                training=mode == tf.estimator.ModeKeys.TRAIN
            )

    out = tf.squeeze(out, axis=1, name='causal_net_out')

    return out


def non_cause_net(inputs, params, mode):
    out = inputs
    init = tf.contrib.layers.xavier_initializer(uniform=False)

    with tf.name_scope('standardization'):

        for i in range(len(inputs)):
            out[i] = tf.layers.dense(
                out[i],
                units=params['units'],
                activation=tf.nn.leaky_relu,
                kernel_initializer=init,
                name='standardization_layer_' + str(i)
            )

            if len(out[i].get_shape()) > 2:
                out[i] = tf.reduce_sum(
                    out[i],
                    axis=1
                )

    with tf.name_scope('additive_mix'):
        out = tf.add_n(
            out,
            name='standardized_out'
        )
        out = tf.layers.batch_normalization(
            out,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        out = tf.layers.dropout(
            out,
            rate=params['drop_out'],
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

    with tf.name_scope('non_cause_residual_network'):

        for i in range(len(params['block'])):
            out = res_block(
                out,
                params=params['block'][i],
                init=init,
                mode=mode,
                name='state_network_residual_block' + str(i)
            )

    return out


def basic_encoder(encoder_in, sequence_length, params, mode):
    """
    Takes in the dense representation of the ICD-10 codes and feeds it to a multilayer LSTM RNN to be used as the
    encoder of a seq2seq model
    :param encoder_in: the output of the cause network. A Tensor of shape [batch_size, max_seq_length, dim_cause]
    :param sequence_length: a rank one vector containing the length of each causal chain present in the batch
    :param params: the model's hyperparameters represented as a dict with entries:
                    - 'units': a list of int defining each LSTM layer's size
                    - 'drop_out': a float to be used as drop out keep prob parameter during training. The same value is
                      used for every layer for simplicity
    :param mode: a tf.estimator.ModeKeys
    :return: - the encoded causal sequence, a Tensor of shape [batch_size, max_seq_length, params['units'][-1]
             - the LSTM cells final states, a list of LSTMStateTuple
    """
    with tf.name_scope('encoder_cell'):
        encoder_cell = []

        for i in range(len(params['units'])):

            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
                params['units'][i],
            )
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=params['drop_out'] ** (mode == tf.estimator.ModeKeys.TRAIN),
            )
            encoder_cell.append(cell)

        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cell)

    outputs, state = tf.nn.dynamic_rnn(
        encoder_cell,
        encoder_in,
        sequence_length=sequence_length,
        dtype=tf.float32,
    )
    return outputs, state


def basic_decoder(encoder_out, encoder_state, sequence_length, labels, params, mode):
    """
    Basic decoder for seq2seq models.
    :param encoder_out: the encoder's output
    :param encoder_state: the encoder's output state (LSTMTuple)
    :param sequence_length: the encoder outputs' lengths. a tensor of shape [batch_size]
    :param labels: the output's ground truth (used by decoder helper in training mode)
    :param params: a dict with entries:
                    - 'units' a list of int defining each LSTM layer's size
                    - 'drop_out': a float to be used as drop out keep prob parameter during training. The same value is
                      used for every layer for simplicity
                    - 'bahdanau' an int that defines the attention mechanism's size
    :param mode: a tf.estimator.ModeKeys
    :return: the model's ICD-10 predictions
    """

    batch_size = labels.get_shape()[0]

    with tf.name_scope('decoder_cell'):
        decoder_cells = []
        for i in range(len(params['units'])):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
                params['units'][i]
            )
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=params['drop_out'] ** (mode == tf.estimator.ModeKeys.TRAIN),
            )
            decoder_cells.append(cell)

        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)

        train_labels = make_training_labels(labels, batch_size)

    if 'bahdanau' in list(params.keys()):

        with tf.name_scope('attention_cell'):

            attention_cell = tf.contrib.seq2seq.BahdanauAttention(
                num_units=params['bahdanau'],
                memory=encoder_out,
                memory_sequence_length=sequence_length,
                dtype=tf.float32
            )

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_cell,
                attention_layer_size=params['bahdanau']
            )

    with tf.name_scope('helper'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            helper = tf.contrib.seq2seq.TrainingHelper(train_labels, sequence_length=[5] * batch_size)

        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=lambda x: tf.one_hot(x, depth=38),
                start_tokens=tf.ones([batch_size], dtype=tf.int32) * GO_SYMBOL,
                end_token=EOS_SYMBOL
            )

    with tf.name_scope('decoder'):
        initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        initial_state = initial_state.clone(cell_state=encoder_state)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            helper,
            initial_state=initial_state,
            output_layer=tf.layers.Dense(
                units=38,
            )
        )

        decoder_out = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True,
            maximum_iterations=4,
        )

    return decoder_out[0][0]


def attention_beam_search_decoder(encoder_out, encoder_state, sequence_length, labels, params, mode):
    """
        Basic decoder for seq2seq models.
        :param encoder_out: the encoder's output
        :param encoder_state: the encoder's output state (LSTMTuple)
        :param sequence_length: the encoder outputs' lengths. a tensor of shape [batch_size]
        :param labels: the output's ground truth (used by decoder helper in training mode)
        :param params: a dict with entries:
                        - 'units' a list of int defining each LSTM layer's size
                        - 'drop_out': a float to be used as drop out keep prob parameter during training. The same value is
                          used for every layer for simplicity
                        - 'bahdanau' an int that defines the attention mechanism's size
        :param mode: a tf.estimator.ModeKeys
        :return: the model's ICD-10 predictions
        """

    batch_size = labels.get_shape()[0]

    with tf.name_scope('decoder_cell'):
        decoder_cells = []
        for i in range(len(params['units'])):
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
                params['units'][i]
            )
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                output_keep_prob=params['drop_out'] ** (mode == tf.estimator.ModeKeys.TRAIN),
            )
            decoder_cells.append(cell)

        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)

    train_labels = make_training_labels(labels, batch_size)

    if mode != tf.estimator.ModeKeys.TRAIN:

        encoder_out = tf.contrib.seq2seq.tile_batch(
            encoder_out,
            multiplier=params['beam_width']
        )

        encoder_state = tf.contrib.seq2seq.tile_batch(
            encoder_state,
            multiplier=params['beam_width']
        )

        sequence_length = tf.contrib.seq2seq.tile_batch(
            sequence_length,
            multiplier=params['beam_width']
        )

    if 'bahdanau' in list(params.keys()):
        with tf.name_scope('attention_cell'):
            attention_cell = tf.contrib.seq2seq.BahdanauAttention(
                num_units=params['bahdanau'],
                memory=encoder_out,
                memory_sequence_length=sequence_length,
                dtype=tf.float32
            )

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_cell,
                attention_layer_size=params['bahdanau']
            )

    with tf.name_scope('decoder'):

        if mode == tf.estimator.ModeKeys.TRAIN:
            helper = tf.contrib.seq2seq.TrainingHelper(train_labels, sequence_length=[5] * batch_size)

            initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            initial_state = initial_state.clone(cell_state=encoder_state)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell,
                helper,
                initial_state=initial_state,
                output_layer=tf.layers.Dense(
                    units=38,
                )
            )

            decoder_out = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=True,
                maximum_iterations=4,
            )

            decode_out = decoder_out[0][0]

        else:

            initial_state = decoder_cell.zero_state(
                batch_size=batch_size * params['beam_width'],
                dtype=tf.float32
            )
            initial_state = initial_state.clone(cell_state=encoder_state)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                decoder_cell,
                embedding=lambda x: tf.one_hot(x, depth=38),
                start_tokens=tf.ones([batch_size], dtype=tf.int32) * GO_SYMBOL,
                end_token=EOS_SYMBOL,
                initial_state=initial_state,
                beam_width=params['beam_width'],
                output_layer=tf.layers.Dense(
                    units=38,
                )
            )

            decoder_out = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=False,
                maximum_iterations=4,
            )

            decode_out = decoder_out[0]

    return decode_out



def tcn_encoder(encoder_in, sequence_length, params, mode):
    """
            Takes in the dense representation of the ICD-10 codes and feeds it to a multilayer LSTM RNN to be used as the
            encoder of a seq2seq model
            :param encoder_in: the output of the cause network. A Tensor of shape [batch_size, max_seq_length, dim_cause]
            :param sequence_length: a rank one vector containing the length of each causal chain present in the batch
            :param params: the model's hyperparameters (same for each individual block) represented as a dict with entries:
                            - 'kernel' list of int defining each layer's kernel size
                            - 'drop_out' the dropout probability of the blocks' layers
            :param mode: a tf.estimator.ModeKeys
            :return: - The encoded causal sequence, a Tensor of shape [batch_size, max_seq_length, params['units'][-1]
                     - An equivalent of an RNN state tensor (time axis reduced mean of the encoded causal sequence)
        """

    out = encoder_in
    init = tf.contrib.layers.xavier_initializer(uniform=False)

    # 5 res_blocks because log2(max_seq_len) = 4.6
    for i in range(5):
        out = temporal_block(
            out,
            params=params,
            dilation_rate=2 ** i,
            kernel_initializer=init,
            mode=mode,
            name='tcn_encoder_block_' + str(i)
        )

    sequence_mask = tf.one_hot(
        sequence_length,
        depth=tf.reduce_max(sequence_length),
        on_value=1.0,
        off_value=0.0
    )

    sequence_mask = tf.cumprod(
        1 - sequence_mask,
        axis=-1
    )
    out = out * tf.expand_dims(sequence_mask, axis=-1)

    print(out)

    state = tf.reduce_sum(out, axis=-2) / tf.expand_dims(tf.cast(sequence_length, tf.float32), axis=-1)

    return out, state
