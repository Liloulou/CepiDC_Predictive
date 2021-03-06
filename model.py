import tensorflow as tf
from tensorflow.contrib import feature_column
import my_input_fn as pipe
import model_utils
import custom_metrics

EOS_SYMBOL = 38
GO_SYMBOL = 37

date_keys = ['date_' + x for x in ['dc', 'nais']]
loc_keys = ['loc_' + x for x in ['dom', 'dc', 'nais']]
miscellaneous = ['sexe', 'activ', 'etatmat', 'lieudc', 'jvecus']
misc_bucket_sizes = {'sexe': 2, 'activ': 3, 'etatmat': 4, 'lieudc': 6}
cause_chain = ['c' + str(x) for x in range(1, 25)]


def make_input_layers(dataset, feature_columns):
    feature_dict = {}

    features, labels = dataset

    with tf.name_scope('date_inputs'):
        feature_dict['date'] = tf.feature_column.input_layer(features, feature_columns['date'])

    with tf.name_scope('loc_inputs'):
        feature_dict['loc'] = tf.reshape(
            feature_column.sequence_input_layer(features, feature_columns['loc'], trainable=False)[0],
            shape=[-1, 5, len(loc_keys), 36])

    with tf.name_scope('misc_inputs'):
        feature_dict['misc'] = tf.feature_column.input_layer(features, feature_columns['misc'], trainable=False)

    with tf.name_scope('cause_columns'):
        cause_input = []

        for cause_column in feature_columns['cause']:
            cause_input.append(feature_column.sequence_input_layer(features, cause_column, trainable=False)[0])

        feature_dict['cause'] = tf.stack(cause_input, axis=-2)
        feature_dict['cause_len'] = pipe.compute_len(feature_dict['cause'])
        feature_dict['cause'] = feature_dict['cause'][:, :4, :tf.reduce_max(feature_dict['cause_len'])]

    with tf.name_scope('labels_one_hot_encoding'):
        label_inputs = feature_column.sequence_input_layer(
            {'labels': labels}, feature_columns['labels'], trainable=False)[0]

    return feature_dict, label_inputs


def basic_rnn_model(features, labels, params, mode, reuse=None):

    with tf.variable_scope('inference_network', reuse=reuse):
        out = features['cause']
        batch_size = out.get_shape()[0]
        causal_chain_length = features['cause_len']

        with tf.name_scope('reshape_causal_chain'):
            out = tf.reshape(out, shape=[-1, 4 * tf.reduce_max(causal_chain_length), 36])

        with tf.name_scope('encoder'):
            encoder_cells = []

            for i in range(3):
                encoder_cells.append(tf.nn.rnn_cell.LSTMCell(200))

            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cells)

            outputs, state = tf.nn.dynamic_rnn(
                encoder_cell,
                out,
                sequence_length=causal_chain_length,
                dtype=tf.float32,
            )

            encoder_out = state[-2:]

        with tf.name_scope('decoder'):
            decoder_cells = []
            for i in range(2):
                decoder_cells.append(tf.nn.rnn_cell.LSTMCell(200))

            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)

            with tf.name_scope('make_training_labels'):
                go_symbol = tf.one_hot([36] * batch_size, depth=38)

                go_symbol = tf.expand_dims(go_symbol, axis=1)

                train_labels = tf.pad(
                    labels,
                    paddings=[[0, 0], [0, 0], [0, 2]],
                    mode='CONSTANT',
                    constant_values=0
                )

                train_labels = tf.concat((go_symbol, train_labels), axis=1)

            if mode == tf.estimator.ModeKeys.TRAIN:
                helper = tf.contrib.seq2seq.TrainingHelper(train_labels, sequence_length=[5] * batch_size)

            else:
                def embed_lookup(x):

                    return tf.one_hot(
                        x,
                        depth=38,
                    )

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embed_lookup,
                    start_tokens=tf.ones([batch_size], dtype=tf.int32) * GO_SYMBOL,
                    end_token=EOS_SYMBOL
                )

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell,
                helper,
                initial_state=encoder_out,
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


def tcn_attention_rnn_model(features, labels, params, mode):
    processed_codes_sequences = model_utils.causal_cause_net(
        cause=features['cause'],
        params=params['cause'],
        mode=mode
    )

    encoder_out, encoder_state = model_utils.tcn_encoder(
        encoder_in=processed_codes_sequences,
        sequence_length=features['cause_len'],
        params=params['encoder'],
        mode=mode
    )

    encoder_state = model_utils.from_tcn_encoder_to_rnn_decoder_states(
        encoder_state,
        params=params,
        mode=mode
    )

    logits = model_utils.basic_decoder(
        encoder_out=encoder_out,
        encoder_state=encoder_state,
        sequence_length=features['cause_len'],
        labels=labels,
        params=params['decoder'],
        mode=mode
    )

    return logits


def full_data_tcn_attention_rnn_model(features, labels, params, mode):

    with tf.name_scope('cause_network'):

        cause_out = model_utils.cause_causal_net(
            cause=features['cause'],
            params=params['cause'],
            mode=mode
        )

    with tf.name_scope('date_network'):

        date_out = model_utils.date_causal_net(
            date=features['date'],
            params=params['date'],
            mode=mode
        )

    with tf.name_scope('loc_network'):

        loc_out = model_utils.entry_net(
            features['loc'],
            params=params['loc'],
            mode=mode,
            key='loc'
        )

    with tf.name_scope('misc_network'):

        misc_out = model_utils.entry_net(
            features['misc'],
            params=params['misc'],
            mode=mode,
            key='misc'
        )

    with tf.name_scope('encoder'):
        encoder_out, encoder_state = model_utils.tcn_encoder(
            encoder_in=cause_out,
            sequence_length=features['cause_len'],
            params=params['encoder'],
            mode=mode
        )

    with tf.name_scope('decoder_state_network'):
        non_cause = model_utils.non_cause_net(
            [date_out, loc_out, misc_out],
            params=params['state'],
            mode=mode,
        )

        encoder_state = model_utils.from_tcn_encoder_to_rnn_decoder_states(
            [encoder_state, non_cause],
            params=params,
            mode=mode
        )

    with tf.name_scope('decoder'):
        logits = model_utils.basic_decoder(
            encoder_out=encoder_out,
            encoder_state=encoder_state,
            sequence_length=features['cause_len'],
            labels=labels,
            params=params['decoder'],
            mode=mode
        )

    return logits


def conv_rnn_attention_model(features, labels, params, mode):

    processed_codes_sequences = model_utils.cause_net(
        cause=features['cause'],
        params=params['cause'],
        mode=mode
    )

    encoded_out, encoder_state = model_utils.basic_encoder(
        encoder_in=processed_codes_sequences,
        sequence_length=features['cause_len'],
        params=params['encoder'],
        mode=mode
    )

    encoder_state = model_utils.from_rnn_encoder_to_decoder_states(
        encoder_states=encoder_state,
        params=params,
        mode=mode
    )

    logits = model_utils.basic_decoder(
        encoder_out=encoded_out,
        encoder_state=encoder_state,
        sequence_length=features['cause_len'],
        labels=labels,
        params=params['decoder'],
        mode=mode
    )

    return logits


def get_loss(logits, labels):
    weights = tf.ones(shape=tf.shape(labels), dtype=tf.float32)

    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        labels,
        weights
    )

    return loss


def get_train_op(loss, params):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        params['learning_rate'],
        global_step,
        params['decay_steps'],
        params['decay_rate'],
        staircase=False
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss, global_step=global_step)


def get_model_fn(features, labels, mode, params):

    features, labels = pipe.make_input_layers((features, labels), params['feature_columns'])
    labels_ind = tf.argmax(labels, axis=-1)

    logits = full_data_tcn_attention_rnn_model(features, labels, params['inference'], mode)
    predictions = tf.argmax(logits, axis=-1)
    loss = get_loss(logits, labels_ind)
    train_op = get_train_op(loss, params['train'])

    eval_metrics = {
        'char_accuracy': tf.metrics.accuracy(labels_ind, predictions),
        'cim_10_accuracy': custom_metrics.word_level_accuracy(labels_ind, predictions)
    }

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics
    )

def get_model_fn_beam_search(features, labels, mode, params):

    features, labels = pipe.make_input_layers((features, labels), params['feature_columns'])
    labels_ind = tf.argmax(labels, axis=-1)

    logits = full_data_tcn_attention_rnn_model(features, labels, params['inference'], mode)

    if mode == tf.estimator.ModeKeys.TRAIN:

        loss = get_loss(logits, labels_ind)
        train_op = get_train_op(loss, params['train'])

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
        )

    else:
        print(logits)
        predictions = logits.predicted_ids[:, :, 0]
        print(predictions)
        top_5_predictions = logits.predicted_ids[:, :, :5]
        beam_out = logits.beam_search_decoder_output,
        loss = tf.reduce_mean(beam_out[0][0])

        eval_metrics = {
            'char_accuracy': tf.metrics.accuracy(labels_ind, predictions),
            'cim_10_accuracy': custom_metrics.word_level_accuracy(labels_ind, predictions, name='CIM_10_accuracy'),
            'top_5_accuracy': custom_metrics.top_5_accuracy(
                tf.tile(
                    tf.expand_dims(
                        labels_ind,
                        axis=-1
                    ),
                    multiples=[1, 1, 5]
                ),
                top_5_predictions,
                name='top_5_accuracy'
            )
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metrics
        )
    
