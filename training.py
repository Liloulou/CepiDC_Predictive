import tensorflow as tf
import subprocess
import pickle
import numpy as np
import my_input_fn as pipe
import model
from tensorflow.contrib import feature_column
import os

model_dir = 'model_directory/'
hparams = {
        'batch_size': 10,
        'num_epochs': 500,
        'train': {
            'learning_rate': 0.001,
            'decay_rate': 0.98,
            'decay_steps': 2000
        },
        'inference': {
            'misc': {
                'units': [50, 30],
                'keep_prob': [0.01, 0.01]
            },
            'date': {
                'filters': [50, 50],
                'kernel': [2, 3],
                'units': [50, 30],
                'keep_prob': [0.01, 0.01]
            },
            'loc': {
                'filters': [70, 70],
                'kernel': [2, 3],
                'units': [50, 30],
                'keep_prob': [0.01, 0.01]
            },
            'cause': {
                'block': [
                    {
                        'dimension': 2,
                        'filters': [5, 5],
                        'kernel': [2, 3],
                        'drop_out': 0.01
                    },
                    {
                        'dimension': 2,
                        'filters': [5, 5],
                        'kernel': [2, 3],
                        'drop_out': 0.01
                    },
                ],
                'units': 10,
                'drop_out': 0.01
            },
            'encoder': {
                'dimension': 1,
                'filters': [5, 5],
                'kernel': [3, 3],
                'drop_out': 0.01
            },
            'decoder': {
                'bahdanau': 50,
                'units': [10, 10],
                'drop_out': 0.01
            }
        }
    }


def get_next_model_dir():
    list_name = [int(name[name.find('_') + 1:]) for name in os.listdir('model_directory')]

    if len(list_name) is 0:
        last_model = 0
    else:
        last_model = max(list_name)

    return 'model_directory/model_' + str(last_model + 1)


def make_hparams():
    # search granularity
    step = 5

    # drop out parameters
    drop_out_1 = int(np.random.uniform(1, 8)) * 0.1
    drop_out_2 = int(np.random.uniform(1, 8)) * 0.1
    drop_out_3 = int(np.random.uniform(1, 8)) * 0.1

    # cause network pre-definition
    cause_max_val = 500
    lay_per_block_cause = np.random.binomial(1, 0.5) + 2

    # recurrent encoder/decoder pre-definition
    rec_max_val = 500
    decoder_len = int(np.random.uniform(1, 4))
    encoder_len = np.minimum(4, decoder_len + int(np.random.uniform(0, 4)))

    hparams = {}
    hparams['batch_size'] = int(np.random.uniform(32, 100))
    hparams['num_epochs'] = 500
    hparams['train'] = {
        'learning_rate': 10 ** -int(np.random.uniform(2, 4)),
        'decay_rate': np.random.uniform(0.95, 1),
        'decay_steps': int(np.random.uniform(1500, 4000))
    }
    bla = {}
    bla['cause'] = {
        'block': [
            {
                'dimension': 2,
                'filters': [int(np.random.uniform(1, step + 1) * cause_max_val // step)] * lay_per_block_cause,
                'kernel': [int(np.random.uniform(2, 4))] * lay_per_block_cause,
                'drop_out': drop_out_1
            },
            {
                'dimension': 2,
                'filters': [int(np.random.uniform(1, step + 1) * cause_max_val // step)] * lay_per_block_cause,
                'kernel': [int(np.random.uniform(2, 4))] * lay_per_block_cause,
                'drop_out': drop_out_1
            },
            {
                'dimension': 2,
                'filters': [int(np.random.uniform(1, step + 1) * cause_max_val // step)] * lay_per_block_cause,
                'kernel': [int(np.random.uniform(2, 4))] * lay_per_block_cause,
                'drop_out': drop_out_1
            },
            {
                'dimension': 2,
                'filters': [int(np.random.uniform(1, step + 1) * cause_max_val // step)] * lay_per_block_cause,
                'kernel': [int(np.random.uniform(2, 4))] * lay_per_block_cause,
                'drop_out': drop_out_1
            },
        ],
        'units': int(np.random.uniform(1, step)) * cause_max_val // step,
        'drop_out': drop_out_1
    }
    bla['encoder'] = {
        'units': [int(np.random.uniform(1, step + 1) * rec_max_val // step)] * encoder_len,
        'drop_out': drop_out_2,
    }
    bla['decoder'] = {
        'bahdanau': int(np.random.uniform(1, step + 1) * rec_max_val // step),
        'units': [int(np.random.uniform(1, step + 1) * rec_max_val // step)] * decoder_len,
        'drop_out': drop_out_3,
    }
    hparams['inference'] = bla

    return hparams


tf.logging.set_verbosity(tf.logging.INFO)

for i in range(20):

    next_model_dir = get_next_model_dir()

    hparams['feature_columns'] = pipe.make_columns()
    run_config = tf.estimator.RunConfig(
        model_dir=get_next_model_dir(),
        save_checkpoints_steps=500,
        keep_checkpoint_max=1,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model.get_model_fn,
        config=run_config,
        params=hparams
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: pipe.really_simple_input_fn('train', hparams['batch_size'], hparams['num_epochs']),
        max_steps=100000,
        hooks=[tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator,
            metric_name='loss',
            max_steps_without_decrease=10000,
            run_every_secs=None,
            run_every_steps=500
        )]
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: pipe.really_simple_input_fn('valid', hparams['batch_size'], 20),
        steps=100,
        start_delay_secs=60,
        throttle_secs=120
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    pickle.dump(
        hparams,
        open(
            'data/' + '/h_parameters/h_params_' + next_model_dir[next_model_dir.find('_') + 1:],
            'wb'
        )
    )

