import tensorflow as tf
from tensorflow.contrib import feature_column

COL_NAMES = [
    'Unnamed: 0',
    'jdc',   'mdc',   'adc',
    'jnais', 'mnais', 'anais',
    'sexe', 'activ', 'etatmat', 'lieudc', 'jvecus', 'image', 'causeini',
    'c1',  'c2',  'c3',  'c4',  'c5',  'c6',  'c7',  'c8',  'c9',  'c10', 'c11', 'c12',
    'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24',
    'fdepdc', 'fdepdom',
    'lieu_x_centroiddc',  'lieu_y_centroiddc',  'lieu_z_moyendc',
    'lieu_superficiedc',  'lieu_populationdc',  'lieu_statutdc',
    'lieu_x_centroiddom', 'lieu_y_centroiddom', 'lieu_z_moyendom',
    'lieu_superficiedom', 'lieu_populationdom', 'lieu_statutdom'
]

COL_TYPES = [[0.]] * (6 + 1) + [[0]] * 4 + [[0.]] + [[0]] + \
            [['']] * (1 + 24) + [[0.]] * 2 + \
            [[0.]] * 5 + [[0]] + \
            [[0.]] * 5 + [[0]]

loc_keys = [['lieu_' + y + x for x in ['dom', 'dc']] for y in ['statut', 'quant']]
date_keys = ['date_' + x for x in ['dc', 'nais']]
miscellaneous = ['sexe', 'activ', 'etatmat', 'lieudc', 'jvecus']
bucket_sizes = {'sexe': 2, 'activ': 3, 'etatmat': 4, 'lieudc': 6, 'lieu_statutdc': 6, 'lieu_statutdom': 6}
cause_chain = ['c' + str(x) for x in range(1, 25)]
vocab_list = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]
mean_dict = {
    'date_nais': [[1934.6344833246696], [6.395205434766596], [15.677822136824627]],
    'date_dc': [[2013], [6.298270277778363], [15.675359823892013]],
    'jvecus': 28620.254809402246,
    'lieu_quantdc': [6680.859375, 66543.929688, 166.329453, 3135.393311, 51.170135, 0.292135],
    'lieu_quantdom': [6682.340820, 66559.054688, 178.279785, 2653.858643, 32.751801, 0.286041]
}
std_dev_dict = {
    'date_nais': [[16.200837191507265], [3.4454328128621263], [8.818802916420342]],
    'date_dc': [[0.0000001], [3.5486695665231016], [8.791450686446204]],
    'jvecus': 5914.239761843669,
    'lieu_quantdc': [2130.321289, 2541.645508, 196.084808, 3912.409668, 72.844269, 1.568440],
    'lieu_quantdom': [2129.201172, 2540.051025, 211.765198, 3472.379639, 61.208401, 1.578119]
}
cause_name_list = ['c' + str(i) for i in range(0, 25)]


def _parse_line(line):
    """
    takes in a line of a csv file and returns its data as a feature dictionary
    :param line: the csv file's loaded line
    :return: the associated feature dictionary
    """

    fields = tf.decode_csv(line, record_defaults=COL_TYPES)
    features = dict(zip(COL_NAMES, fields))

    return features


def _convert(tensor, key):
    """
    Converts a string into a 1d tensor of char if the variable is of ICD-10 form, or into a 2d sparse
    tensor otherwise (for later sparse concatenation during _make_loc_variables()
    :param
    tensor: a tf.string
    :return: the entry string's associated char sequence
    """
    cause_name_list = ['c' + str(i) for i in range(0, 25)] + ['causeini']

    split = tf.string_split([tensor], delimiter='')

    if key in cause_name_list:
        split = tf.squeeze(
            tf.sparse_to_dense(split.indices, [1, 4], split.values, default_value=''),
            axis=0
        )

    return split


def _convert_string_format(features):
    """
    Takes in a feature dictionary, and splits each features defined as a string into a sequence of char.
    :param
    dataset: dictionary of features
    :return:
    features: the dataset's features, with string format converted into char sequences
    labels: The dataset's labels, converted into a sequence of char if necessary
    """

    for key in features.keys():
        if features[key].dtype != tf.float32:
            if features[key].dtype == tf.int32:
                features[key] = features[key] - 1
            else:
                features[key] = _convert(features[key], key)

    return features


def _make_date_variables(features, date_keys):
    """
    turns a dataset's date type variables into quantitative sequences for simplicity
    :param dataset: a feature dictinoary with date-like features, whose keys should be in format "a'date_name'" ,
    "m'date_name'" and "j'date_name'" for a given date's year, month and day entries, respectively
    :param date_keys: a dictionary containing every date_name present in the dataset
    :return: the modified feature dictionary with each entry of type "a'date_name'" , "m'date_name'"
    and "j'date_name'" replaced with an unique entry of key "date_'date_name'" pointing towards the date variable in
    sequence format
    """

    for key in date_keys:
        year = features.pop('a' + key)
        month = features.pop('m' + key)
        day = features.pop('j' + key)
        features['date_' + key] = tf.expand_dims(tf.stack((year, month, day), axis=-1), axis=-1)

    return features


def _make_loc_variables_old(features, loc_keys):
    """
    turns a dataset's location type variables into char sequences for simplicity
    :param dataset: a dictionary of features with location-like features, whose keys should be in format
    "dep'loc_name'" and "com'loc_name'" for a given location's department and city codes respectively
    :param loc_keys: a dictionary containing every loc_name present in the dataset
    :return: a modified feature dict where each dictionary entry of type "dep'loc_name'" and "com'loc_name'" is
    replaced with an unique entry of key "loc_'loc_name'" pointing towards the loc variable
    """

    for key in loc_keys:
        dep = features.pop('dep' + key)
        com = features.pop('com' + key)
        stacked = tf.sparse_concat(axis=-1, sp_inputs=[dep, com])

        features['loc_' + key] = tf.squeeze(
            tf.sparse_to_dense(
                sparse_indices=stacked.indices,
                output_shape=[1, 5],
                sparse_values=stacked.values,
                default_value=''
            ),
            axis=0
        )

    return features


def _make_loc_variables(features, loc_keys):

    for key in loc_keys:
        x = features.pop('lieu_x_centroid' + key)
        y = features.pop('lieu_y_centroid' + key)
        z = features.pop('lieu_z_moyen' + key)
        sup = features.pop('lieu_superficie' + key)
        pop = features.pop('lieu_population' + key)
        fdep = features.pop('fdep' + key)
        features['lieu_quant' + key] = tf.stack((x, y, z, sup, pop, fdep), axis=-1)

    return features


def _preprocess(line):
    """
    Overheads all csv processing functions.
    :param line: a raw csv line
    :return:
    """
    data = _parse_line(line)
    data = _convert_string_format(data)
    data = _make_date_variables(data, ['dc', 'nais'])
    data = _make_loc_variables(data, ['dom', 'dc'])

    labels = data.pop('causeini')

    return data, labels


def really_simple_input_fn(dataset_name, batch_size, num_epochs):
    """
    A predefined input function to feed an Estimator csv based cepidc files
    :param dataset_name: the file's ending type (either 'train, 'valid' or 'test')
    :param batch_size: the size of batches to feed the computational graph
    :param num_epochs: the number of time the entire dataset should be exposed to a gradient descent iteration
    :return: a BatchedDataset as a tuple of a feature dictionary and the labels
    """

    dataset = tf.data.TextLineDataset('cepidc_2013_' + dataset_name + '.csv').skip(1)

    dataset = dataset.map(_preprocess)

    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True).repeat(num_epochs) #TODO put that back after testing
    
    dataset = dataset.prefetch(buffer_size=batch_size)
    
    return dataset


def input_normalizer(tensor, key):
    """
    Normalizes a qualitative variable 0 mean and unit variance
    :param tensor: an ND tensor
    :param key: the key identifying the tensor variable in the dataset dict
    :return: the mean-var normalized tensor
    """
    return (tensor - mean_dict[key]) / std_dev_dict[key]


def make_columns():
    """
    Builds the feature_columns required by the estimator to link the Dataset and the model_fn
    :return:
    """
    # build feature_column for date variables
    with tf.name_scope('date_columns'):

        date_columns = []

        date_columns.append(
            tf.feature_column.numeric_column(
                date_keys[0],
                shape=(3, 1),
                normalizer_fn=lambda x: input_normalizer(x, date_keys[0])
            ))

        date_columns.append(
            tf.feature_column.numeric_column(
                date_keys[1],
                shape=(3, 1),
                normalizer_fn=lambda x: input_normalizer(x, date_keys[1])
            ))

    # build feature_columns for localisation variables
    with tf.name_scope('loc_columns'):

        with tf.name_scope('dom_columns'):

            loc_dom_columns = []

            loc_dom_columns.append(
                tf.feature_column.numeric_column(
                    loc_keys[1][0],
                    shape=6,
                    normalizer_fn=lambda x: input_normalizer(x, loc_keys[1][0])
                )
            )

            loc_dom_columns.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_identity(
                    loc_keys[0][0],
                    num_buckets=bucket_sizes[loc_keys[0][0]]
                )
            ))

        with tf.name_scope('dc_columns'):

            loc_dc_columns = []

            loc_dc_columns.append(
                tf.feature_column.numeric_column(
                    loc_keys[1][1],
                    shape=6,
                    normalizer_fn=lambda x: input_normalizer(x, loc_keys[1][1])
                )
            )

            loc_dc_columns.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_identity(
                    loc_keys[0][1],
                    num_buckets=bucket_sizes[loc_keys[0][1]]
                )
            ))

        loc_cat_columns = []
        with tf.name_scope('statut_columns'):

            for key in loc_keys[0]:
                loc_cat_columns.append(tf.feature_column.indicator_column(
                    tf.feature_column.categorical_column_with_identity(
                        key,
                        num_buckets=bucket_sizes[key]
                    )
                ))

    # build feature_columns for miscellaneous variables
    with tf.name_scope('misc_columns'):

        misc_columns = []

        for misc_key in miscellaneous:
            if misc_key != 'jvecus':
                misc_columns.append(
                    tf.feature_column.indicator_column(
                        tf.feature_column.categorical_column_with_identity(
                            misc_key,
                            num_buckets=bucket_sizes[misc_key]
                        )))

            else:
                misc_columns.append(
                    tf.feature_column.numeric_column(
                        misc_key,
                        normalizer_fn=lambda x: input_normalizer(x, misc_key)))

    # build feature columns for causal chain variables
    with tf.name_scope('cause_columns'):

        cause_columns = []

        for key in cause_chain:
            cause_columns.append(tf.feature_column.indicator_column(
                feature_column.sequence_categorical_column_with_vocabulary_list(key, vocab_list, default_value=0)))

    # build feature column for the labels
    with tf.name_scope('labels_one_hot_encoding'):

        label_column = tf.feature_column.indicator_column(
            feature_column.sequence_categorical_column_with_vocabulary_list('labels', vocab_list, default_value=0))

    columns_dict = {
        'date': date_columns,
        'loc_dom': loc_dom_columns,
        'loc_dc': loc_dc_columns,
        'misc': misc_columns,
        'cause': cause_columns,
        'labels': label_column
    }

    return columns_dict


def compute_len(tensor):
    """
    Takes in the current batch of causal chain input and outputs each individual's causal chain length
    :param tensor: a 4D tensor of dimension [batch_size, code_size, number of causes, encoding_size] containing the
    causal chain's ICD-10 one hot encodings as alphadecimal characters (36 states)
    :return: a 1D tensor of dimension [batch_size] containing each individual's causal chain length
    """

    first = tf.cast(tf.equal(tf.reduce_max(tensor, axis=-1), 1), tf.int32)
    second = tf.reduce_max(first, axis=-2)

    return tf.reduce_sum(second, axis=-1)


def make_input_layers(dataset, feature_columns):
    feature_dict = {}

    features, labels = dataset

    with tf.name_scope('date_inputs'):
        feature_dict['date'] = tf.reshape(
            tf.feature_column.input_layer(features, feature_columns['date']),
            shape=[-1, 2, 3, 1]
        )

    with tf.name_scope('loc_inputs'):
        feature_dict['loc'] = tf.stack(
            (
                tf.feature_column.input_layer(features, feature_columns['loc_dom'], trainable=False),
                tf.feature_column.input_layer(features, feature_columns['loc_dc'], trainable=False)
            ),
            axis=1
        )

    with tf.name_scope('misc_inputs'):
        feature_dict['misc'] = tf.feature_column.input_layer(
            features,
            feature_columns['misc'],
            trainable=False
        )

    with tf.name_scope('cause_columns'):
        cause_input = []

        for cause_column in feature_columns['cause']:
            cause_input.append(feature_column.sequence_input_layer(features, cause_column, trainable=False)[0])

        feature_dict['cause'] = tf.stack(cause_input, axis=-2)
        feature_dict['cause_len'] = compute_len(feature_dict['cause'])
        feature_dict['cause'] = feature_dict['cause'][:, :4, :tf.reduce_max(feature_dict['cause_len'])]

    with tf.name_scope('labels_one_hot_encoding'):
        label_inputs = feature_column.sequence_input_layer(
            {'labels': labels}, feature_columns['labels'], trainable=False)[0]

    return feature_dict, label_inputs
