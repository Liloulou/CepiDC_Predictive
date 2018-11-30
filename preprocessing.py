import numpy as np
import pandas as pd

to_delete = ['csp', 'depnai3', 'depdc3', 'depdom3', 'annaismere', '_name_', 'image', 'num_certificat', 'nat3']
geo_columns = ['INSEE_COM', 'X_CENTROID', 'Y_CENTROID', 'Z_MOYEN', 'SUPERFICIE', 'POPULATION', 'STATUT']

dict_type = {'image': np.str,
             '_NAME_': np.str,
             'c1': np.str,
             'c2': np.str,
             'c3': np.str,
             'c4': np.str,
             'c5': np.str,
             'c6': np.str,
             'c7': np.str,
             'c8': np.str,
             'c9': np.str,
             'c10': np.str,
             'c11': np.str,
             'c12': np.str,
             'c13': np.str,
             'c14': np.str,
             'c15': np.str,
             'c16': np.str,
             'c17': np.str,
             'c18': np.str,
             'c19': np.str,
             'c20': np.str,
             'c21': np.str,
             'c22': np.str,
             'c23': np.str,
             'c24': np.str,
             'jdc': np.int32,
             'mdc': np.int32,
             'adc': np.int32,
             'jnais': np.int32,
             'mnais': np.int32,
             'anais': np.int32,
             'depdom': np.int32,
             'depdc': np.int32,
             'depnais': np.int32,
             'numcertificat': np.str,
             'depdc3': np.int32,
             'comdc': np.int32,
             'sexe': np.int32,
             'depnais3': np.int32,
             'comnais': np.int32,
             'activ': np.int32,
             'csp': np.str,
             'situat': np.str,
             'nat3': np.str,
             'depdom3': np.int32,
             'comdom': np.int32,
             'etatmat': np.int32,
             'lieudc': np.int32,
             'jvecus': np.int32,
             'image': np.str,
             'agexact': np.int32,
             'causeini': np.str,
             'AnNaisMere': np.str}


def clean(tout):
    tout = tout.replace('NUL', '')
    tout = tout.replace('NU', '')
    tout = tout.replace('N', '')

    for val in ['R96', 'R97', 'R98', 'R99']:
        tout = tout.loc[tout['causeini'] != val]

    return tout


def merge_and_delete_init_cause(all_codes, tout):
    all_codes = all_codes.loc[np.logical_not(all_codes.duplicated(subset='image'))]
    tout = tout.loc[np.logical_not(tout.duplicated(subset='image'))]

    intersect = np.setxor1d(tout['image'], all_codes['image'])

    all_codes = all_codes.loc[~all_codes['image'].isin(intersect)]
    tout = tout.loc[~tout['image'].isin(intersect)]
    merged = tout.merge(all_codes, on='image', how='inner')
    merged = clean(merged)
    """
    cause_chain_columns = []
    for i in range(1, 25):
        cause_chain_columns.append('c' + str(i))

    other_columns = np.array(merged.columns[np.argwhere(np.logical_not(merged.columns.isin(cause_chain_columns)))])

    cause_ini = np.array(merged['causeini'])
    cause_chain = np.array(merged[cause_chain_columns])
    is_cause_ini = np.array(np.transpose(np.transpose(cause_chain) == cause_ini))

    tot = is_cause_ini.shape[0]
    for i in range(tot):
        temp_cause_chain = cause_chain[i, np.logical_not(is_cause_ini[i])]
        cause_chain[i] = np.pad(
            temp_cause_chain,
            pad_width=[0, 24 - len(temp_cause_chain)],
            mode='constant', constant_values=np.nan)

    merged[cause_chain_columns] = cause_chain
    """
    return merged


def clean_dataset(dataset):
    dataset = dataset.drop(columns=[
        'csp',
        'depnais3',
        'depdc3',
        'depdom3',
        'AnNaisMere',
        '_NAME_',
        'numcertificat',
        'nat3',
        'depnais',
        'comnais',
        'situat', 'agexact'])

    cause_chain_columns = []
    for i in range(2, 25):
        cause_chain_columns.append('c' + str(i))

    other_columns = np.array(
        dataset.columns[np.argwhere(np.logical_not(np.isin(dataset.columns, cause_chain_columns)))])[:, 0]

    row_with_missing = dataset[other_columns].isnull()

    dataset = dataset.loc[~row_with_missing.any(1)]

    return dataset


def test_correctness(init_data, final_data):
    images = final_data['image']

    cause_chain_columns = []
    for i in range(1, 25):
        cause_chain_columns.append('c' + str(i))

    other_columns = np.array(
        final_data.columns[np.argwhere(np.logical_not(np.isin(final_data.columns, cause_chain_columns)))])[:, 0]

    init_subset = init_data.loc[init_data['image'].isin(images)][other_columns]

    test = np.array(init_subset) == np.array(final_data[other_columns])

    return np.logical_and.reduce(test, axis=None)


def add_fdep(dataset):
    fdep = pd.read_csv('data/geo_data/fdep13_communes.csv')
    fdep = fdep.rename(index=str, columns={'fdep13': 'fdep'})

    fdep['fdep'] = fdep['fdep'].str.replace(',', repl='.')

    fdep['CODGEO'].loc[fdep['CODGEO'].str.len() == 4] = fdep['CODGEO'].loc[fdep['CODGEO'].str.len() == 4].str.pad(
        width=5, side='left', fillchar='0'
    )

    dataset = dataset.loc[dataset['inseedc'].isin(fdep['CODGEO'])]

    dataset = pd.merge(
        dataset, fdep[['CODGEO', 'fdep']], how='left', left_on='inseedc', right_on='CODGEO', validate='m:1'
    )
    dataset = pd.merge(
        dataset, fdep[['CODGEO', 'fdep']], how='left', left_on='inseedom',
        right_on='CODGEO', suffixes=('dc', 'dom'), validate='m:1'
    )

    dataset = dataset.drop(columns=['CODGEOdc', 'CODGEOdom'])

    return dataset


def add_geo_data(dataset):

    geo = pd.read_csv('data/geo_data/geo_data.csv')
    geo = geo[geo_columns]
    geo.columns = ['lieu_' + x.lower() for x in geo.columns]

    for type in ['dc', 'dom']:
        dataset['insee' + type] = dataset['dep' + type] + dataset['com' + type]
        dataset = dataset.loc[dataset['insee' + type].isin(geo['lieu_insee_com'])]

    dataset = add_fdep(dataset)

    dataset = pd.merge(dataset, geo, how='left', left_on='inseedc', right_on='lieu_insee_com', validate='m:1')
    dataset = pd.merge(
        dataset, geo, how='left', left_on='inseedom', right_on='lieu_insee_com', suffixes=('dc', 'dom'), validate='m:1'
    )

    dataset = dataset.drop(columns=['depdom', 'depdc', 'comdc', 'comdom', 'comdc',
                                    'inseedc', 'inseedom', 'lieu_insee_comdc', 'lieu_insee_comdom'])
    dataset['lieu_statutdom'] += 1
    dataset['lieu_statutdc'] += 1

    return dataset


all_codes = pd.read_csv('data/cepidc_raw/all_codes_2013.csv', dtype=np.str)
tout = pd.read_csv('data/cepidc_raw/tout_2013.csv', dtype=np.str)
dataset = merge_and_delete_init_cause(all_codes, tout)
dataset = clean_dataset(dataset)


dataset = add_geo_data(dataset)

dataset = dataset.sample(frac=1)  # shuffle dataset

train = dataset.iloc[:-75000]
valid = dataset.iloc[-75000:-37500]
test = dataset.iloc[-37500:]

train.to_csv('cepidc_2013_train.csv')
valid.to_csv('cepidc_2013_valid.csv')
test.to_csv('cepidc_2013_test.csv')
