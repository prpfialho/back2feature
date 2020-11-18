import os, pickle
import random as rn
import unicodedata
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, \
    balanced_accuracy_score, accuracy_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler, PowerTransformer, MinMaxScaler
from assin.commons import read_xml


def init_seed(tf):
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(123)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see:
    # https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                                  # , allow_soft_placement=True)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    # tf.set_random_seed(1234)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # tf.keras.backend.set_session(sess)
    tf.random.set_seed(1234)


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def get_ms(dsetpart, tokenizer, cased=True, maxlen=128, pathprefx='~/PycharmProjects/phd/msr'):
    dev_dict = {}
    tr_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        'ents': [],
        'sims': np.array([])
    }

    all_data_byid = {}
    with open(pathprefx + dsetpart + '.txt') as f:
        for line in f:
            lineitems = line.strip().split('\t')
            all_data_byid[lineitems[1] + '-' + lineitems[2]] = (
                lineitems[3] if cased else remove_accents(lineitems[3].lower()),
                lineitems[4] if cased else remove_accents(lineitems[4].lower()),
                int(lineitems[0][-1:]))

    if dsetpart == 'train':
        tr_data_byid = {}
        dev_data_byid = {}

        dev_ids = []
        with open('~/backupffs/phd/data/glue-msrp-dev_ids.tsv') as f:
            for line in f:
                lineitems = line.strip().split('\t')
                dev_ids.append(lineitems[0] + '-' + lineitems[1])

        for key, value in all_data_byid.items():
            if key in dev_ids:
                dev_data_byid[key] = value
            else:
                tr_data_byid[key] = value

        dev_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            'ents': []
        }

        entails = []
        for _, value in dev_data_byid.items():
            pair_dict = tokenizer.encode_plus(value[0],
                                              value[1],
                                              max_length=maxlen,
                                              pad_to_max_length=True).data
            dev_dict['input_ids'].append(pair_dict['input_ids'])
            if 'token_type_ids' in pair_dict:
                dev_dict['token_type_ids'].append(pair_dict['token_type_ids'])
            dev_dict['attention_mask'].append(pair_dict['attention_mask'])

            entails.append(value[2])

        dev_dict['input_ids'] = np.array(dev_dict['input_ids'])
        dev_dict['token_type_ids'] = np.array(dev_dict['token_type_ids'])
        dev_dict['attention_mask'] = np.array(dev_dict['attention_mask'])

        ents = np.array(entails).reshape((len(entails), 1))
        num_classes = int(ents.max() + 1)

        ents_classprobs = []
        for e in entails:
            tmp_arr = []
            for itr in range(num_classes):
                tmp_arr.append(0)
            tmp_arr[e] = 1
            ents_classprobs.append(tmp_arr)

        dev_dict['ents'] = np.array(ents_classprobs)
    else:
        tr_data_byid = all_data_byid

    entails = []
    for _, value in tr_data_byid.items():
        entails.append(value[2])

        if tokenizer:
            pair_dict = tokenizer.encode_plus(value[0],
                                              value[1],
                                              max_length=maxlen,
                                              pad_to_max_length=True).data
            tr_dict['input_ids'].append(pair_dict['input_ids'])
            if 'token_type_ids' in pair_dict:
                tr_dict['token_type_ids'].append(pair_dict['token_type_ids'])
            tr_dict['attention_mask'].append(pair_dict['attention_mask'])

    tr_dict['input_ids'] = np.array(tr_dict['input_ids'])
    tr_dict['token_type_ids'] = np.array(tr_dict['token_type_ids'])
    tr_dict['attention_mask'] = np.array(tr_dict['attention_mask'])

    ents = np.array(entails).reshape((len(entails), 1))
    num_classes = int(ents.max() + 1)

    ents_classprobs = []
    for e in entails:
        tmp_arr = []
        for itr in range(num_classes):
            tmp_arr.append(0)
        tmp_arr[e] = 1
        ents_classprobs.append(tmp_arr)

    tr_dict['ents'] = np.array(ents_classprobs)

    return tr_dict, dev_dict


str_to_entailment = {'NEUTRAL': 0,
                     'ENTAILMENT': 1,
                     'CONTRADICTION': 2}


def sel_countfeats(a):
    return [col
            for col in range(a.shape[1])
            if any([a[row][col] > 1 for row in range(a.shape[0])])
            ]


def vec_scale(tr_lex_x_split, dvec=None, scalr=None):
    if dvec:
        tr_lex_x_vec = dvec.transform(tr_lex_x_split)
    else:
        dvec = DictVectorizer(sparse=False)
        tr_lex_x_vec = dvec.fit_transform(tr_lex_x_split)

    if scalr:
        tr_lex_x_vec = scalr.transform(tr_lex_x_vec)
    else:
        # scale all features to range 0 - 1
        scalr = make_pipeline(
            make_column_transformer(
                (make_pipeline(RobustScaler(), PowerTransformer(), MinMaxScaler()), sel_countfeats),
                remainder='passthrough'),
        )

        tr_lex_x_vec = scalr.fit_transform(tr_lex_x_vec)

    return tr_lex_x_vec, dvec, scalr


def lex_features(dsetpart, dvec_lex=None, scalr_lex=None,
                 cprefx='~/backupffs/phd/cache/assin2/lex-pt_XyID_assin2-'):

    tr_l = pickle.load(open(cprefx + dsetpart + '.p2', "rb"), encoding='latin1')

    dpart_lex_x = tr_l[0]
    dpart_lex_ids = list(np.array(tr_l[-1], dtype=int))
    tr_ent = [int(i.split('\t')[0]) for i in tr_l[1]]
    tr_sim = [float(i.split('\t')[1]) for i in tr_l[1]]

    if dvec_lex and scalr_lex:
        tr_x_lex, _, _ = vec_scale(dpart_lex_x, dvec_lex, scalr_lex)
    else:
        tr_x_lex, dvec_lex, scalr_lex = vec_scale(dpart_lex_x)

    y_regr = np.array(tr_sim).reshape((len(tr_sim), ))
    y_clf = np.array(tr_ent).reshape((len(tr_ent), ))

    return tr_x_lex, dvec_lex, scalr_lex, y_clf, y_regr, dpart_lex_ids


def bert_in(s1, s2, entails, sims, tokenizer, maxlen=128):
    ents = np.array(entails).reshape((len(entails), 1))
    num_classes = int(ents.max() + 1)

    ents_classprobs = []
    for e in entails:
        tmp_arr = []
        for itr in range(num_classes):
            tmp_arr.append(0)
        tmp_arr[e] = 1
        ents_classprobs.append(tmp_arr)

    tr_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        'ents': np.array(ents_classprobs),
        'sims': np.array(sims).reshape((len(sims), 1))
    }

    if tokenizer:
        for i in range(len(s1)):
            pair_dict = tokenizer.encode_plus(s1[i], s2[i], max_length=maxlen, pad_to_max_length=True).data
            tr_dict['input_ids'].append(pair_dict['input_ids'])
            if 'token_type_ids' in pair_dict:
                tr_dict['token_type_ids'].append(pair_dict['token_type_ids'])
            tr_dict['attention_mask'].append(pair_dict['attention_mask'])

        # for a in tr_dict:
        tr_dict['input_ids'] = np.array(tr_dict['input_ids'])
        if len(tr_dict['token_type_ids']) > 0:
            tr_dict['token_type_ids'] = np.array(tr_dict['token_type_ids'])
        tr_dict['attention_mask'] = np.array(tr_dict['attention_mask'])
        # tr_dict['input_ids'] = np.array(
        #     [i + [0]*((len(max(tr_dict['input_ids'], key=len)))-len(i)) for i in tr_dict['input_ids']])
        # tr_dict['token_type_ids'] = np.array(
        #     [i + [0] * ((len(max(tr_dict['token_type_ids'], key=len))) - len(i)) for i in tr_dict['token_type_ids']])
        # tr_dict['attention_mask'] = np.array(
        #     [i + [0] * ((len(max(tr_dict['attention_mask'], key=len))) - len(i)) for i in tr_dict['attention_mask']])

    return tr_dict


def bert_in_tsv(dsetpath, dsetpart, tokenizer, cased=True, maxlen=128, norm_sim=False):
    s1 = []
    s2 = []
    entails = []
    sims = []
    pairids = []
    with open(dsetpath + dsetpart + '.tsv') as f:
        for line in f:
            lineitems = line.strip().split('\t')

            s1.append(lineitems[0] if cased else remove_accents(lineitems[0].lower()))
            s2.append(lineitems[1] if cased else remove_accents(lineitems[1].lower()))

            # pairids.append(lineitems[0])

            ent_value = 0  # NEUTRAL
            entstr = lineitems[2]
            if entstr == 'ENTAILMENT':
                ent_value = 1
            if entstr == 'CONTRADICTION':
                ent_value = 2
            if entstr == 'PARAPHRASE':
                ent_value = 3

            entails.append(ent_value)

    return bert_in(s1, s2, entails, sims, tokenizer, maxlen), pairids


def bert_in_sicks(dsetpath, dsetpart, tokenizer, cased=True, maxlen=128, norm_sim=False):
    s1 = []
    s2 = []
    entails = []
    sims = []
    pairids = []
    with open(dsetpath) as f:
        for line in f:
            lineitems = line.strip().split('\t')

            if not lineitems[-1].strip() in dsetpart:  # or lineitems[-1].startswith('TRIAL'):
                continue

            s1.append(lineitems[1] if cased else remove_accents(lineitems[1].lower()))
            s2.append(lineitems[2] if cased else remove_accents(lineitems[2].lower()))

            pairids.append(lineitems[0])

            ent_value = 0  # NEUTRAL
            entstr = lineitems[3]
            if entstr == 'ENTAILMENT':
                ent_value = 1
            if entstr == 'CONTRADICTION':
                ent_value = 2

            entails.append(ent_value)
            sim = float(lineitems[4])
            sims.append(sim / 5 if norm_sim else sim)

            # isparaphrase = '0'
            # ab = lineitems[5]
            # ba = lineitems[6]
            # if ab == 'A_entails_B' and ba == 'B_entails_A':
            #     isparaphrase = '1'  # paraphrase

    return bert_in(s1, s2, entails, sims, tokenizer, maxlen), pairids


def bert_in_assin12(dsetpart, tokenizer, cased=True, maxlen=128, norm_sim=False,
                    pathprefx='~/backupffs/phd/data/assin/assin2-'):
    dev = read_xml(pathprefx + dsetpart + '.xml', True)
    s1 = []
    s2 = []
    entails = []
    sims = []
    pairids = []
    for pairobj in dev:
        s1.append(pairobj.t if cased else remove_accents(pairobj.t.lower()))
        s2.append(pairobj.h if cased else remove_accents(pairobj.h.lower()))
        entails.append(pairobj.entailment)
        sims.append(pairobj.similarity / 5 if norm_sim else pairobj.similarity)
        pairids.append(pairobj.id)

    return bert_in(s1, s2, entails, sims, tokenizer, maxlen), pairids


def model_eval(predictions, te_regr2, te_y_clf1, multiclass='macro'):
    preds_clf = predictions
    if te_regr2.shape[0] > 0:
        preds_clf = predictions[0]
        preds_regr = predictions[1]

        te_regr1 = te_regr2.reshape((len(te_regr2),))
        predregr = np.array(preds_regr, dtype=np.float64).reshape((preds_regr.shape[0],))

        mse = mean_squared_error(te_regr1, predregr)
        pe = pearsonr(te_regr1, predregr)[0]
        sp = spearmanr(te_regr1, predregr)[0]
        print('\n# mse pe sp')
        print(mse)
        print(pe)
        print(sp)

    predclf = np.argmax(preds_clf, axis=1)
    predclf = predclf.reshape((predclf.shape[0], 1))

    te_y_clf = np.argmax(te_y_clf1, axis=1)
    te_y_clf = te_y_clf.reshape((te_y_clf.shape[0], 1))

    f1 = f1_score(te_y_clf, predclf, average=multiclass)
    pre = precision_score(te_y_clf, predclf, average=multiclass)
    rec = recall_score(te_y_clf, predclf, average=multiclass)
    acc = accuracy_score(te_y_clf, predclf)
    print('\n# acc f1-' + multiclass + ' pre rec')
    print(acc)
    print(f1)
    print(pre)
    print(rec)
