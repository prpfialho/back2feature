import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
# gpu_idx = 0
# tf.config.experimental.set_visible_devices([gpus[gpu_idx]], 'GPU')        # , gpus[2], gpus[3]
# tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from zutils import model_eval, get_ms, lex_features, bert_in_sicks
from transformers import TFBertModel, BertConfig, BertTokenizer, AutoTokenizer, AutoConfig
# , TFXLMRobertaForMaskedLM, TFXLMRobertaModel, cached_path, TFBertPreTrainedModel
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random as rn
import sys, os, pickle


if __name__ == "__main__":
    np.random.seed(123)
    rn.seed(123)
    tf.random.set_seed(1234)

    batchsize = 32
    numepochs = 4
    max_len = 128
    multiclass = 'macro'  # 'binary' macro
    usecache = 1
    # bert-base-cased bert-large-cased-whole-word-masking
    modldir = (sys.argv[3] if len(sys.argv) > 3 else
               'bert-large-cased-whole-word-masking')
               # '/afs/l2f/home/pfialho/backupffs/phd/data/bert-models/wwm_cased_L-24_H-1024_A-16/')

    modlsavedir = '/afs/l2f/home/pfialho/backupffs/phd/cache/bert-large-cased-wwm/' + \
                  (sys.argv[2] if len(sys.argv) > 2 else 'ft_ms') + '/'

    runnum = (sys.argv[1] if len(sys.argv) > 1 else '1')
    outfile = 'run' + runnum + '.p2'
    modlsavepath = modlsavedir + 'tfmodel' + runnum + '/'

    w_lex = ('lex' in modlsavedir)

    print(modldir)
    print(modlsavedir)

    if modldir.startswith('/'):
        tokenizer = BertTokenizer.from_pretrained(modldir + 'vocab.txt', do_lower_case=False)
        config = BertConfig.from_pretrained(modldir + 'bert_config.json', output_hidden_states=True)
    else:
        modlstr = modldir   # bert-base-multilingual-uncased     xlm-roberta-base
        modlcache = '/afs/l2f/home/pfialho/backupffs/phd/data/hf/' + modlstr + '/'
        config = AutoConfig.from_pretrained(modlstr, output_hidden_states=True, cache_dir=modlcache)        # BertConfig
        tokenizer = AutoTokenizer.from_pretrained(modlstr, cache_dir=modlcache)     # BertTokenizer

    # prepare inputs
    if 'sick' in modlsavedir:
        sickpath = '../phd/SICK.txt'
        tr_dict, _ = bert_in_sicks(sickpath, 'TRAIN', tokenizer, maxlen=max_len)
        dev_dict, _ = bert_in_sicks(sickpath, 'TRIAL', tokenizer, maxlen=max_len)
        te_dict, _ = bert_in_sicks(sickpath, 'TEST', tokenizer, maxlen=max_len)
    elif 'ms' in modlsavedir:
        tr_dict, dev_dict = get_ms('train', tokenizer, maxlen=max_len)
        te_dict, _ = get_ms('test', tokenizer, maxlen=max_len)

    tr_len = tr_dict["ents"].shape[0]
    num_classes = tr_dict["ents"].shape[1]

    if w_lex:
        tr_x_lex, dvec_lex, scalr_lex, y_clf, y_regr = lex_features('train-only')
        dev_x_lex, _, _, dev_y_clf, dev_y_regr = lex_features('dev', dvec_lex=dvec_lex, scalr_lex=scalr_lex)
        te_x_lex, _, _, te_y_clf, te_y_regr = lex_features('test', dvec_lex=dvec_lex, scalr_lex=scalr_lex)

        assert (te_y_clf == te_dict['ents']).all()
        assert (te_y_regr == te_dict['sims']).all()
        assert (y_clf == tr_dict['ents']).all()
        assert (y_regr == tr_dict['sims']).all()
        assert (dev_y_clf == dev_dict['ents']).all()
        assert (dev_y_regr == dev_dict['sims']).all()

    if len(te_dict['token_type_ids']) > 0:
        if w_lex:
            te_x = [te_dict["input_ids"], te_dict["token_type_ids"], te_dict["attention_mask"], te_x_lex]  #
        else:
            te_x = [te_dict["input_ids"], te_dict["token_type_ids"], te_dict["attention_mask"]]
    else:
        if w_lex:
            te_x = [te_dict["input_ids"], te_dict["attention_mask"], te_x_lex]
        else:
            te_x = [te_dict["input_ids"], te_dict["attention_mask"]]

    if usecache and os.path.isfile(modlsavedir + outfile):
        predictions = pickle.load(open(modlsavedir + outfile, "rb"))
    else:
        if usecache and os.path.exists(modlsavepath) and len(os.listdir(modlsavepath)) > 0:
            print('# loading ' + modlsavedir)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            model = keras.models.load_model(modlsavedir)
        else:
            if len(tr_dict['token_type_ids']) > 0:
                if w_lex:
                    tr_x = [tr_dict["input_ids"], tr_dict["token_type_ids"], tr_dict["attention_mask"], tr_x_lex]
                    dev_x = [dev_dict["input_ids"], dev_dict["token_type_ids"], dev_dict["attention_mask"], dev_x_lex]
                else:
                    tr_x = [tr_dict["input_ids"], tr_dict["token_type_ids"], tr_dict["attention_mask"]]
                    dev_x = [dev_dict["input_ids"], dev_dict["token_type_ids"], dev_dict["attention_mask"]]
            else:
                if w_lex:
                    tr_x = [tr_dict["input_ids"], tr_dict["attention_mask"], tr_x_lex]
                    dev_x = [dev_dict["input_ids"], dev_dict["attention_mask"], dev_x_lex]
                else:
                    tr_x = [tr_dict["input_ids"], tr_dict["attention_mask"]]
                    dev_x = [dev_dict["input_ids"], dev_dict["attention_mask"]]

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                if modldir.startswith('/'):
                    encoder = TFBertModel.from_pretrained(modldir, config=config, from_pt=True)
                else:
                    encoder = TFBertModel.from_pretrained(modlstr, config=config, cache_dir=modlcache)

                if w_lex:
                    lex_inp = keras.Input(shape=(94,), name="lex")

                    # lex_dense = layers.Dense(94)(lex_inp)
                    # , activation = 'sigmoid',
                    # kernel_initializer = keras.initializers.TruncatedNormal(
                    #     stddev=config.initializer_range)
                    # lex_dense.trainable = False

                input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
                attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)

                if len(tr_dict['token_type_ids']) > 0:
                    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
                    encbert = encoder.bert([input_ids, attention_mask, token_type_ids])

                    if w_lex:
                        inps = [input_ids, token_type_ids, attention_mask, lex_inp]
                    else:
                        inps = [input_ids, token_type_ids, attention_mask]
                else:
                    encbert = encoder.bert([input_ids, attention_mask])     # [1]

                    if w_lex:
                        inps = [input_ids, attention_mask, lex_inp]
                    else:
                        inps = [input_ids, attention_mask]

                embedding = encbert[1]
                dropout = layers.Dropout(config.hidden_dropout_prob)(embedding)   # embedding

                if w_lex:
                    # clf_bert = layers.Dense(2, name='clf_bert', activation='softmax',
                    #                         kernel_initializer=keras.initializers.TruncatedNormal(
                    #                             stddev=config.initializer_range)
                    #                         )(embedding)  # embedding  dropout
                    #
                    # regr_bert = layers.Dense(1, name='regr_bert',
                    #                          kernel_initializer=keras.initializers.TruncatedNormal(
                    #                              stddev=config.initializer_range)
                    #                          )(embedding)  # embedding  dropout
                    #
                    # clf_lex = layers.Dense(2, name='clf_lex', activation='softmax',
                    #                        kernel_initializer=keras.initializers.TruncatedNormal(
                    #                            stddev=config.initializer_range)
                    #                        )(lex_dense)  # embedding  dropout
                    #
                    # regr_lex = layers.Dense(1, name='regr_lex',
                    #                         kernel_initializer=keras.initializers.TruncatedNormal(
                    #                             stddev=config.initializer_range)
                    #                         )(lex_dense)  # embedding  dropout
                    #
                    # clf = layers.Average(name='clf')([clf_bert, clf_lex])
                    # regr = layers.Average(name='regr')([regr_bert, regr_lex])

                    bl_clf = layers.concatenate([embedding, lex_inp])
                    bl_regr = layers.concatenate([embedding, lex_inp])

                    clf = layers.Dense(2, name='clf', activation='softmax',
                                       kernel_initializer=keras.initializers.TruncatedNormal(
                                           stddev=config.initializer_range)
                                       )(bl_clf)  # embedding  dropout

                    regr = layers.Dense(1, name='regr',
                                        kernel_initializer=keras.initializers.TruncatedNormal(
                                            stddev=config.initializer_range)
                                        )(bl_regr)  # embedding  dropout
                else:
                    clf = layers.Dense(num_classes, name='clf', activation='softmax',
                                       kernel_initializer=keras.initializers.TruncatedNormal(
                                           stddev=config.initializer_range)
                                       )(dropout)  # embedding  dropout

                    modout = [clf]
                    lossdef = {"clf": keras.losses.CategoricalCrossentropy()}
                    losswdef = [1.0]
                    metricsdef = {"clf": keras.metrics.CategoricalAccuracy()}

                    if tr_dict['sims'].shape[0] > 0:
                        regr = layers.Dense(1, name='regr',
                                            kernel_initializer=keras.initializers.TruncatedNormal(
                                                stddev=config.initializer_range)
                                            )(dropout)  # embedding  dropout
                        modout.append(regr)
                        lossdef["regr"] = keras.losses.MeanAbsoluteError()
                        losswdef.append(1.0)
                        metricsdef["regr"] = keras.metrics.MeanSquaredError()

                model = keras.Model(
                    inputs=inps,
                    outputs=modout,
                )

                model.compile(optimizer=keras.optimizers.Adam(lr=3e-5),
                              loss=lossdef,
                              loss_weights=losswdef,
                              metrics=metricsdef
                              )

                model.summary()

                if len(modout) > 1:
                    model.fit(tr_x, [tr_dict["ents"], tr_dict["sims"]], epochs=numepochs, batch_size=batchsize,
                              validation_data=(dev_x, [dev_dict["ents"], dev_dict["sims"]]))
                else:
                    model.fit(tr_x, [tr_dict["ents"]], epochs=numepochs, batch_size=batchsize,
                              validation_data=(dev_x, [dev_dict["ents"]]))

        predictions = model.predict(te_x)

        if len(sys.argv) > 1 and \
                os.path.isdir(modlsavedir) and \
                not os.path.isfile(modlsavedir + outfile):
            pickle.dump(predictions, open(modlsavedir + outfile, "wb"), protocol=2)
            print('# saved predictions ' + modlsavedir)

    print(outfile)
    model_eval(predictions, te_dict["sims"], te_dict["ents"], multiclass=multiclass)

    # if usecache and len(os.listdir(modlsavepath)) == 0:
    #     model.save(modlsavepath, save_format='tf')
    #     print('# saved model ' + modlsavepath)
