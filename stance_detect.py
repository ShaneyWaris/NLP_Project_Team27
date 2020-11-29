import dill
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import sh_fs
import sr_fs


def vectorize(text, i, sbert_model, embed) :
    if i%4 == 0 :
        return sbert_model.encode(text).reshape((1,-1))
    elif i%4 == 1 :
        return np.concatenate((sbert_model.encode(text).reshape((1,-1)), sr_fs.getfeaturearray([text]).reshape((1,-1)), sh_fs.get_shaney_features(text).reshape((1,-1))), axis=1)
    elif i%4 == 2 :
        return embed([text]).numpy().reshape((1,-1))
    else :
        return np.concatenate((embed([text]).numpy().reshape((1,-1)), sr_fs.getfeaturearray([text]).reshape((1,-1)), sh_fs.get_shaney_features(text).reshape((1,-1))), axis=1)

def get_stance(headline, body) :

    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    model_path = 'model/'
    models = [('RF LaBSE', 'rf_bert.ml'), ('RF LaBSE+', 'rf_bert_plus.ml'), ('RF UE', 'rf_u.ml'),
              ('RF UE+', 'rf_u_plus.ml'),
              ('XGB LaBSE', 'xgb_bert.ml'), ('XGB LaBSE+', 'xgb_bert_plus.ml'), ('XGB UE', 'xgb_u.ml'),
              ('XGB UE+', 'xgb_u_plus.ml'),
              ('MLP LaBSE', 'mlp_bert.ml'), ('MLP LaBSE+', 'mlp_bert_plus.ml'), ('MLP UE', 'mlp_u.ml'),
              ('MLP UE+', 'mlp_u_plus.ml'),
              ('SVC LaBSE', 'svc_bert.ml'), ('SVC LaBSE+', 'svc_bert_plus.ml'), ('SVC UE', 'svc_u.ml'),
              ('SVC UE+', 'svc_u_plus.ml')]

    model_stance = []
    y_dict = {0 : 'agree', 1 : 'disagree', 2 : 'discuss', 3 : 'unrelated'}

    for i in range(len(models)) :
        try:
            m = dill.load(open(model_path+models[i][1], 'rb'))
            vec = vectorize(headline+body, i, sbert_model, embed)
            p = m.predict(vec.reshape((1,-1)))
            model_stance.append((models[i][0], y_dict[int(p[0])]))

        except EOFError:
            print(models[i][0])

    del embed
    del sbert_model

    return model_stance
