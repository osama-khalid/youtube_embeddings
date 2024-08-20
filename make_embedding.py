#qlogin -q UI-GPU-HM -l gpu=true
import pickle as pkl
import tqdm
import numpy as np
import dask.array as da




def youtube_embedding(TRACES,DIMS=100,C=1):
    traces = pkl.load(open(TRACES, 'rb'))



    output_keys = {}
    d = 0
    for item in traces[:int(len(traces)*C)]:
        K = list(item.keys())
        key = K[0]
        vid = item[key]
        for item_ in vid:
            if item_ not in output_keys:
                output_keys[item_] = d
                d += 1

    output_vecs = {}
    for item in tqdm.tqdm(traces[:int(len(traces)*C)]):
        K = list(item.keys())
        input_key = K[0]
        output_vecs[input_key] = np.zeros(len(output_keys))
        for item_ in vid:

            output_key = item_
            output_vecs[input_key][output_keys[output_key]] += 1
    output_keys = {}
    output_probs = {}

    KEYS = []
    for key in tqdm.tqdm(output_vecs):
        if np.sum(output_vecs[key])>0:
            KEYS.append(key)
            output_probs[key] = output_vecs[key]/np.sum(output_vecs[key])        

    Y_vecs = []
    for key in tqdm.tqdm(output_probs):
        Y_vecs.append(output_probs[key])


    Y_vec = da.array(Y_vecs)
    u, s, v = da.linalg.svd_compressed(Y_vec, k=DIMS)
    u_numpy = u.compute()
    output_vector = {}
    for i in range(0,len(KEYS)):
        output_vector[KEYS[i]] = u_numpy[i]

    return(output_vector)

FILES = ['liberal_traces.pkl','conservative_traces.pkl']
for FILE in FILES:
    HEAD = FILE.split('_')[0]
    vectors = youtube_embedding(FILE)

    output_file = HEAD+'_embedding.pkl'
    with open(output_file, 'wb') as f:
        pkl.dump(vectors, f)