import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as pl
embs = np.load('text_embs.npy')


attacks = ['original_text','ocr','spread_1','spread_3','newsprint','s&p','s&p_0.4','blur_text_5','s&p_text_0.2']


orig = embs[0,:,:]

# others = embs[1:,:,:]





for i in range(1,len(attacks)):
    att_embs = embs[i,:,:]
    dist = pairwise_distances(orig, att_embs, metric='cosine')
    values = np.diagonal(dist)
    list_values = values.tolist()
    fig = pl.hist(list_values)
    pl.title(attacks[i])
    pl.savefig('figures/'+attacks[i]+'.png')
    pl.figure().clear()
    pl.close()
    pl.cla()
    pl.clf()