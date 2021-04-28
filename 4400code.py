import py_entitymatching as em
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ltable = em.read_csv_metadata("ltable.csv", key='id')
rtable = em.read_csv_metadata("rtable.csv", key='id')
train = pd.read_csv("train.csv")
train['_id'] = train.index
clmns = train.columns.tolist()
clmns = clmns[-1:] + clmns[:-1]
train = train[clmns]

def pairs(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    left = ltable.loc[pairs[:, 0], :]
    right = rtable.loc[pairs[:, 1], :]
    left.columns = [col + "_l" for col in left.columns]
    right.columns = [col + "_r" for col in right.columns]
    left.reset_index(inplace=True, drop=True)
    right.reset_index(inplace=True, drop=True)
    LR = pd.concat([left, right], axis=1)
    return LR

tpairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
tdf = pairs(ltable, rtable, tpairs)
tdf['label'] = train['label']
tdf['_id'] = tdf.index
clmns = tdf.columns.tolist()
clmns = clmns[-1:] + clmns[:-1]
tdf = tdf[clmns]
training_labels = train.label.values

em.set_key(tdf, '_id')
em.set_ltable(tdf, ltable)
em.set_rtable(tdf, rtable)
em.set_fk_ltable(tdf, 'id_l')
em.set_fk_rtable(tdf, 'id_r')
em.get_fk_ltable(tdf)
trte = em.split_train_test(tdf, train_proportion = 0.9)
tr = trte['train']
te = trte['test']

brandblock = em.AttrEquivalenceBlocker()
cblock = brandblock.block_tables(ltable, rtable, 'brand', 'brand', ['id', 'title', 'category', 'brand', 'modelno', 'price'], ['id', 'title', 'category', 'brand', 'modelno', 'price'] )

token = em.get_tokenizers_for_matching()
sim = em.get_sim_funs_for_matching()
ratt = em.get_attr_types(rtable)
ratt['title'] = 'str_bt_5w_10w'
blck = em.get_attr_corres(ltable, rtable)
blck['corres'].pop(0)
blck['corres'].pop(1)
blck['corres'].pop(1)
blck['corres'].pop(1)
blck['corres'].pop(1)

fttable = em.get_features(ltable, rtable, em.get_attr_types(ltable), ratt, blck, token, sim)

tv = em.extract_feature_vecs(tr, fttable = fttable)
tv1 = tv.iloc[:, 3:]
tv['label'] = train['label']

tlist = tv1.values.tolist()
rf2.fit(tlist, training_labels[:4500])

cv = em.extract_feature_vecs(cblock, fttable = fttable)
cv1 = cv.iloc[:, 3:]
clist = cv1.values.tolist()
p = rf2.predict(clist)

matches = cblock.loc[p == 1, ["ltable_id", "rtable_id"]]
matches = list(map(tuple, matches.values))
trainpairs = tdf.loc[training_labels == 1, ["id_l", "id_r"]]
trainpairs = set(list(map(tuple, trainpairs.values)))
rpairs = [pair for pair in matches if pair not in trainpairs]
rpairs = np.array(rpairs)
rdf = pd.DataFrame(rpairs, columns=["ltable_id", "rtable_id"])
rdf.to_csv("matches.csv", index=False)