from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import datetime

s = datetime.datetime.now()

data_train_path = "./train.npz"
train_db = np.load("train.npz")
mfcc_train = train_db['mfccs']

model = KMeans(n_clusters=32, init='k-means++',max_iter=300, n_jobs=-1, verbose=1)
model.fit(mfcc_train)

filename = 'kmeans_model_cluster_32.sav'
joblib.dump(model, filename)

e = datetime.datetime.now()
diff = e - s
print("Done. elapsed time:{}s".format(diff.seconds))
