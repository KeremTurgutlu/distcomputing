#USER DEFINITIONS
from user_definition import *

################################
file_roots = ['../data/A/C/A']

#PYTHON
import h5py
import json
from collections import defaultdict
import numpy as np
import os

#SPARK
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import Row

#SPARK CONFIGS
conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', executor_mem)
        .set('spark.driver.memory', driver_mem)
        .set('spark.driver.maxResultSize', max_result)
        .set('spark.yarn.executor.memoryOverhead', overhead))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

#PYMONGO
from pymongo import MongoClient

print 'Imports Ready'
print 'Driver Memory', sc._conf.get('spark.driver.memory')
print 'Executor Memory', sc._conf.get('spark.executor.memory')
print 'Max Result Size', sc._conf.get('spark.driver.maxResultSize')
print '\n'
#########################################################
                ###FUNCTIONS###

def songstodict(songs_val):
    songs_fields = songs_val.dtype.fields
    songs_dict = {}
    for v, k in zip(songs_val[0], songs_val.dtype.names):
        if isinstance(v, np.int32):
            v = int(v)
        elif isinstance(v, np.float):
            v = float(v)
        else:
            v = str(v)
        songs_dict[k]= v
    return songs_dict

def h5todict(h5_file_path):
    file = h5py.File(h5_file_path)
    D = {}
    for k in list(file.keys()):
        d = {}
        for c in list(file[k].keys()):
            if c != 'songs':
                val = file[k][c].value
                if isinstance(val, np.ndarray):
                    if len(val) > 0:
                        if isinstance(val[0], bytes):
                            d[c] = [str(e) for e in val.tolist()]
                        else:
                            d[c] = val.tolist()
                    else:
                        d[c] = val.tolist()
                else:
                    d[c] = val
            else:
                d[c] = songstodict(file[k][c].value)
            D[k] = d
    return D


#GET .H5 FILENAMES
h5_file_paths = []
for root in file_roots:
    for path, subdirs, files in os.walk(root):
        for name in files:
            h5_file_paths.append(os.path.join(path, name))
print '{} filenames gathered (PYTHON)'.format(len(h5_file_paths))

#DISTRIBUTED CONVERSION FROM .H5 TO PYTHON DICT WITH PYSPAR
file_paths = sc.parallelize(h5_file_paths)
songs_rdd = file_paths.map(lambda x: h5todict(x))
songs = songs_rdd.collect()
print '{} songs collected (PYSPARK)'.format(len(songs))

#INSERT FILES TO MONGODB
client = MongoClient()
db = client[dbname]
db[collection_name].drop() #drop collection if it was created before
collection=db[collection_name]
# insert songs here
for song in songs:
    collection.insert_one(song)

print '{} songs are inserted to collection: "{}" (MONGODB)'.format(collection.count(), collection_name)












