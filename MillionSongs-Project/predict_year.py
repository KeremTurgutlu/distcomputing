#USER DEFINITIONS
from user_definition import *

#PYTHON
import numpy as np
import gc
import matplotlib.pyplot as plt

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

#SPARK ML
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics

#PYMONGO
from pymongo import MongoClient

print 'Imports Ready'
print 'Driver Memory', sc._conf.get('spark.driver.memory')
print 'Executor Memory', sc._conf.get('spark.executor.memory')
print 'Max Result Size', sc._conf.get('spark.driver.maxResultSize')
print '\n'
#########################################################
                ###FUNCTIONS###

# function to create year prediction fields
def getYearFeatures(d):
    iu1 = np.triu_indices(12) #take upper triangle
    return {'mean_timbre':np.mean(np.array(d['analysis']['segments_timbre']), 0),
           'cov_timbre':np.cov(np.array(d['analysis']['segments_timbre']).T)[iu1],
           'energy':d['analysis']['songs']['energy'],
           'danceability':d['analysis']['songs']['danceability'],
           'year': d['musicbrainz']['songs']['year'],
           'title': d['metadata']['songs']['title'],
           'artist_name': d['metadata']['songs']['artist_name']}


# decades from 1920s to 2010s
decade_tresh = np.array([1920 + i * 10 for i in range((2011 - 1922) % 10 + 1)])


def getDFeatures(d, is_test=False):
    row_dict = {}
    if is_test:
        row_dict['decade'] = d['year']
    else:
        row_dict['decade'] = int(np.argmax(np.where(np.array(decade_tresh) <= d['year'])))
    row_dict['energy'] = d['energy']
    row_dict['danceability'] = d['danceability']
    row_dict['title'] = d['title']
    row_dict['artist_name'] = d['artist_name']
    cov_timbre_dict = d['cov_timbre']
    mean_timbre_dict = d['mean_timbre']
    for i, k in enumerate(cov_timbre_dict):
        row_dict['cov_timbre_{}'.format(i)] = float(k)
    for i, k in enumerate(mean_timbre_dict):
        row_dict['mean_timbre_{}'.format(i)] = float(k)
    return Row(**row_dict)


#########################################################
                ###YEAR PREDICTION###

# filter data for year prediction
client = MongoClient()
db = client[dbname]
collection = db[collection_name]

fields = {'musicbrainz.songs.year':1,
                    'analysis.segments_timbre':1,
                    'analysis.songs.energy':1,
                    'analysis.songs.danceability':1,
                    'metadata.songs.title':1,
                    'metadata.songs.artist_name': 1,
          '_id':0}
train_year_data = collection.find({'musicbrainz.songs.year':{'$gte':1920}}, fields)
test_year_data = collection.find({'musicbrainz.songs.year':{'$eq':0}}, fields)

# create train and test rdds
train_year_rdd = sc.parallelize(list(train_year_data))
test_year_rdd = sc.parallelize(list(test_year_data))

# plot year distribution
#year_counts = train_year_rdd.map(lambda x: (x['musicbrainz']['songs']['year'], 1)).\
#                reduceByKey(lambda x,y: x+y).collect()
#plt.bar([str(i[0]) for i in year_counts], [i[1]for i in year_counts])
#plt.title('Song Release Year Distribution')
#plt.savefig('../plots/year_dist.png')

del train_year_data
del test_year_data
gc.collect()

print 'Train and test rdds ready'
print 'Songs with and missing year data: {}, {}'.format(train_year_rdd.count(), test_year_rdd.count())
print '\n'

# create train and test dataframes
train_full_df = train_year_rdd.map(lambda d: getYearFeatures(d)).map(lambda d: getDFeatures(d)).toDF()
# take sample
test_df = test_year_rdd.map(lambda d: getYearFeatures(d)).map(lambda d: getDFeatures(d, True)).toDF()
del train_year_rdd
del test_year_rdd
gc.collect()
print 'Train and test dataframes are ready'
print '\n'

# split to train and validation dataframes
print 'Creating train and val splits'
train_df, val_df = train_full_df.randomSplit([0.8, 0.2])
# create vector assembler
x_cols = [c for c in train_df.columns if c not in  ['decade', 'title', 'artist_name']]
va = VectorAssembler(inputCols=x_cols, outputCol='features')
train_va = va.transform(train_df).select('features', 'decade').cache()
val_va = va.transform(val_df).select('features', 'decade').cache()
test_va =  va.transform(test_df).select('features', 'decade').cache()
print 'Train {} Val {} Test {}'.format(train_va.count(), val_va.count(), test_va.count())
print '\n'

#########################################################
                ###LOGISTIC REGRESSION###

### PICK BEST MODEL WITH CV ON TRAIN, GET SCORE ON VAL AND IMPUTE MISSING YEAR ON TEST

print 'Modeling with multinomial logistic regression'
print '\n'

k = 5
print 'Doing {} fold cross-validation for param tuning'.format(k)
# create cross validation
lr = LogisticRegression(featuresCol='features', labelCol='decade',maxIter=1000, fitIntercept=True)
evaluator = MulticlassClassificationEvaluator(labelCol='decade', metricName='f1')
cv = CrossValidator().setEstimator(lr).setEvaluator(evaluator).setNumFolds(5)
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [1000]).addGrid(lr.regParam, [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]).build()
cv.setEstimatorParamMaps(paramGrid)
cvmodel = cv.fit(train_va)
best_model = cvmodel.bestModel
print 'Training done'

# Evaluation on Validation Data
val_preds = best_model.transform(val_va)
val_preds = val_preds.withColumn("prediction", val_preds["prediction"].cast(DoubleType()))
val_preds = val_preds.withColumn("decade", val_preds["decade"].cast(DoubleType()))
evaluator = MulticlassClassificationEvaluator(labelCol='decade', metricName='f1')
mm = MulticlassMetrics(val_preds.select('prediction', 'decade').rdd)
print mm.confusionMatrix()
print('VALIDATION F1: {}'.format(evaluator.evaluate(val_preds)))

# Make Prediction for Missing Songs
test_preds = best_model.transform(test_va)
test_preds.show(5)


#########################################################
                ###RANDOM FOREST###

### PICK BEST MODEL WITH CV ON TRAIN, GET SCORE ON VAL AND IMPUTE MISSING YEAR ON TEST


print 'Modeling with random forest classifier'
print '\n'

# do cross validation
k = 5
print 'Doing {} fold cross-validation for param tuning'.format(k)

# Cross Validation Setup
rf = RandomForestClassifier(featuresCol='features', labelCol='decade')
cv = CrossValidator().setEstimator(rf).setEvaluator(evaluator).setNumFolds(k)
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [3,5,10,20,50]).addGrid(rf.maxDepth, [3,5,7,10]).build()
cv.setEstimatorParamMaps(paramGrid)
cvmodel = cv.fit(train_va)
best_model = cvmodel.bestModel
print 'Training done'

# Evaluation on Validation Data
val_preds = best_model.transform(val_va)
val_preds = val_preds.withColumn("prediction", val_preds["prediction"].cast(DoubleType()))
val_preds = val_preds.withColumn("decade", val_preds["decade"].cast(DoubleType()))
evaluator = MulticlassClassificationEvaluator(labelCol='decade', metricName='f1')
mm = MulticlassMetrics(val_preds.select('prediction', 'decade').rdd)
print mm.confusionMatrix()
print('VALIDATION F1: {}'.format(evaluator.evaluate(val_preds)))

# Make Prediction for Missing Songs
test_preds = best_model.transform(test_va)
test_preds.show(5)

# Test song titles
print test_df.select('title', 'artist_name').rdd.take(5)


















