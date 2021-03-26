import sys

import glow
from glow import *
from glow.wgr import *
from glow.wgr.functions import *
from glow.wgr.linear_model import *
from glow import gwas
import json

import numpy as np
import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

from matplotlib import pyplot as plt
from bioinfokit import visuz

import builtins

import types
from pyspark.mllib.common import _py2java, _java2py

import time

def pydataframe(self,qry,schema):
    sc = spark.sparkContext
    jschema = spark._jvm.org.apache.spark.sql.types.StructType.fromJson(schema.json())
    return _java2py(sc,self.dataframe(qry,jschema))
    
def gor(self,qry):
    sc = spark.sparkContext
    df = _py2java(sc,self)
    ReflectionUtil = spark._jvm.py4j.reflection.ReflectionUtil
    Rowclass = ReflectionUtil.classForName("org.apache.spark.sql.Row")
    ct = spark._jvm.scala.reflect.ClassTag.apply(Rowclass)
    gds = spark._jvm.org.gorpipe.spark.GorDatasetFunctions(df,ct,ct)
    return _java2py(sc,gds.gor(qry,True,sgs))

def createGorSession(self):
    sgs = self._jvm.org.gorpipe.spark.SparkGOR.createSession(self._jsparkSession)
    sgs.pydataframe = types.MethodType(pydataframe,sgs)
    return sgs

def createGorSessionWOptions(self,gorproject,cachedir,config,alias):
    sgs = self._jvm.org.gorpipe.spark.SparkGOR.createSession(self._jsparkSession,gorproject,cachedir,config,alias)
    sgs.pydataframe = types.MethodType(pydataframe,sgs)
    return sgs

if __name__ == "__main__":
    root = sys.argv[1]
    freeze = sys.argv[2][1:-1]+"/"
    vars = sys.argv[3]
    pheno = sys.argv[4]
    covar = sys.argv[5]
    split = sys.argv[6]
    jobname = sys.argv[7]
    varspb = sys.argv[8]
    samplebc = sys.argv[9]

    spark = SparkSession\
        .builder\
        .appName(jobname)\
        .getOrCreate()

    glow.register(spark, False)
    spark.udf.registerJavaFunction("chartodoublearray", "org.gorpipe.spark.udfs.CharToDoubleArray", ArrayType(DoubleType()))

    rootfreeze = root+freeze

    label_df = pd.read_csv(root+pheno, sep='\t', index_col=0)
    covariate_df = None
    if len(covar)>0:
        covariates = pd.read_csv(root+covar, sep='\t', index_col=0)
        covariate_df = covariates.fillna(covariates.mean())
        covariate_df = (covariate_df - covariate_df.mean()) / covariate_df.std()
        covariate_df

    setattr(DataFrame, 'gor', gor)
    setattr(SparkSession, 'createGorSession', createGorSession)
    setattr(SparkSession, 'createGorSessionWOptions', createGorSessionWOptions)

    schema = StructType([StructField("contigName",StringType(),True),StructField("start",IntegerType(),True),StructField("names",StringType(),True),StructField("referenceAllele",StringType(),True),StructField("alternateAlleles",StringType(),True),StructField("values",StringType(),True)])
    gs = spark.createGorSession()

    gorquery = ""
    if len(split)>0:
        gorquery = "pgor -split <(gor "+root+split+") "+rootfreeze+"variants.gord | varjoin -i "+root+vars+" | calc names chrom+'-'+pos+'-'+ref+'-'+alt | csvsel "+rootfreeze+"buckets.tsv <(nor "+root+pheno+" | select 1 | replace pn substr(pn,1,10)) -u 3 -gc names,ref,alt -vs 1"
    else:
        gorquery = "pgor "+rootfreeze+"variants.gord | varjoin -i "+root+vars+" | calc names chrom+'-'+pos+'-'+ref+'-'+alt | csvsel "+rootfreeze+"buckets.tsv <(nor "+root+pheno+" | select 1 | replace pn substr(pn,1,10)) -u 3 -gc names,ref,alt -vs 1"

    variants = gs.pydataframe(gorquery,schema)
    variants = variants.withColumn("values",expr("chartodoublearray(values)"))
    variant_dfm = variants.withColumn('values',mean_substitute(col('values'))).filter(size(array_distinct('values')) > 1)

    sample_ids = label_df.index.tolist()
    variants_per_block = int(varspb)
    sample_block_count = int(samplebc)
    block_df, sample_blocks = block_variants_and_samples(variant_dfm, sample_ids, variants_per_block, sample_block_count)

    sample_blocks_path = root+"sample_blocks"
    block_matrix_path = root+"block_matrix"
    with open(sample_blocks_path, 'w') as f:
        json.dump(sample_blocks, f)
    block_df.write.mode('overwrite').save(block_matrix_path)

    block_df = spark.read.load(block_matrix_path)
    with open(sample_blocks_path, 'r') as f:
        sample_blocks = json.load(f)

    def chunk_columns(df, chunk_size):
        for start in range(0, df.shape[1], chunk_size):
            chunk = df.iloc[:, range(start, builtins.min(start + chunk_size, df.shape[1]))]
            yield chunk

    chunk_size = 10
    loco_estimates = []
    if covariate_df is None:
        for label_df_chunk in chunk_columns(label_df, chunk_size):
            loco_estimates.append(estimate_loco_offsets(block_df, label_df_chunk, sample_blocks))
    else:
        for label_df_chunk in chunk_columns(label_df, chunk_size):
            loco_estimates.append(estimate_loco_offsets(block_df, label_df_chunk, sample_blocks, covariate_df))
    
    all_traits_loco_df = pd.concat(loco_estimates, axis='columns')
    all_traits_loco_df.to_csv(root+"user_data/"+jobname)

    spark.stop()
