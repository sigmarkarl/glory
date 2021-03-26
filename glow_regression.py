import sys

import glow
from glow import *
from glow.wgr.functions import *
from glow.wgr.linear_model import *
from glow import gwas

import numpy as np
import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

from matplotlib import pyplot as plt
from bioinfokit import visuz

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

def call_regression(contig, variant_dfm, label_df, covariate_df, binary_offsets):
    #variant_dfm_chr = variant_dfm.where(col('contigName') == contig)
    if covariate_df is None:
        log_reg_df = None
        if binary_offsets is None:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values',
                contigs=[contig]
            )
        else:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                offset_df=binary_offsets,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values',
                contigs=[contig]
            )
        log_reg_df.write.format("csv").option("delimiter","\t").option("header","true").mode("overwrite").save(root+"user_data/"+jobname+"_"+contig)
    else:
        if binary_offsets is None:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                covariates,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values',
                contigs=[contig]
            )
        else:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                covariates,
                offset_df=binary_offsets,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values',
                contigs=[contig]
            )
        log_reg_df.write.format("csv").option("delimiter","\t").option("header",True).mode("overwrite").save(root+"user_data/"+jobname+"_"+contig)

def call_regression_all(variant_dfm, label_df, covariate_df, binary_offsets):
    if covariate_df is None:
        log_reg_df = None
        if binary_offsets is None:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values'
            )
        else:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                offset_df=binary_offsets,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values'
            )
        log_reg_df.write.format("csv").option("delimiter","\t").option("header","true").mode("overwrite").save(root+"user_data/"+jobname)
    else:
        if binary_offsets is None:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                covariates,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values'
            )
        else:
            log_reg_df = gwas.logistic_regression(
                variant_dfm,
                label_df,
                covariates,
                offset_df=binary_offsets,
                correction='approx-firth',
                pvalue_threshold=0.05,
                values_column='values'
            )
        log_reg_df.write.format("csv").option("delimiter","\t").option("header",True).mode("overwrite").save(root+"user_data/"+jobname)

if __name__ == "__main__":
    root = sys.argv[1]
    freeze = sys.argv[2][1:-1]+"/"
    pheno = sys.argv[3]
    covar = sys.argv[4]
    split = sys.argv[5]
    offsets = sys.argv[6]
    jobname = sys.argv[7]
    splitctg = sys.argv[8]
    repart = sys.argv[9]

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

    binary_offsets = None
    if len(offsets)>0:
        binary_offsets = pd.read_csv(root+offsets).set_index(['#pn','contigName'])

    setattr(DataFrame, 'gor', gor)
    setattr(SparkSession, 'createGorSession', createGorSession)
    setattr(SparkSession, 'createGorSessionWOptions', createGorSessionWOptions)

    schema = StructType([StructField("contigName",StringType(),True),StructField("start",IntegerType(),True),StructField("names",StringType(),True),StructField("referenceAllele",StringType(),True),StructField("alternateAlleles",StringType(),True),StructField("values",StringType(),True)])
    gs = spark.createGorSession()
    
    #if len(repart)>0:
    #    variant_dfm = variant_dfm.repartition(int(repart))
    
    #variant_dfm.show()
    #variant_dfm.printSchema()
    #variant_dfm = spark.read.load(root+"user_data/bll_tmp")

    #variant_dfm.write.mode("overwrite").save(root+"user_data/"+jobname+"_tmp")
    #variant_dfm = spark.read.load(root+"user_data/"+jobname+"_tmp")

    if len(splitctg)>0:
        contigs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14',  'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
        for contig in contigs:
            gorquery = "pgor -split <(gor -p "+contig+" "+root+split+") "+rootfreeze+"variants.gord | calc names chrom+'-'+pos+'-'+ref+'-'+alt | csvsel "+rootfreeze+"buckets.tsv <(nor "+root+pheno+" | select 1 | replace pn substr(pn,1,10)) -u 3 -gc names,ref,alt -vs 1"
            #writesplit = "gor -p "+contig+" "+root+split+" | write "+root+split+contig+".gor"
            #gs.stream(writesplit).count()
            #gorquery = "spark -split "+root+split+contig+".gor "+rootfreeze+"variants.gord | calc names chrom+'-'+pos+'-'+ref+'-'+alt | csvsel "+rootfreeze+"buckets.tsv <(nor "+root+pheno+" | select 1 | replace pn substr(pn,1,10)) -u 3 -gc names,ref,alt -vs 1"
            variants = gs.pydataframe(gorquery,schema)
            variants = variants.withColumn("values",expr("chartodoublearray(values)"))
            variant_dfm = variants.withColumn('values',mean_substitute(col('values'))).filter(size(array_distinct('values')) > 1)
            call_regression(contig, variant_dfm, label_df, covariate_df, binary_offsets)
    else:
        gorquery = "pgor -split <(gor "+root+split+") "+rootfreeze+"variants.gord | calc names chrom+'-'+pos+'-'+ref+'-'+alt | csvsel "+rootfreeze+"buckets.tsv <(nor "+root+pheno+" | select 1 | replace pn substr(pn,1,10)) -u 3 -gc names,ref,alt -vs 1"
        variants = gs.pydataframe(gorquery,schema)
        variants = variants.withColumn("values",expr("chartodoublearray(values)"))
        variant_dfm = variants.withColumn('values',mean_substitute(col('values'))).filter(size(array_distinct('values')) > 1)
        call_regression_all(variant_dfm, label_df, covariate_df, binary_offsets)

    spark.stop()
