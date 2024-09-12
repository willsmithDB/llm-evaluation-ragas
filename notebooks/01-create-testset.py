# Databricks notebook source
# MAGIC %pip install langchain langchain_community ragas 

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://mlflow.org/docs/latest/index.html")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


df = spark.createDataFrame(docs).drop(col("metadata")).withColumn("id", monotonically_increasing_id())

display(df)

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from mlflow.deployments import get_deploy_client

generator_llm = ChatDatabricks(
    target_uri="databricks",
    endpoint= "databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1,
)

# COMMAND ----------

critic_llm = ChatDatabricks(
    target_uri="databricks",
    endpoint= "databricks-meta-llama-3-1-70b-instruct",
    temperature=0.1,
)

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint="databricks-gte-large-en")

# COMMAND ----------

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# COMMAND ----------

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

# COMMAND ----------

import pyspark.pandas as pd 

mlflow_testset_df = spark.createDataFrame(testset.to_pandas())
display(mlflow_testset_df.limit(20))

# COMMAND ----------

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

# COMMAND ----------

from ragas import RunConfig

run_config = RunConfig()

run_config.timeout = 360
run_config.wait = 360
run_config.thread_timeout = 360

# COMMAND ----------

from ragas import evaluate

result = evaluate(
    mlflow_testset_df["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm = generator_llm, 
    embeddings = embeddings, 
    run_config = run_config
)

result

# COMMAND ----------

eval_results = result.to_pandas()
eval_results_df = spark.createDataFrame(eval_results)
display(eval_results_df.limit(20))
