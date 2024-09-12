# Databricks notebook source
# MAGIC %pip install langchain langchain_community ragas 

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

from datasets import load_dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
amnesty_qa

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Answer Relevancy**: 
# MAGIC - Answer relevancy measures how well the generated response addresses the user's query or question. It assesses whether the response is on-topic, relevant, and useful. A high answer relevancy score indicates that the model has successfully understood the user's intent and provided a response that is pertinent to the conversation. This metric is often evaluated using human judgments, where annotators rate the response as relevant or not.  
# MAGIC
# MAGIC **Faithfulness**: 
# MAGIC - Faithfulness measures how accurately the generated response reflects the input context or prompt. It assesses whether the model has preserved the essential information, tone, and intent of the original input. A high faithfulness score indicates that the model has successfully captured the essence of the input and generated a response that is consistent with it. This metric is often evaluated using metrics such as ROUGE (Recall-Oriented Understudy for Gisting Evaluation) or METEOR (Metric for Evaluation of Translation with Explicit ORdering).  
# MAGIC
# MAGIC **Context Recall**: 
# MAGIC - Context recall measures the proportion of relevant context information that is retained in the generated response. It assesses how well the model recalls specific details, entities, or events mentioned in the input context. A high context recall score indicates that the model has successfully retained important information from the input context and incorporated it into the response. This metric is often evaluated using metrics such as recall@k, which measures the proportion of relevant context information recalled in the top-k responses.  
# MAGIC
# MAGIC **Context Precision**: 
# MAGIC - Context precision measures the proportion of relevant context information that is accurately represented in the generated response. It assesses how well the model avoids introducing extraneous or incorrect information that is not present in the input context. A high context precision score indicates that the model has successfully avoided "hallucinating" or introducing irrelevant information and has stuck to the facts present in the input context. This metric is often evaluated using metrics such as precision@k, which measures the proportion of accurate context information in the top-k responses.

# COMMAND ----------

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from mlflow.deployments import get_deploy_client

evaluation_llm = ChatDatabricks(
    target_uri="databricks",
    endpoint= "databricks-meta-llama-3-1-70b-instruct",
    temperature=0.8,
)

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint="databricks-gte-large-en")

# COMMAND ----------

amnesty_qa["eval"]

# COMMAND ----------

# DBTITLE 1,Extend timeouts for additional metrics to run
from ragas import RunConfig

run_config = RunConfig()

run_config.timeout = 360
run_config.wait = 360
run_config.thread_timeout = 360

# COMMAND ----------

run_config

# COMMAND ----------

from ragas import evaluate

result = evaluate(
    amnesty_qa["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm = evaluation_llm, 
    embeddings = embeddings, 
    run_config = run_config
)

result

# COMMAND ----------

df = result.to_pandas()
spark_df = spark.createDataFrame(df)
display(spark_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC # Shorter example with default run config: 

# COMMAND ----------

from datasets import Dataset 
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}

dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset,metrics=[faithfulness,answer_correctness], llm = evaluation_llm, embeddings = embeddings)
score.to_pandas()
