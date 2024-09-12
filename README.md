# Using Ragas for evaluating Retrieval Augmented Generation Chains on Databricks

Notes: 
- Validated with Databricks runtimes: 14.3 ML LTS and 13.3 ML LTS
- This is NOT official Databricks code and should be utilized as reference code. 

## Introduction

Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG denotes a class of LLM applications that use external data to augment the LLM’s context. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard. This is where Ragas (RAG Assessment) comes in.

## Key Features

- **Dataset Generation**: Ragas allows for the creation of diverse and comprehensive datasets that can be used to test the performance of RAG models.
- **Customizable**: Users can tailor the datasets to include specific types of queries and documents, ensuring that the testing scenarios are relevant to their use cases.
- **Integration with Databricks**: Ragas can be seamlessly integrated with Databricks, enabling users to leverage the powerful data processing and machine learning capabilities of the platform.

## Benefits

- **Improved Model Performance**: By using well-constructed testing datasets, users can identify weaknesses in their RAG models and make necessary adjustments to improve performance.
- **Efficiency**: Automating the dataset generation process saves time and resources, allowing data scientists to focus on model development and optimization.
- **Scalability**: Ragas can handle large volumes of data, making it suitable for enterprise-level applications.
