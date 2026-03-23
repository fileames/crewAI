# OracleVectorSearchTool

## Description

This tool is specifically crafted for conducting vector searches within Oracle AI Vector Search tables. Use this tool to find semantically similar documents stored in Oracle Database.

Oracle Database 23ai and later can store vectors natively and query them with `vector_distance(...)`. This tool follows CrewAI's existing vector-search tool shape while using Oracle-native SQL under the hood.

## Installation

Install the CrewAI tools package with Oracle support:

```shell
pip install crewai-tools[oracledb]
```

or

```shell
uv add crewai-tools --extra oracledb
```

## Example

```python
from crewai_tools import (
    OracleVectorSearchConfig,
    OracleVectorSearchQueryConfig,
    OracleVectorSearchTool,
)

tool = OracleVectorSearchTool(
    oracle_config=OracleVectorSearchConfig(
        user="app_user",
        password="secret",
        dsn="dbhost.example.com:1521/freepdb1",
        table_name="DOCS_VECTORS",
        limit=3,
        distance_strategy="COSINE",
    ),
    query_config=OracleVectorSearchQueryConfig(
        score_threshold=0.6,
        filter={"source": "docs"},
    ),
)
```

Running a search with native JSON numeric filtering:

```python
results = tool._run(
    query="oracle vector",
    filter_by="priority",
    filter_value=5,
    score_threshold=0.6,
)
```

Using richer Oracle-style filters:

```python
results = tool._run(
    query="oracle vector",
    filters='{"$or":[{"source":"docs"},{"priority":{"$gte":3}}]}',
)
```

Using a custom embedding function:

```python
tool = OracleVectorSearchTool(
    oracle_config=OracleVectorSearchConfig(
        user="app_user",
        password="secret",
        dsn="dbhost.example.com:1521/freepdb1",
        table_name="DOCS_VECTORS",
    ),
    custom_embedding_fn=my_embedding_function,
)
```

Preloading data into Oracle:

```python
tool.create_table()
tool.add_texts(
    ["CrewAI integrates with Oracle AI Vector Search."],
    metadatas=[{"source": "docs"}],
)
tool.create_vector_index()
```

## Arguments

- `oracle_config`: Oracle connection and search settings. Required.
- `query_config`: Optional default query behavior including `limit`, `score_threshold`, and Oracle-style metadata filters.
- `custom_embedding_fn`: Optional callable or import path used instead of OpenAI embeddings.
- `embedding_model`: OpenAI embedding model used when `custom_embedding_fn` is not provided.
- `dimensions`: Embedding dimension used when creating tables and validating inserted embeddings.

`oracle_config` supports:

- `user`, `password`, `dsn`: Oracle connection fields when a client is not provided.
- `table_name`: Oracle table containing your text, metadata, and vector columns.
- `limit`: Number of search results to return.
- `score_threshold`: Optional maximum vector distance. Only rows with `distance <= score_threshold` are returned.
- `distance_strategy`: One of `COSINE`, `EUCLIDEAN`, or `DOT`.
- `index_name`: Optional default vector index name used by `create_vector_index()`.

The tool creates and expects a fixed Oracle schema:
- `id`
- `text`
- `metadata` as native Oracle `JSON`
- `embedding`

`_run()` also supports:

- `filters`: JSON string for richer Oracle metadata filters such as `{"$or":[{"source":"docs"},{"topic":{"$eq":"oracle"}}]}`
- `limit`: Per-call result limit override
- `score_threshold`: Per-call maximum distance override

Result format:

- `context`: The matched text payload.
- `metadata`: Oracle JSON metadata decoded back into Python values.
- `distance`: Raw Oracle `vector_distance(...)` value.
- `score`: Kept for consistency with other CrewAI vector tools, but currently equal to `distance`.
