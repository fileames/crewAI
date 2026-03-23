import json
import os
import uuid

from crewai_tools import OracleVectorSearchConfig, OracleVectorSearchTool
import pytest


ORACLE_USERNAME_ENV = "ORACLE_VECTOR_SEARCH_USERNAME"
ORACLE_PASSWORD_ENV = "ORACLE_VECTOR_SEARCH_PASSWORD"
ORACLE_DSN_ENV = "ORACLE_VECTOR_SEARCH_DSN"


def _oracle_env_config() -> tuple[str, str, str] | None:
    username = os.getenv(ORACLE_USERNAME_ENV, "onnxuser")
    password = os.getenv(ORACLE_PASSWORD_ENV, "onnxuser")
    dsn = os.getenv(ORACLE_DSN_ENV, "100.94.148.194:1550/cdb1_pdb1.regress.rdbms.dev.us.oracle.com")
    if not username or not password or not dsn:
        return None
    return username, password, dsn


def _embed_text(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "oracle" in lowered else 0.0,
        1.0 if "crewai" in lowered else 0.0,
        1.0 if "vector" in lowered else 0.0,
    ]


@pytest.mark.timeout(120)
def test_oracle_vector_search_tool_with_real_connection(
    pytestconfig: pytest.Config,
) -> None:
    creds = _oracle_env_config()
    if creds is None:
        pytest.skip(
            f"Set {ORACLE_USERNAME_ENV}, {ORACLE_PASSWORD_ENV}, and {ORACLE_DSN_ENV} to run Oracle integration tests."
        )

    if getattr(pytestconfig.option, "block_network", False):
        pytest.skip(
            "Network access is blocked by pytest addopts. Re-run this test without --block-network to use a real Oracle connection."
        )

    oracledb = pytest.importorskip("oracledb")
    username, password, dsn = creds
    table_name = f'CREWAI_ORACLE_TOOL_{uuid.uuid4().hex[:12].upper()}'

    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            user=username,
            password=password,
            dsn=dsn,
            table_name=table_name,
            limit=2,
            distance_strategy="COSINE",
        ),
        custom_embedding_fn=_embed_text,
        dimensions=3,
    )

    try:
        tool.create_table()
        inserted_ids = tool.add_texts(
            [
                "CrewAI integrates with Oracle vector search.",
                "This unrelated text is about cooking pasta.",
            ],
            metadatas=[
                {"source": "docs", "topic": "oracle", "priority": 5},
                {"source": "kitchen", "topic": "food", "priority": 1},
            ],
        )

        assert len(inserted_ids) == 2
        assert tool.table_exists() is True
        assert tool.vector_index_exists() is False

        unfiltered_results = json.loads(tool._run(query="oracle vector"))
        assert len(unfiltered_results) >= 1
        assert unfiltered_results[0]["metadata"]["source"] == "docs"
        assert "Oracle vector search" in unfiltered_results[0]["context"]
        assert unfiltered_results[0]["score"] == pytest.approx(
            unfiltered_results[0]["distance"]
        )

        filtered_results = json.loads(
            tool._run(
                query="oracle vector",
                filter_by="source",
                filter_value="docs",
            )
        )
        assert len(filtered_results) >= 1
        assert all(row["metadata"]["source"] == "docs" for row in filtered_results)

        numeric_filter_results = json.loads(
            tool._run(
                query="oracle vector",
                filter_by="priority",
                filter_value=5,
            )
        )
        assert len(numeric_filter_results) >= 1
        assert all(result["metadata"]["priority"] == 5 for result in numeric_filter_results)

        json_filter_results = json.loads(
            tool._run(
                query="oracle vector",
                filters=json.dumps(
                    {
                        "$or": [
                            {"source": "docs"},
                            {"topic": {"$eq": "oracle"}},
                        ]
                    }
                ),
                score_threshold=0.5,
            )
        )
        assert len(json_filter_results) >= 1
        assert all(result["distance"] <= 0.5 for result in json_filter_results)

        created_index_name = tool.create_vector_index()
        assert created_index_name
        assert tool.vector_index_exists() is True
    finally:
        try:
            with tool.client.cursor() as cursor:
                cursor.execute(f'DROP TABLE "{table_name}" PURGE')
            tool.client.commit()
        except Exception:
            pass

        openai_client = getattr(tool, "_openai_client", None)
        if openai_client is not None:
            openai_client.close()

        client = getattr(tool, "client", None)
        if client is not None:
            client.close()
