from typing import List
from unittest.mock import MagicMock

from feast.infra.offline_stores.contrib.trino_offline_store.trino import (
    TrinoRetrievalJob,
)
from feast.infra.offline_stores.contrib.trino_offline_store.trino_source import (
    SavedDatasetTrinoStorage,
)

DESTINATION = "memory.feast.persisted"


def _persist_queries(allow_overwrite: bool) -> List[str]:
    client = MagicMock()
    job = TrinoRetrievalJob(
        query="SELECT 1",
        client=client,
        config=MagicMock(),
        full_feature_names=False,
    )
    job.persist(
        storage=SavedDatasetTrinoStorage(table=DESTINATION),
        allow_overwrite=allow_overwrite,
    )
    return [call.kwargs["query_text"] for call in client.execute_query.call_args_list]


def test_persist_without_overwrite_only_creates_table() -> None:
    assert _persist_queries(allow_overwrite=False) == [
        f"CREATE TABLE {DESTINATION} AS (SELECT 1)"
    ]


def test_persist_with_overwrite_drops_existing_table_first() -> None:
    assert _persist_queries(allow_overwrite=True) == [
        f"DROP TABLE IF EXISTS {DESTINATION}",
        f"CREATE TABLE {DESTINATION} AS (SELECT 1)",
    ]
