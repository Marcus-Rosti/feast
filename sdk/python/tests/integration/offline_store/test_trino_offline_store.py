from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
import pyarrow as pa
import pytest
from testcontainers.core.container import DockerContainer

from feast import Entity, FeatureStore, FeatureView, Field
from feast.infra.offline_stores.contrib.trino_offline_store.connectors.upload import (
    upload_pandas_dataframe_to_trino,
)
from feast.infra.offline_stores.contrib.trino_offline_store.tests.data_source import (
    trino_container,  # noqa: F401
)
from feast.infra.offline_stores.contrib.trino_offline_store.trino import (
    TrinoOfflineStore,
    TrinoOfflineStoreConfig,
    TrinoRetrievalJob,
)
from feast.infra.offline_stores.contrib.trino_offline_store.trino_queries import Trino
from feast.infra.offline_stores.contrib.trino_offline_store.trino_source import (
    SavedDatasetTrinoStorage,
    TrinoSource,
)
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.monitoring.monitoring_service import MonitoringService
from feast.monitoring.monitoring_utils import (
    MON_TABLE_FEATURE,
    MON_TABLE_FEATURE_SERVICE,
    MON_TABLE_FEATURE_VIEW,
    MON_TABLE_JOB,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.repo_config import RepoConfig
from feast.types import Float32, Float64, Int32, Int64, String
from feast.value_type import ValueType


@pytest.fixture(scope="module")
def trino_server_info(trino_container: DockerContainer) -> Dict[str, Any]:  # noqa: F811
    host = trino_container.get_container_host_ip()
    port = int(trino_container.get_exposed_port("8080"))
    return {
        "host": host,
        "port": port,
        "catalog": "memory",
        "dataset": "feast_integration",
        "user": "test_user",
    }


@pytest.fixture(scope="module")
def live_trino_client(trino_server_info: Dict[str, Any]) -> Trino:
    client = Trino(
        host=trino_server_info["host"],
        port=trino_server_info["port"],
        user=trino_server_info["user"],
        catalog=trino_server_info["catalog"],
        source="feast-integration-tests",
        http_scheme="http",
        verify=False,
        extra_credential=None,
        auth=None,
    )
    client.execute_query(
        f"CREATE SCHEMA IF NOT EXISTS {trino_server_info['catalog']}.{trino_server_info['dataset']}"
    )
    return client


@pytest.fixture(scope="module")
def repo_config(
    trino_server_info: Dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> RepoConfig:
    repo_path = tmp_path_factory.mktemp("trino_repo")
    return RepoConfig(
        project="test_project",
        registry=str(repo_path / "registry.db"),
        provider="local",
        online_store=SqliteOnlineStoreConfig(path=str(repo_path / "online.db")),
        offline_store=TrinoOfflineStoreConfig(
            host=trino_server_info["host"],
            port=trino_server_info["port"],
            catalog=trino_server_info["catalog"],
            dataset=trino_server_info["dataset"],
            connector={"type": "memory"},
            user=trino_server_info["user"],
        ),
    )


@pytest.fixture(scope="module")
def driver_stats_source(
    trino_server_info: Dict[str, Any], live_trino_client: Trino
) -> TrinoSource:
    catalog = trino_server_info["catalog"]
    dataset = trino_server_info["dataset"]
    table_name = f"{catalog}.{dataset}.driver_stats"

    live_trino_client.execute_query(f"DROP TABLE IF EXISTS {table_name}")

    df = pd.DataFrame(
        {
            "driver_id": [1001, 1001, 1002, 1003],
            "conv_rate": [0.2, 0.8, 0.5, None],
            "acc_rate": [0.3, 0.9, 0.6, None],
            "city": ["NYC", "LA", "NYC", None],
            "event_timestamp": [
                datetime(2025, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 14, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 12, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 13, 10, 0, 0, tzinfo=timezone.utc),
            ],
            "created_timestamp": [
                datetime(2025, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 14, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 12, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 1, 13, 10, 0, 0, tzinfo=timezone.utc),
            ],
        }
    )
    upload_pandas_dataframe_to_trino(
        client=live_trino_client,
        df=df,
        table=table_name,
        connector_args={"type": "memory"},
    )

    return TrinoSource(
        name="driver_stats_source",
        table=table_name,
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )


@pytest.fixture
def driver_entity() -> Entity:
    return Entity(
        name="driver_id",
        value_type=ValueType.INT64,
        join_keys=["driver_id"],
    )


@pytest.fixture
def driver_feature_view(
    driver_entity: Entity, driver_stats_source: TrinoSource
) -> FeatureView:
    return FeatureView(
        name="driver_stats",
        entities=[driver_entity],
        ttl=timedelta(days=7),
        schema=[
            Field(name="conv_rate", dtype=Float32),
            Field(name="acc_rate", dtype=Float32),
            Field(name="city", dtype=String),
        ],
        source=driver_stats_source,
    )


@pytest.mark.integration
class TestTrinoOfflineStoreIntegration:
    """Integration tests for TrinoOfflineStore core retrieval and query execution."""

    def test_historical_retrieval_pit(
        self,
        repo_config: RepoConfig,
        driver_entity: Entity,
        driver_feature_view: FeatureView,
    ) -> None:
        """Point-in-time correctness of historical retrieval to DataFrame and PyArrow."""
        store = FeatureStore(config=repo_config)
        store.apply([driver_entity, driver_feature_view])

        entity_df = pd.DataFrame(
            {
                "driver_id": [1001, 1001, 1002],
                "event_timestamp": [
                    datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                    datetime(2025, 1, 11, 12, 0, 0, tzinfo=timezone.utc),
                    datetime(2025, 1, 13, 12, 0, 0, tzinfo=timezone.utc),
                ],
            }
        )

        job = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "driver_stats:conv_rate",
                "driver_stats:acc_rate",
                "driver_stats:city",
            ],
        )

        # 1. Verify Pandas conversion and PIT matches
        df = job.to_df()
        assert len(df) == 3
        df["event_timestamp"] = df["event_timestamp"].dt.tz_localize(None)

        # Row at 2025-01-11 gets 2025-01-10 observation
        ts_jan11 = pd.Timestamp("2025-01-11 12:00:00")
        r_jan11 = df[df["event_timestamp"] == ts_jan11].iloc[0]
        assert r_jan11["driver_id"] == 1001
        assert pytest.approx(r_jan11["conv_rate"], rel=1e-2) == 0.2
        assert pytest.approx(r_jan11["acc_rate"], rel=1e-2) == 0.3
        assert r_jan11["city"] == "NYC"

        # Row at 2025-01-13 gets 2025-01-12 observation
        ts_jan13 = pd.Timestamp("2025-01-13 12:00:00")
        r_jan13 = df[df["event_timestamp"] == ts_jan13].iloc[0]
        assert r_jan13["driver_id"] == 1002
        assert pytest.approx(r_jan13["conv_rate"], rel=1e-2) == 0.5
        assert pytest.approx(r_jan13["acc_rate"], rel=1e-2) == 0.6
        assert r_jan13["city"] == "NYC"

        # Row at 2025-01-15 gets 2025-01-14 observation
        ts_jan15 = pd.Timestamp("2025-01-15 12:00:00")
        r_jan15 = df[df["event_timestamp"] == ts_jan15].iloc[0]
        assert r_jan15["driver_id"] == 1001
        assert pytest.approx(r_jan15["conv_rate"], rel=1e-2) == 0.8
        assert pytest.approx(r_jan15["acc_rate"], rel=1e-2) == 0.9
        assert r_jan15["city"] == "LA"

        # 2. Verify PyArrow Table conversion on fresh retrieval job
        job_arrow = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "driver_stats:conv_rate",
                "driver_stats:acc_rate",
                "driver_stats:city",
            ],
        )
        arrow_table = job_arrow.to_arrow()
        assert isinstance(arrow_table, pa.Table)
        assert arrow_table.num_rows == 3
        assert "conv_rate" in arrow_table.column_names
        assert "city" in arrow_table.column_names

    def test_pull_latest_from_table_or_query(
        self,
        repo_config: RepoConfig,
        driver_stats_source: TrinoSource,
    ) -> None:
        """Verifies windowed latest-record retrieval partitioned by entity key."""
        job = TrinoOfflineStore.pull_latest_from_table_or_query(
            config=repo_config,
            data_source=driver_stats_source,
            join_key_columns=["driver_id"],
            feature_name_columns=["conv_rate", "acc_rate", "city"],
            timestamp_field="event_timestamp",
            created_timestamp_column="created_timestamp",
            start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 16, 0, 0, 0, tzinfo=timezone.utc),
        )
        df = job.to_df().sort_values("driver_id").reset_index(drop=True)
        assert len(df) == 3

        # Driver 1001 should have the newer record from Jan 14 (conv_rate=0.8, city=LA)
        d1001 = df[df["driver_id"] == 1001].iloc[0]
        assert pytest.approx(d1001["conv_rate"], rel=1e-2) == 0.8
        assert d1001["city"] == "LA"

    def test_pull_all_from_table_or_query(
        self,
        repo_config: RepoConfig,
        driver_stats_source: TrinoSource,
    ) -> None:
        """Verifies pulling all records within a timestamp window."""
        job = TrinoOfflineStore.pull_all_from_table_or_query(
            config=repo_config,
            data_source=driver_stats_source,
            join_key_columns=["driver_id"],
            feature_name_columns=["conv_rate", "acc_rate"],
            timestamp_field="event_timestamp",
            start_date=datetime(2025, 1, 11, 0, 0, 0, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
        )
        df = job.to_df()
        # Records between Jan 11 and Jan 15: 1001 (Jan 14), 1002 (Jan 12), 1003 (Jan 13)
        assert len(df) == 3
        assert set(df["driver_id"]) == {1001, 1002, 1003}

    def test_persist_to_trino(
        self,
        repo_config: RepoConfig,
        driver_entity: Entity,
        driver_feature_view: FeatureView,
        trino_server_info: Dict[str, Any],
        live_trino_client: Trino,
    ) -> None:
        """Verifies persisting historical retrieval results into a Trino destination table."""
        store = FeatureStore(config=repo_config)
        store.apply([driver_entity, driver_feature_view])

        entity_df = pd.DataFrame(
            {
                "driver_id": [1001, 1002],
                "event_timestamp": [
                    datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                    datetime(2025, 1, 13, 12, 0, 0, tzinfo=timezone.utc),
                ],
            }
        )

        job = store.get_historical_features(
            entity_df=entity_df,
            features=["driver_stats:conv_rate"],
        )
        assert isinstance(job, TrinoRetrievalJob)

        dest_table = f"{trino_server_info['catalog']}.{trino_server_info['dataset']}.persisted_historical"
        live_trino_client.execute_query(f"DROP TABLE IF EXISTS {dest_table}")

        # Test to_trino direct export
        created_table = job.to_trino(destination_table=dest_table)
        assert created_table == dest_table

        query_res = live_trino_client.execute_query(
            f"SELECT COUNT(*) FROM {dest_table}"
        )
        assert query_res.data[0][0] == 2

        # Test persist with SavedDatasetTrinoStorage
        dest_storage = f"{trino_server_info['catalog']}.{trino_server_info['dataset']}.persisted_storage"
        live_trino_client.execute_query(f"DROP TABLE IF EXISTS {dest_storage}")

        job2 = store.get_historical_features(
            entity_df=entity_df,
            features=["driver_stats:conv_rate"],
        )
        job2.persist(storage=SavedDatasetTrinoStorage(table=dest_storage))

        query_res2 = live_trino_client.execute_query(
            f"SELECT COUNT(*) FROM {dest_storage}"
        )
        assert query_res2.data[0][0] == 2

        # Persisting to an existing table fails unless overwriting is allowed
        with pytest.raises(Exception):
            store.get_historical_features(
                entity_df=entity_df,
                features=["driver_stats:conv_rate"],
            ).persist(storage=SavedDatasetTrinoStorage(table=dest_storage))

        store.get_historical_features(
            entity_df=entity_df,
            features=["driver_stats:conv_rate"],
        ).persist(
            storage=SavedDatasetTrinoStorage(table=dest_storage),
            allow_overwrite=True,
        )

        query_res3 = live_trino_client.execute_query(
            f"SELECT COUNT(*) FROM {dest_storage}"
        )
        assert query_res3.data[0][0] == 2

    def test_historical_retrieval_query_entity_df(
        self,
        repo_config: RepoConfig,
        driver_entity: Entity,
        driver_feature_view: FeatureView,
        trino_server_info: Dict[str, Any],
    ) -> None:
        """Verifies historical retrieval when entity_df is supplied as a SQL query string."""
        store = FeatureStore(config=repo_config)
        store.apply([driver_entity, driver_feature_view])

        catalog = trino_server_info["catalog"]
        dataset = trino_server_info["dataset"]
        entity_query = f"SELECT driver_id, event_timestamp FROM {catalog}.{dataset}.driver_stats WHERE driver_id = 1001"

        job = store.get_historical_features(
            entity_df=entity_query,
            features=["driver_stats:conv_rate"],
        )
        assert isinstance(job, TrinoRetrievalJob)
        df = job.to_df()
        assert len(df) == 2
        assert "conv_rate" in df.columns
        assert set(df["driver_id"]) == {1001}

    def test_historical_retrieval_with_on_demand_feature_view(
        self,
        repo_config: RepoConfig,
        driver_entity: Entity,
        driver_feature_view: FeatureView,
    ) -> None:
        """Verifies on-demand feature view transformations computed over Trino features."""

        @on_demand_feature_view(
            sources=[driver_feature_view],
            schema=[Field(name="conv_rate_doubled", dtype=Float64)],
            mode="pandas",
        )
        def conv_rate_doubled_fv(inputs: pd.DataFrame) -> pd.DataFrame:
            df = pd.DataFrame()
            df["conv_rate_doubled"] = inputs["conv_rate"] * 2.0
            return df

        store = FeatureStore(config=repo_config)
        store.apply([driver_entity, driver_feature_view, conv_rate_doubled_fv])

        entity_df = pd.DataFrame(
            {
                "driver_id": [1001],
                "event_timestamp": [
                    datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                ],
            }
        )

        job = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "driver_stats:conv_rate",
                "conv_rate_doubled_fv:conv_rate_doubled",
            ],
        )
        df = job.to_df()
        assert len(df) == 1
        assert pytest.approx(df["conv_rate"].iloc[0], rel=1e-2) == 0.8
        assert pytest.approx(df["conv_rate_doubled"].iloc[0], rel=1e-2) == 1.6

    def test_historical_retrieval_types_roundtrip(
        self,
        repo_config: RepoConfig,
        driver_entity: Entity,
        trino_server_info: Dict[str, Any],
        live_trino_client: Trino,
    ) -> None:
        """Verifies end-to-end type mapping for Int32, Int64, Float32, Float64, and String."""
        catalog = trino_server_info["catalog"]
        dataset = trino_server_info["dataset"]
        types_table = f"{catalog}.{dataset}.offline_types_roundtrip"

        live_trino_client.execute_query(f"DROP TABLE IF EXISTS {types_table}")

        types_df = pd.DataFrame(
            {
                "driver_id": [5001],
                "event_timestamp": [
                    datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
                ],
                "created_timestamp": [
                    datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
                ],
                "int32_col": [12345],
                "int64_col": [9223372036854775800],
                "float32_col": [1.5],
                "float64_col": [2.718281828459],
                "str_col": ["trino_types_ok"],
            }
        )
        upload_pandas_dataframe_to_trino(
            client=live_trino_client,
            df=types_df,
            table=types_table,
            connector_args={"type": "memory"},
        )

        types_source = TrinoSource(
            name="types_source",
            table=types_table,
            timestamp_field="event_timestamp",
            created_timestamp_column="created_timestamp",
        )
        types_fv = FeatureView(
            name="types_fv",
            entities=[driver_entity],
            ttl=timedelta(days=7),
            schema=[
                Field(name="int32_col", dtype=Int32),
                Field(name="int64_col", dtype=Int64),
                Field(name="float32_col", dtype=Float32),
                Field(name="float64_col", dtype=Float64),
                Field(name="str_col", dtype=String),
            ],
            source=types_source,
        )

        store = FeatureStore(config=repo_config)
        store.apply([driver_entity, types_fv])

        entity_df = pd.DataFrame(
            {
                "driver_id": [5001],
                "event_timestamp": [
                    datetime(2025, 1, 16, 12, 0, 0, tzinfo=timezone.utc)
                ],
            }
        )
        job = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "types_fv:int32_col",
                "types_fv:int64_col",
                "types_fv:float32_col",
                "types_fv:float64_col",
                "types_fv:str_col",
            ],
        )
        res_df = job.to_df()
        assert res_df["int32_col"].iloc[0] == 12345
        assert res_df["int64_col"].iloc[0] == 9223372036854775800
        assert pytest.approx(res_df["float32_col"].iloc[0], rel=1e-3) == 1.5
        assert pytest.approx(res_df["float64_col"].iloc[0], rel=1e-5) == 2.718281828459
        assert res_df["str_col"].iloc[0] == "trino_types_ok"

    def test_historical_retrieval_full_feature_names_and_to_sql(
        self,
        repo_config: RepoConfig,
        driver_entity: Entity,
        driver_feature_view: FeatureView,
    ) -> None:
        """Verifies full_feature_names prefixing and to_sql query generation."""
        store = FeatureStore(config=repo_config)
        store.apply([driver_entity, driver_feature_view])

        entity_df = pd.DataFrame(
            {
                "driver_id": [1001],
                "event_timestamp": [
                    datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                ],
            }
        )

        job = store.get_historical_features(
            entity_df=entity_df,
            features=["driver_stats:conv_rate"],
            full_feature_names=True,
        )
        assert isinstance(job, TrinoRetrievalJob)

        sql = job.to_sql()
        assert isinstance(sql, str)
        assert "SELECT" in sql.upper()

        df = job.to_df()
        assert "driver_stats__conv_rate" in df.columns
        assert pytest.approx(df["driver_stats__conv_rate"].iloc[0], rel=1e-2) == 0.8


@pytest.mark.integration
class TestTrinoMonitoringIntegration:
    """Integration tests for TrinoOfflineStore monitoring push-down and lifecycle."""

    def test_ensure_monitoring_tables_idempotency(
        self,
        repo_config: RepoConfig,
        trino_server_info: Dict[str, Any],
        live_trino_client: Trino,
    ) -> None:
        """Verifies creating monitoring DDL tables and that subsequent calls succeed idempotently."""
        TrinoOfflineStore.ensure_monitoring_tables(repo_config)

        res = live_trino_client.execute_query(
            f"SHOW TABLES FROM {trino_server_info['catalog']}.{trino_server_info['dataset']}"
        )
        tables = {row[0] for row in res.data}
        assert MON_TABLE_FEATURE in tables
        assert MON_TABLE_FEATURE_VIEW in tables
        assert MON_TABLE_FEATURE_SERVICE in tables
        assert MON_TABLE_JOB in tables

        # Re-running must be completely idempotent
        TrinoOfflineStore.ensure_monitoring_tables(repo_config)

    def test_get_monitoring_max_timestamp(
        self,
        repo_config: RepoConfig,
        driver_stats_source: TrinoSource,
        trino_server_info: Dict[str, Any],
        live_trino_client: Trino,
    ) -> None:
        """Verifies max timestamp retrieval from populated and empty tables."""
        max_ts = TrinoOfflineStore.get_monitoring_max_timestamp(
            config=repo_config,
            data_source=driver_stats_source,
            timestamp_field="event_timestamp",
        )
        assert max_ts is not None
        assert max_ts.replace(tzinfo=None) == datetime(2025, 1, 14, 10, 0, 0)

        # Empty table test returns None
        empty_table = (
            f"{trino_server_info['catalog']}.{trino_server_info['dataset']}.empty_stats"
        )
        live_trino_client.execute_query(f"DROP TABLE IF EXISTS {empty_table}")
        live_trino_client.execute_query(
            f"CREATE TABLE {empty_table} (event_timestamp TIMESTAMP WITH TIME ZONE)"
        )
        empty_source = TrinoSource(
            name="empty_source",
            table=empty_table,
            timestamp_field="event_timestamp",
        )
        assert (
            TrinoOfflineStore.get_monitoring_max_timestamp(
                config=repo_config,
                data_source=empty_source,
                timestamp_field="event_timestamp",
            )
            is None
        )

    def test_compute_monitoring_metrics(
        self,
        repo_config: RepoConfig,
        driver_stats_source: TrinoSource,
    ) -> None:
        """Verifies Trino SQL push-down metrics computation for numeric and categorical columns."""
        metrics = TrinoOfflineStore.compute_monitoring_metrics(
            config=repo_config,
            data_source=driver_stats_source,
            feature_columns=[
                ("conv_rate", "numeric"),
                ("acc_rate", "numeric"),
                ("city", "categorical"),
            ],
            timestamp_field="event_timestamp",
            histogram_bins=5,
            top_n=5,
        )
        assert len(metrics) == 3

        conv = next(m for m in metrics if m["feature_name"] == "conv_rate")
        assert conv["feature_type"] == "numeric"
        assert conv["row_count"] == 4
        assert conv["null_count"] == 1
        assert pytest.approx(conv["null_rate"], rel=1e-2) == 0.25
        assert pytest.approx(conv["min_val"], rel=1e-2) == 0.2
        assert pytest.approx(conv["max_val"], rel=1e-2) == 0.8
        assert pytest.approx(conv["mean"], rel=1e-2) == 0.5
        assert conv["histogram"] is not None
        assert "bins" in conv["histogram"]
        assert "counts" in conv["histogram"]
        assert len(conv["histogram"]["counts"]) == 5

        city = next(m for m in metrics if m["feature_name"] == "city")
        assert city["feature_type"] == "categorical"
        assert city["row_count"] == 4
        assert city["null_count"] == 1
        assert city["histogram"] is not None
        top_values = {e["value"]: e["count"] for e in city["histogram"]["values"]}
        assert top_values == {"NYC": 2, "LA": 1}
        assert city["histogram"]["unique_count"] == 2

        # Timestamp filtering test
        filtered = TrinoOfflineStore.compute_monitoring_metrics(
            config=repo_config,
            data_source=driver_stats_source,
            feature_columns=[("conv_rate", "numeric")],
            timestamp_field="event_timestamp",
            start_date=datetime(2025, 1, 11, 0, 0, 0, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
        )
        f_conv = filtered[0]
        assert f_conv["row_count"] == 3
        assert f_conv["null_count"] == 1
        assert pytest.approx(f_conv["min_val"], rel=1e-2) == 0.5
        assert pytest.approx(f_conv["max_val"], rel=1e-2) == 0.8

    def test_save_and_query_monitoring_metrics(
        self,
        repo_config: RepoConfig,
    ) -> None:
        """Verifies saving monitoring metrics and querying with filters and deduplication."""
        TrinoOfflineStore.ensure_monitoring_tables(repo_config)
        feature_view_name = f"driver_stats_fv_{uuid.uuid4().hex[:8]}"

        metric_row = {
            "project_id": "test_project",
            "feature_view_name": feature_view_name,
            "feature_name": "conv_rate",
            "feature_type": "numeric",
            "granularity": "daily",
            "metric_date": date(2025, 1, 15),
            "computed_at": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            "is_baseline": True,
            "row_count": 100,
            "null_count": 5,
            "null_rate": 0.05,
            "mean": 0.75,
            "stddev": 0.1,
            "min_val": 0.1,
            "max_val": 0.9,
            "p50": 0.75,
            "p75": 0.85,
            "p90": 0.88,
            "p95": 0.89,
            "p99": 0.90,
            "histogram": json.dumps(
                {"bin_edges": [0.1, 0.5, 0.9], "frequencies": [40, 55]}
            ),
        }

        TrinoOfflineStore.save_monitoring_metrics(
            config=repo_config,
            metric_type="feature",
            metrics=[metric_row],
        )

        res = TrinoOfflineStore.query_monitoring_metrics(
            config=repo_config,
            project="test_project",
            metric_type="feature",
            filters={
                "feature_view_name": feature_view_name,
                "feature_name": "conv_rate",
            },
        )
        assert len(res) == 1
        record = res[0]
        assert record["project_id"] == "test_project"
        assert record["feature_name"] == "conv_rate"
        assert record["row_count"] == 100
        assert record["is_baseline"] is True

        # Deduplication on updated computed_at
        newer_metric_row = {
            **metric_row,
            "computed_at": datetime(2025, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
            "row_count": 150,
        }
        TrinoOfflineStore.save_monitoring_metrics(
            config=repo_config,
            metric_type="feature",
            metrics=[newer_metric_row],
        )

        res_dedup = TrinoOfflineStore.query_monitoring_metrics(
            config=repo_config,
            project="test_project",
            metric_type="feature",
            filters={
                "feature_view_name": feature_view_name,
                "feature_name": "conv_rate",
            },
        )
        assert len(res_dedup) == 1
        assert res_dedup[0]["row_count"] == 150

    def test_clear_monitoring_baseline_rewrite(
        self,
        repo_config: RepoConfig,
    ) -> None:
        """Verifies clearing baseline records via connector rewrite fallback."""
        TrinoOfflineStore.ensure_monitoring_tables(repo_config)
        feature_view_name = f"driver_stats_clear_{uuid.uuid4().hex[:8]}"

        baseline_metric = {
            "project_id": "test_project",
            "feature_view_name": feature_view_name,
            "feature_name": "conv_rate",
            "feature_type": "numeric",
            "granularity": "daily",
            "metric_date": date(2025, 1, 15),
            "computed_at": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            "is_baseline": True,
            "row_count": 50,
            "null_count": 0,
            "null_rate": 0.0,
            "mean": 0.5,
            "stddev": 0.1,
            "min_val": 0.1,
            "max_val": 0.9,
            "p50": 0.5,
            "p75": 0.6,
            "p90": 0.7,
            "p95": 0.8,
            "p99": 0.9,
            "histogram": None,
        }
        TrinoOfflineStore.save_monitoring_metrics(
            config=repo_config,
            metric_type="feature",
            metrics=[baseline_metric],
        )

        # Clear baseline
        TrinoOfflineStore.clear_monitoring_baseline(
            config=repo_config,
            project="test_project",
            feature_view_name=feature_view_name,
            feature_name="conv_rate",
        )

        cleared_res = TrinoOfflineStore.query_monitoring_metrics(
            config=repo_config,
            project="test_project",
            metric_type="feature",
            filters={
                "feature_view_name": feature_view_name,
                "feature_name": "conv_rate",
            },
        )
        assert len(cleared_res) == 1
        assert cleared_res[0]["is_baseline"] is False

    def test_monitoring_service_end_to_end(
        self,
        repo_config: RepoConfig,
        driver_entity: Entity,
        driver_feature_view: FeatureView,
    ) -> None:
        """End-to-end integration with MonitoringService against live Trino."""
        store = FeatureStore(config=repo_config)
        store.apply([driver_entity, driver_feature_view])

        svc = MonitoringService(store)
        result = svc.compute_metrics(
            project="test_project",
            feature_view_name="driver_stats",
            start_date=date(2025, 1, 10),
            end_date=date(2025, 1, 14),
            granularity="daily",
        )
        assert result["status"] == "completed"
        assert result["computed_features"] > 0
        assert result["computed_feature_views"] == 1

        queried = svc.get_feature_metrics(
            project="test_project",
            feature_view_name="driver_stats",
        )
        assert len(queried) > 0
