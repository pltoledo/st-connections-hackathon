import uuid
from typing import Any, Optional, Union

import chromadb
import streamlit as st
from chromadb import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from streamlit.connections import ExperimentalBaseConnection


class ChromaDBConnection(ExperimentalBaseConnection[chromadb.API]):
    def _connect(self, **kwargs) -> chromadb.Client:
        # Extract client_type from kwargs or secrets or default to 'ephemeral'
        client_type = kwargs.pop("client_type", self._secrets.get("client_type", "ephemeral"))

        # Create Settings object, check secrets for telemetry settings
        settings = kwargs.pop("settings", Settings())
        if "chorma_telemetry_enabled" in self._secrets:
            settings.anonymized_telemetry = self._secrets["chorma_telemetry_enabled"]

        # Depending on client_type, create and return respective chromadb Client object
        if client_type == "http":
            host = self._secrets.get("chroma_http_host", "localhost")
            port = self._secrets.get("chroma_http_port", "8000")
            ssl = self._secrets.get("chroma_http_ssl", False)
            headers = self._secrets.get("chroma_http_headers", {})
            return chromadb.HttpClient(
                host=host, port=port, ssl=ssl, headers=headers, settings=settings
            )
        elif client_type == "persistent":
            chroma_db_path = kwargs.pop("path", self._secrets.get("chroma_db_path", "./chroma"))
            return chromadb.PersistentClient(path=chroma_db_path, settings=settings)
        else:
            return chromadb.EphemeralClient(settings=settings)

    @property
    def cursor(self):
        return self._instance

    @st.cache_data
    def _get_collection(self, name: str, ebdd_fun: EmbeddingFunction = None):
        return self.cursor.get_collection(name, embedding_function=ebdd_fun)

    def create(self, name: str, ebdd_fun: EmbeddingFunction = None, distance_metric: str = "l2"):
        if distance_metric not in ("l2", "ip", "cosine"):
            raise ValueError(
                f"Distance metric {distance_metric} not supported. Please choose one of 'l2', 'ip', 'cosine'"
            )
        self.cursor.create_collection(
            name, embedding_function=ebdd_fun, metadata={"hnsw:space": distance_metric}
        )

    def drop(self, name: str):
        self.cursor.delete_collection(name)

    def count(self, name: str):
        collection = self._get_collection(name)
        return collection.count()

    def peek(self, name: str):
        collection = self._get_collection(name)
        return collection.count()

    def rename(self, old_name: str, new_name: str):
        collection = self._get_collection(old_name)
        collection.modify(name=new_name)

    def insert(
        self,
        name: str,
        documents: list[str],
        embeddings: Optional[list[Union[float, int]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ):
        if ids is None:
            ids = [uuid.uuid4().hex for _ in range(len(documents))]
        collection = self._get_collection(name)
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def get(self, name: str, ebdd_fun: EmbeddingFunction = None):
        return self._get_collection(name, ebdd_fun)

    def query(self):
        pass

    def update(
        self,
        name: str,
        ids: list[str],
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[Union[float, int]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ):
        collection = self._get_collection(name)
        collection.update(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def upsert(self):
        pass

    def delete(self):
        pass
