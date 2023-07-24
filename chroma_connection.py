import uuid
from collections import ChainMap
from copy import deepcopy
from typing import Any, Optional, Union

import chromadb
import streamlit as st
from chromadb import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from streamlit.connections import ExperimentalBaseConnection
from streamlit.connections.util import extract_from_dict

_CHROMA_CLIENT_PARAMS = {
    "in-memory": {},
    "client": {"host", "port", "ssl", "headers"},
    "persistent": "path",
}


class ChromaDBConnection(ExperimentalBaseConnection[chromadb.API]):
    def _connect(self, **kwargs) -> chromadb.Client:
        kwargs = deepcopy(kwargs)
        mode = kwargs.get("mode", "in-memory")
        if mode not in _CHROMA_CLIENT_PARAMS:
            raise ValueError(
                f"Mode {mode} not supported. Please choose one of {list(_CHROMA_CLIENT_PARAMS.keys())}"
            )
        client_param_kwargs = extract_from_dict(_CHROMA_CLIENT_PARAMS[mode], kwargs)
        client_param_kwargs = {f"chroma_{k}": v for k, v in client_param_kwargs.items()}
        client_params = ChainMap(client_param_kwargs, self._secrets.to_dict())

        settings_kwargs = ChainMap(kwargs, self._secrets.get("chroma_settings", {}))
        settings = Settings(**settings_kwargs)

        if mode == "client":
            host = client_params.get("chroma_host", "localhost")
            port = client_params.get("chroma_port", "8000")
            ssl = client_params.get("chroma_ssl", False)
            headers = client_params.get("chroma_headers", {})
            return chromadb.HttpClient(
                host=host, port=port, ssl=ssl, headers=headers, settings=settings
            )
        elif mode == "persistent":
            path = client_params.get("chroma_path", "./chroma")
            return chromadb.PersistentClient(path=path, settings=settings)
        else:
            return chromadb.EphemeralClient(settings=settings)

    @property
    def cursor(self):
        return self._instance

    def _get_collection(self, name: str, embedding_function: EmbeddingFunction = None):
        return self.cursor.get_collection(name, embedding_function=embedding_function)

    def create(
        self, name: str, embedding_function: EmbeddingFunction = None, distance_metric: str = "l2"
    ):
        if distance_metric not in ("l2", "ip", "cosine"):
            raise ValueError(
                f"Distance metric {distance_metric} not supported. Please choose one of 'l2', 'ip', 'cosine'"
            )
        self.cursor.create_collection(
            name, embedding_function=embedding_function, metadata={"hnsw:space": distance_metric}
        )

    def drop(self, name: str, embedding_function: EmbeddingFunction = None):
        self.cursor.delete_collection(name, embedding_function)

    def count(self, name: str, embedding_function: EmbeddingFunction = None):
        collection = self._get_collection(name, embedding_function)
        return collection.count()

    def peek(self, name: str, embedding_function: EmbeddingFunction = None):
        collection = self._get_collection(name, embedding_function)
        return collection.count()

    def rename(self, old_name: str, new_name: str, embedding_function: EmbeddingFunction = None):
        collection = self._get_collection(old_name, embedding_function)
        collection.modify(name=new_name)

    def insert(
        self,
        name: str,
        documents: list[str],
        embeddings: Optional[list[Union[float, int]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
        embedding_function: EmbeddingFunction = None,
    ):
        if ids is None:
            ids = [uuid.uuid4().hex for _ in range(len(documents))]
        collection = self._get_collection(name, embedding_function)
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def get(self, name: str, embedding_function: EmbeddingFunction = None, **kwargs):
        collection = self._get_collection(name, embedding_function)
        return collection.get(**kwargs)

    def query(
        self,
        name: str,
        query_vector: list[Union[str, float, int]],
        query_type: str = "text",
        ttl: int = 3600,
        embedding_function: EmbeddingFunction = None,
        **kwargs,
    ) -> dict:
        @st.cache_data(ttl=ttl)
        def _query(query_vector: list[Union[str, float, int]], **kwargs) -> dict:
            collection = self._get_collection(name, embedding_function)
            if query_type == "text":
                return collection.query(query_texts=query_vector, **kwargs)
            elif query_type == "embeddings":
                return collection.query(query_embeddings=query_vector, **kwargs)
            else:
                raise ValueError(
                    f"Query type {query_type} not supported. Please choose one of 'text', 'embeddings'"
                )

        return _query(query_vector, **kwargs)

    def update(
        self,
        name: str,
        ids: list[str],
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[Union[float, int]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        embedding_function: EmbeddingFunction = None,
    ):
        collection = self._get_collection(name, embedding_function)
        collection.update(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def upsert(
        self,
        name: str,
        ids: list[str],
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[Union[float, int]]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
        embedding_function: EmbeddingFunction = None,
    ):
        collection = self._get_collection(name, embedding_function)
        collection.upsert(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def delete(
        self, name: str, ids: list[str], embedding_function: EmbeddingFunction = None, **kwargs
    ):
        collection = self._get_collection(name, embedding_function)
        collection.delete(ids=ids, **kwargs)
