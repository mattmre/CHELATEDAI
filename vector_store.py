"""Vector store boundary abstractions (F-044)."""

from abc import ABC, abstractmethod
from urllib.parse import urlparse

from qdrant_client import QdrantClient


class VectorStore(ABC):
    """Dependency inversion boundary for vector database operations."""

    @abstractmethod
    def collection_exists(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_collection(self, *args, **kwargs):
        pass

    @abstractmethod
    def query_points(self, *args, **kwargs):
        pass

    @abstractmethod
    def retrieve(self, *args, **kwargs):
        pass

    @abstractmethod
    def upsert(self, *args, **kwargs):
        pass

    @abstractmethod
    def scroll(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_collection(self, *args, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_client(self):
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant-backed implementation of VectorStore."""

    def __init__(self, qdrant_location=":memory:", client_cls=QdrantClient):
        if qdrant_location is None:
            raise ValueError("qdrant_location cannot be None")

        if (
            qdrant_location == ":memory:"
            or qdrant_location.startswith("http://")
            or qdrant_location.startswith("https://")
        ):
            if qdrant_location.startswith("http://") or qdrant_location.startswith("https://"):
                parsed = urlparse(qdrant_location)
                if not parsed.hostname:
                    raise ValueError(f"Invalid Qdrant URL: missing hostname in '{qdrant_location}'")
            self._client = client_cls(location=qdrant_location)
        else:
            self._client = client_cls(path=qdrant_location)

    def collection_exists(self, *args, **kwargs):
        return self._client.collection_exists(*args, **kwargs)

    def create_collection(self, *args, **kwargs):
        return self._client.create_collection(*args, **kwargs)

    def query_points(self, *args, **kwargs):
        if "query_vector" in kwargs and "query" not in kwargs:
            kwargs["query"] = kwargs.pop("query_vector")
        return self._client.query_points(*args, **kwargs)

    def retrieve(self, *args, **kwargs):
        return self._client.retrieve(*args, **kwargs)

    def upsert(self, *args, **kwargs):
        return self._client.upsert(*args, **kwargs)

    def scroll(self, *args, **kwargs):
        return self._client.scroll(*args, **kwargs)

    def get_collection(self, *args, **kwargs):
        return self._client.get_collection(*args, **kwargs)

    def close(self):
        if self._client is not None:
            self._client.close()

    def get_client(self):
        return self._client

    def __getattr__(self, name: str):
        """Compatibility passthrough for raw QdrantClient attributes."""
        return getattr(self._client, name)


def create_vector_store(location=":memory:", backend="qdrant", client_cls=QdrantClient):
    """Factory for vector store implementations."""
    if backend != "qdrant":
        raise ValueError(f"Unsupported vector store backend: {backend}")
    return QdrantVectorStore(qdrant_location=location, client_cls=client_cls)
