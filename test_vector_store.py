"""
Test Vector Store Abstraction (F-044)

Tests for the vector store abstraction layer and Qdrant implementation.
"""

import unittest
from vector_store import VectorStore, QdrantVectorStore, create_vector_store
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np


class TestVectorStoreAbstraction(unittest.TestCase):
    """Test the vector store abstraction interface."""
    
    def test_create_vector_store_qdrant(self):
        """Test factory creates QdrantVectorStore."""
        store = create_vector_store(location=":memory:", backend="qdrant")
        self.assertIsInstance(store, QdrantVectorStore)
        self.assertIsInstance(store, VectorStore)
        store.close()
    
    def test_create_vector_store_invalid_backend(self):
        """Test factory rejects invalid backend."""
        with self.assertRaises(ValueError) as ctx:
            create_vector_store(location=":memory:", backend="invalid")
        self.assertIn("Unsupported", str(ctx.exception))
    
    def test_qdrant_vector_store_memory_init(self):
        """Test QdrantVectorStore initialization with :memory:."""
        store = QdrantVectorStore(qdrant_location=":memory:")
        self.assertIsNotNone(store)
        self.assertIsNotNone(store.get_client())
        store.close()
    
    def test_qdrant_vector_store_none_location_raises(self):
        """Test QdrantVectorStore rejects None location."""
        with self.assertRaises(ValueError) as ctx:
            QdrantVectorStore(qdrant_location=None)
        self.assertIn("cannot be None", str(ctx.exception))
    
    def test_qdrant_vector_store_invalid_url_raises(self):
        """Test QdrantVectorStore validates URL format."""
        with self.assertRaises(ValueError) as ctx:
            QdrantVectorStore(qdrant_location="http://")
        self.assertIn("Invalid Qdrant URL", str(ctx.exception))


class TestQdrantVectorStoreOperations(unittest.TestCase):
    """Test QdrantVectorStore operations."""
    
    def setUp(self):
        """Create in-memory vector store for testing."""
        self.store = QdrantVectorStore(qdrant_location=":memory:")
        self.collection_name = "test_collection"
        self.vector_size = 384
    
    def tearDown(self):
        """Clean up vector store."""
        if self.store:
            self.store.close()
    
    def test_collection_operations(self):
        """Test collection creation and existence check."""
        # Initially should not exist
        self.assertFalse(self.store.collection_exists(self.collection_name))
        
        # Create collection
        vectors_config = VectorParams(size=self.vector_size, distance=Distance.COSINE)
        self.store.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
        
        # Now should exist
        self.assertTrue(self.store.collection_exists(self.collection_name))
    
    def test_upsert_and_retrieve(self):
        """Test upserting and retrieving points."""
        # Create collection
        vectors_config = VectorParams(size=self.vector_size, distance=Distance.COSINE)
        self.store.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
        
        # Create test points
        vector1 = np.random.rand(self.vector_size).astype(np.float32).tolist()
        vector2 = np.random.rand(self.vector_size).astype(np.float32).tolist()
        
        points = [
            PointStruct(id=1, vector=vector1, payload={"text": "test1"}),
            PointStruct(id=2, vector=vector2, payload={"text": "test2"})
        ]
        
        # Upsert points
        result = self.store.upsert(collection_name=self.collection_name, points=points)
        self.assertIsNotNone(result)
        
        # Retrieve points
        retrieved = self.store.retrieve(
            collection_name=self.collection_name,
            ids=[1, 2],
            with_vectors=True,
            with_payload=True
        )
        
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(retrieved[0].payload["text"], "test1")
        self.assertEqual(retrieved[1].payload["text"], "test2")
    
    def test_query_points(self):
        """Test querying points by vector."""
        # Create collection
        vectors_config = VectorParams(size=self.vector_size, distance=Distance.COSINE)
        self.store.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
        
        # Create and insert test points
        query_vector = np.random.rand(self.vector_size).astype(np.float32)
        similar_vector = query_vector + 0.01 * np.random.rand(self.vector_size).astype(np.float32)
        dissimilar_vector = np.random.rand(self.vector_size).astype(np.float32)
        
        points = [
            PointStruct(id=1, vector=similar_vector.tolist(), payload={"text": "similar"}),
            PointStruct(id=2, vector=dissimilar_vector.tolist(), payload={"text": "dissimilar"})
        ]
        
        self.store.upsert(collection_name=self.collection_name, points=points)
        
        # Query for similar points
        results = self.store.query_points(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=2
        )
        
        self.assertIsNotNone(results)
        self.assertGreater(len(results.points), 0)
    
    def test_scroll(self):
        """Test scrolling through collection."""
        # Create collection
        vectors_config = VectorParams(size=self.vector_size, distance=Distance.COSINE)
        self.store.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
        
        # Insert multiple points
        points = [
            PointStruct(
                id=i,
                vector=np.random.rand(self.vector_size).astype(np.float32).tolist(),
                payload={"text": f"test{i}"}
            )
            for i in range(10)
        ]
        
        self.store.upsert(collection_name=self.collection_name, points=points)
        
        # Scroll through points
        records, next_offset = self.store.scroll(
            collection_name=self.collection_name,
            limit=5,
            with_vectors=False,
            with_payload=True
        )
        
        self.assertEqual(len(records), 5)
    
    def test_get_client_returns_qdrant_client(self):
        """Test get_client returns QdrantClient for backward compatibility."""
        client = self.store.get_client()
        self.assertIsNotNone(client)
        # Check it's a QdrantClient by checking for expected methods
        self.assertTrue(hasattr(client, 'collection_exists'))
        self.assertTrue(hasattr(client, 'create_collection'))
        self.assertTrue(hasattr(client, 'upsert'))


if __name__ == '__main__':
    unittest.main()
