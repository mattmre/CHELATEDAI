"""
Embedding Backend Abstraction (F-045)

Provides a clean interface for different embedding providers (Ollama, local models),
extracting the branching logic from AntigravityEngine.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Optional
from chelation_logger import get_logger
from config import ChelationConfig

# Safe import for requests (used in Ollama mode)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""
    
    def __init__(self, model_name: str, logger=None):
        """
        Initialize embedding backend.
        
        Args:
            model_name: Name/identifier of the embedding model
            logger: Optional logger instance (creates new if None)
        """
        self.model_name = model_name
        self.logger = logger or get_logger()
        self._vector_size = None
    
    @abstractmethod
    def get_vector_size(self) -> int:
        """Return the embedding dimension size."""
        pass
    
    @abstractmethod
    def embed_raw(self, texts: List[str]) -> np.ndarray:
        """
        Generate raw embeddings for texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape [len(texts), vector_size], dtype float32
        """
        pass
    
    @property
    def vector_size(self) -> int:
        """Get cached vector size."""
        if self._vector_size is None:
            self._vector_size = self.get_vector_size()
        return self._vector_size


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embedding backend using HTTP API."""
    
    def __init__(self, model_name: str, logger=None):
        """
        Initialize Ollama embedding backend.
        
        Args:
            model_name: Ollama model name (e.g., 'nomic-embed-text')
            logger: Optional logger instance
        """
        super().__init__(model_name, logger)
        
        # Check for requests availability
        if not REQUESTS_AVAILABLE:
            raise ImportError("'requests' library required for Ollama mode. Install with: pip install requests")
        
        self.ollama_url = ChelationConfig.OLLAMA_URL
        self.logger.log_event(
            "embedding_backend_init",
            f"Initializing Ollama backend: {model_name}",
            backend="ollama",
            model_name=model_name,
            url=self.ollama_url
        )
        
        # Initialize vector size with fallback
        self._vector_size = ChelationConfig.DEFAULT_VECTOR_SIZE
        try:
            test_vec = self.embed_raw(["test"])[0]
            self._vector_size = len(test_vec)
            self.logger.log_event(
                "embedding_backend_ready",
                f"Ollama backend connected. Vector size: {self._vector_size}",
                vector_size=self._vector_size
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.ollama_url}. Make sure Docker container is running!"
            ) from e
        except (requests.exceptions.RequestException, KeyError, ValueError, TypeError, IndexError) as e:
            self.logger.log_error(
                "connection",
                f"Ollama connection test failed: {e}",
                exception=e
            )
            self.logger.log_event(
                "initialization",
                "Vector size will be validated on first real embedding call",
                level="WARNING"
            )
            # Keep default as fallback
    
    def get_vector_size(self) -> int:
        """Return the embedding dimension size."""
        return self._vector_size
    
    def _sanitize_text(self, text: str, doc_index: Optional[int] = None) -> str:
        """Sanitize embedding input text for Ollama requests."""
        if not isinstance(text, str):
            text = str(text)
        
        if len(text) > ChelationConfig.OLLAMA_INPUT_MAX_CHARS:
            self.logger.log_event(
                "embedding_input_truncated",
                f"Input text exceeded max length ({ChelationConfig.OLLAMA_INPUT_MAX_CHARS}), truncating.",
                level="DEBUG",
                doc_index=doc_index,
                original_length=len(text),
                truncated_length=ChelationConfig.OLLAMA_INPUT_MAX_CHARS,
            )
            text = text[:ChelationConfig.OLLAMA_INPUT_MAX_CHARS]
        
        # Replace non-printable control chars (except whitespace controls)
        return "".join(c if (c.isprintable() or c in "\n\r\t") else " " for c in text)
    
    def embed_raw(self, texts: List[str]) -> np.ndarray:
        """
        Generate raw embeddings using Ollama API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape [len(texts), vector_size], dtype float32
        """
        if not texts:
            return np.array([])
        
        embeddings = [None] * len(texts)
        
        def _get_embedding(i, txt):
            txt = self._sanitize_text(txt, doc_index=i)
            
            # Helper to attempt embedding
            def attempt(t):
                try:
                    res = requests.post(
                        self.ollama_url,
                        json={
                            "model": self.model_name,
                            "prompt": t,
                            "options": {"num_ctx": ChelationConfig.OLLAMA_NUM_CTX}
                        },
                        timeout=ChelationConfig.OLLAMA_TIMEOUT
                    )
                    if res.status_code == 200:
                        return res.json()["embedding"]
                    else:
                        # 500 error usually means context limit or model error
                        return None
                except requests.exceptions.Timeout:
                    self.logger.log_error("timeout", f"Ollama timeout for doc {i}", doc_index=i)
                    return None
                except requests.exceptions.ConnectionError:
                    self.logger.log_error("connection", f"Ollama connection lost for doc {i}", doc_index=i)
                    return None
                except KeyError as e:
                    self.logger.log_error(
                        "api_response",
                        f"Ollama response missing 'embedding' key for doc {i}",
                        exception=e,
                        doc_index=i
                    )
                    return None
                except Exception as e:
                    self.logger.log_error(
                        "embedding",
                        f"Ollama unexpected error for doc {i}",
                        exception=e,
                        doc_index=i
                    )
                    return None
            
            # Try truncation levels from config
            for limit in ChelationConfig.OLLAMA_TRUNCATION_LIMITS:
                current_text = txt[:limit]
                emb = attempt(current_text)
                if emb is not None:
                    break
            
            if emb is None:
                self.logger.log_error(
                    "embedding_failed",
                    f"Failed to embed doc {i} after retries",
                    doc_index=i
                )
                return i, np.zeros(self.vector_size, dtype=np.float32)
            
            return i, np.array(emb, dtype=np.float32)
        
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        with ThreadPoolExecutor(max_workers=ChelationConfig.OLLAMA_MAX_WORKERS) as executor:
            futures = [executor.submit(_get_embedding, i, txt) for i, txt in enumerate(texts)]
            for idx, future in enumerate(futures):
                try:
                    _, emb = future.result(timeout=ChelationConfig.OLLAMA_TIMEOUT)
                    embeddings[idx] = emb
                except TimeoutError:
                    self.logger.log_error(
                        "timeout",
                        f"Embedding timeout for document {idx}, using zero vector",
                        doc_index=idx
                    )
                    embeddings[idx] = np.zeros(self.vector_size, dtype=np.float32)
                except Exception as e:
                    self.logger.log_error(
                        "embedding",
                        f"Embedding failed for document {idx}",
                        exception=e,
                        doc_index=idx
                    )
                    embeddings[idx] = np.zeros(self.vector_size, dtype=np.float32)
        
        return np.array(embeddings, dtype=np.float32)


class LocalEmbeddingBackend(EmbeddingBackend):
    """Local sentence-transformers embedding backend."""
    
    def __init__(self, model_name: str, logger=None):
        """
        Initialize local embedding backend.
        
        Args:
            model_name: Sentence-transformers model name
            logger: Optional logger instance
        """
        super().__init__(model_name, logger)
        
        self.logger.log_event(
            "embedding_backend_init",
            f"Initializing local backend: {model_name}",
            backend="local",
            model_name=model_name
        )
        
        from sentence_transformers import SentenceTransformer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.log_event("initialization", f"Device Selected: {device}", device=device)
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        self._vector_size = self.model.get_sentence_embedding_dimension()
        
        self.logger.log_event(
            "embedding_backend_ready",
            f"Local backend loaded. Vector size: {self._vector_size}",
            device=str(self.model.device),
            vector_size=self._vector_size
        )
    
    def get_vector_size(self) -> int:
        """Return the embedding dimension size."""
        return self._vector_size
    
    def embed_raw(self, texts: List[str]) -> np.ndarray:
        """
        Generate raw embeddings using local model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape [len(texts), vector_size], dtype float32
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings.astype(np.float32)


def create_embedding_backend(model_name: str, logger=None) -> EmbeddingBackend:
    """
    Factory function to create appropriate embedding backend.
    
    Args:
        model_name: Model identifier, prefix with 'ollama:' for Ollama mode
        logger: Optional logger instance
        
    Returns:
        EmbeddingBackend instance (OllamaEmbeddingBackend or LocalEmbeddingBackend)
        
    Examples:
        >>> backend = create_embedding_backend("ollama:nomic-embed-text")
        >>> backend = create_embedding_backend("all-MiniLM-L6-v2")
    """
    if model_name.startswith("ollama:"):
        # Ollama mode - strip prefix
        actual_model_name = model_name.replace("ollama:", "")
        return OllamaEmbeddingBackend(actual_model_name, logger)
    else:
        # Local mode
        return LocalEmbeddingBackend(model_name, logger)
