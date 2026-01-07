"""
Structured Logging Module for ChelatedAI

Provides JSON-formatted logging with performance metrics and debugging info.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys


class ChelationLogger:
    """
    Structured logger for ChelatedAI operations.

    Logs events in JSON format for easy parsing and analysis.
    Includes performance metrics, hyperparameters, and debugging info.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        """
        Initialize logger.

        Args:
            log_path: Path to log file (default: chelation_debug.jsonl)
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.log_path = log_path or Path("chelation_debug.jsonl")
        self.start_time = time.time()
        self.operation_stack = []  # Track nested operations

        # Setup Python logging
        self.logger = logging.getLogger("ChelatedAI")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler (pretty formatted)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (JSON formatted)
        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level.upper()))
        self.logger.addHandler(file_handler)

    def log_event(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        **kwargs
    ):
        """
        Log a structured event.

        Args:
            event_type: Type of event (e.g., 'query', 'training', 'error')
            message: Human-readable message
            level: Logging level
            **kwargs: Additional fields to include in JSON
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": time.time() - self.start_time,
            "event_type": event_type,
            "level": level,
            "message": message,
            **kwargs
        }

        # Log to Python logger
        log_method = getattr(self.logger, level.lower())
        log_method(f"[{event_type}] {message}")

        # Write JSON to file
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + "\n")
        except IOError as e:
            self.logger.error(f"Failed to write to log file: {e}")

    def log_query(
        self,
        query_text: str,
        variance: float,
        action: str,
        top_ids: list,
        jaccard: float,
        **kwargs
    ):
        """
        Log a query event with retrieval metrics.

        Args:
            query_text: The query string
            variance: Global variance metric
            action: Decision made ('FAST', 'CHELATE', etc.)
            top_ids: Top document IDs
            jaccard: Overlap between standard and chelated results
            **kwargs: Additional metrics
        """
        self.log_event(
            event_type="query",
            message=f"Query: '{query_text[:50]}...' | Action: {action}",
            level="INFO",
            query_snippet=query_text[:100],
            global_variance=float(variance),
            action=action,
            top_10_ids=[str(id) for id in top_ids[:10]],
            jaccard_similarity=float(jaccard),
            **kwargs
        )

    def log_training_start(
        self,
        num_samples: int,
        learning_rate: float,
        epochs: int,
        threshold: int,
        **kwargs
    ):
        """
        Log start of training cycle.

        Args:
            num_samples: Number of training samples
            learning_rate: Adapter learning rate
            epochs: Number of epochs
            threshold: Collapse frequency threshold
            **kwargs: Additional hyperparameters
        """
        self.log_event(
            event_type="training_start",
            message=f"Starting training on {num_samples} samples",
            level="INFO",
            num_samples=num_samples,
            learning_rate=learning_rate,
            epochs=epochs,
            collapse_threshold=threshold,
            **kwargs
        )

    def log_training_epoch(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        **kwargs
    ):
        """
        Log training epoch progress.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            loss: Training loss
            **kwargs: Additional metrics
        """
        self.log_event(
            event_type="training_epoch",
            message=f"Epoch {epoch}/{total_epochs} | Loss: {loss:.6f}",
            level="DEBUG",
            epoch=epoch,
            total_epochs=total_epochs,
            loss=float(loss),
            **kwargs
        )

    def log_training_complete(
        self,
        final_loss: float,
        vectors_updated: int,
        vectors_failed: int = 0,
        **kwargs
    ):
        """
        Log completion of training cycle.

        Args:
            final_loss: Final training loss
            vectors_updated: Number of vectors successfully updated
            vectors_failed: Number of failed updates
            **kwargs: Additional metrics
        """
        self.log_event(
            event_type="training_complete",
            message=f"Training complete | Updated: {vectors_updated} | Failed: {vectors_failed}",
            level="INFO",
            final_loss=float(final_loss),
            vectors_updated=vectors_updated,
            vectors_failed=vectors_failed,
            **kwargs
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ):
        """
        Log an error event.

        Args:
            error_type: Type of error (e.g., 'connection', 'validation')
            message: Error message
            exception: Exception object if available
            **kwargs: Additional context
        """
        event_data = {
            "error_type": error_type,
        }

        if exception:
            event_data["exception_type"] = type(exception).__name__
            event_data["exception_message"] = str(exception)

        self.log_event(
            event_type="error",
            message=message,
            level="ERROR",
            **event_data,
            **kwargs
        )

    def log_performance(
        self,
        operation: str,
        duration_seconds: float,
        **kwargs
    ):
        """
        Log performance metrics for an operation.

        Args:
            operation: Name of operation
            duration_seconds: Time taken
            **kwargs: Additional metrics (e.g., throughput, batch_size)
        """
        self.log_event(
            event_type="performance",
            message=f"{operation} completed in {duration_seconds:.3f}s",
            level="DEBUG",
            operation=operation,
            duration_seconds=duration_seconds,
            **kwargs
        )

    def log_checkpoint(
        self,
        checkpoint_type: str,
        checkpoint_path: Path,
        **kwargs
    ):
        """
        Log checkpoint creation/restoration.

        Args:
            checkpoint_type: Type of checkpoint ('save' or 'load')
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional metadata
        """
        self.log_event(
            event_type="checkpoint",
            message=f"Checkpoint {checkpoint_type}: {checkpoint_path}",
            level="INFO",
            checkpoint_type=checkpoint_type,
            checkpoint_path=str(checkpoint_path),
            **kwargs
        )

    def start_operation(self, operation_name: str) -> 'OperationContext':
        """
        Start a timed operation context.

        Args:
            operation_name: Name of the operation

        Returns:
            OperationContext for use with 'with' statement
        """
        return OperationContext(self, operation_name)


class OperationContext:
    """Context manager for timed operations."""

    def __init__(self, logger: ChelationLogger, operation_name: str):
        """
        Initialize operation context.

        Args:
            logger: ChelationLogger instance
            operation_name: Name of operation
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.operation_stack.append(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance."""
        duration = time.time() - self.start_time
        self.logger.operation_stack.pop()

        if exc_type is not None:
            self.logger.log_error(
                error_type="operation_failed",
                message=f"{self.operation_name} failed after {duration:.3f}s",
                exception=exc_val
            )
        else:
            self.logger.log_performance(
                operation=self.operation_name,
                duration_seconds=duration
            )


# Global logger instance
_global_logger = None


def get_logger(
    log_path: Optional[Path] = None,
    console_level: str = "INFO"
) -> ChelationLogger:
    """
    Get or create global logger instance.

    Args:
        log_path: Path to log file (only used on first call)
        console_level: Console logging level (only used on first call)

    Returns:
        ChelationLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ChelationLogger(log_path, console_level)
    return _global_logger


if __name__ == "__main__":
    # Demo usage
    logger = get_logger(Path("demo_log.jsonl"))

    logger.log_event("demo", "Starting demo")

    # Query logging
    logger.log_query(
        query_text="What is machine learning?",
        variance=0.00035,
        action="FAST",
        top_ids=[1, 5, 12, 23],
        jaccard=0.85
    )

    # Training logging
    logger.log_training_start(
        num_samples=100,
        learning_rate=0.01,
        epochs=10,
        threshold=3
    )

    for epoch in range(3):
        logger.log_training_epoch(epoch+1, 3, 0.05 - epoch*0.01)

    logger.log_training_complete(
        final_loss=0.02,
        vectors_updated=95,
        vectors_failed=5
    )

    # Timed operation
    with logger.start_operation("embedding_batch"):
        time.sleep(0.1)  # Simulate work

    # Error logging
    try:
        raise ValueError("Demo error")
    except ValueError as e:
        logger.log_error(
            error_type="validation",
            message="Invalid parameter",
            exception=e,
            param_name="test"
        )

    print(f"\nLog written to: {logger.log_path}")
