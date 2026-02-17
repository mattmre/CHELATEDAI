"""
Unit tests for chelation_logger.py

Tests structured logging functionality including:
- JSON event writing
- Error logging with exception metadata
- OperationContext timing and error handling
- Logger singleton behavior
"""

import json
import logging
import unittest
import tempfile
import time
from pathlib import Path
from datetime import datetime

# Import the logger module
import chelation_logger
from chelation_logger import ChelationLogger, OperationContext, get_logger


def _close_chelatedai_handlers():
    """Close and detach logger handlers to release file locks between tests."""
    py_logger = logging.getLogger("ChelatedAI")
    for handler in list(py_logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
    py_logger.handlers = []


class TestChelationLogger(unittest.TestCase):
    """Test suite for ChelationLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        _close_chelatedai_handlers()
        # Create temporary directory for test logs
        self.temp_dir = tempfile.mkdtemp()
        self.temp_log_path = Path(self.temp_dir) / "test_log.jsonl"
        
        # Reset global logger to ensure test isolation
        chelation_logger._global_logger = None

    def tearDown(self):
        """Clean up test fixtures."""
        _close_chelatedai_handlers()
        # Clean up temp files
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        
        # Reset global logger
        chelation_logger._global_logger = None

    def _read_json_events(self, log_path):
        """
        Read JSON events from log file, handling mixed content.
        
        Robustly parses only valid JSON lines, ignoring non-JSON output
        from standard FileHandler.
        
        Args:
            log_path: Path to log file
            
        Returns:
            List of parsed JSON event dictionaries
        """
        events = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError:
                    # Skip non-JSON lines (e.g., from FileHandler)
                    pass
        return events

    def test_log_event_writes_structured_json(self):
        """Test that log_event writes structured JSON event to file."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",  # Suppress console output in tests
            file_level="DEBUG"
        )

        # Log a test event
        logger.log_event(
            event_type="test_event",
            message="Test message",
            level="INFO",
            custom_field="custom_value",
            numeric_field=42
        )

        # Read and parse the JSON event
        events = self._read_json_events(self.temp_log_path)
        
        # Verify we got exactly one event
        self.assertEqual(len(events), 1)
        
        event = events[0]
        
        # Verify required fields
        self.assertIn("timestamp", event)
        self.assertIn("elapsed_seconds", event)
        self.assertEqual(event["event_type"], "test_event")
        self.assertEqual(event["level"], "INFO")
        self.assertEqual(event["message"], "Test message")
        
        # Verify custom fields
        self.assertEqual(event["custom_field"], "custom_value")
        self.assertEqual(event["numeric_field"], 42)
        
        # Verify timestamp format (ISO 8601)
        timestamp = datetime.fromisoformat(event["timestamp"])
        self.assertIsInstance(timestamp, datetime)
        
        # Verify elapsed_seconds is numeric
        self.assertIsInstance(event["elapsed_seconds"], (int, float))
        self.assertGreaterEqual(event["elapsed_seconds"], 0)

    def test_log_error_captures_exception_metadata(self):
        """Test that log_error captures exception type and message."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Create a test exception
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            test_exception = e

        # Log the error with exception metadata
        logger.log_error(
            error_type="validation_error",
            message="Validation failed",
            exception=test_exception,
            context="test_context"
        )

        # Read and verify the error event
        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        
        # Verify error event structure
        self.assertEqual(event["event_type"], "error")
        self.assertEqual(event["level"], "ERROR")
        self.assertEqual(event["message"], "Validation failed")
        self.assertEqual(event["error_type"], "validation_error")
        
        # Verify exception metadata
        self.assertEqual(event["exception_type"], "ValueError")
        self.assertEqual(event["exception_message"], "Test error message")
        
        # Verify additional context
        self.assertEqual(event["context"], "test_context")

    def test_log_error_without_exception(self):
        """Test that log_error works without exception object."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Log error without exception
        logger.log_error(
            error_type="connection_error",
            message="Connection timeout",
            retry_count=3
        )

        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        
        self.assertEqual(event["event_type"], "error")
        self.assertEqual(event["error_type"], "connection_error")
        self.assertEqual(event["message"], "Connection timeout")
        
        # Should not have exception fields
        self.assertNotIn("exception_type", event)
        self.assertNotIn("exception_message", event)
        
        # Should have custom fields
        self.assertEqual(event["retry_count"], 3)

    def test_operation_context_success_logs_performance(self):
        """Test that OperationContext logs performance event on success."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Use operation context successfully
        with logger.start_operation("test_operation"):
            time.sleep(0.01)  # Small delay to measure

        # Read and verify performance event
        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        
        # Verify performance event structure
        self.assertEqual(event["event_type"], "performance")
        self.assertEqual(event["level"], "DEBUG")
        self.assertEqual(event["operation"], "test_operation")
        
        # Verify duration was captured
        self.assertIn("duration_seconds", event)
        self.assertIsInstance(event["duration_seconds"], (int, float))
        self.assertGreater(event["duration_seconds"], 0)
        self.assertLess(event["duration_seconds"], 1.0)  # Should be < 1s
        
        # Verify message format
        self.assertIn("test_operation", event["message"])
        self.assertIn("completed", event["message"])

    def test_operation_context_failure_logs_error(self):
        """Test that OperationContext logs operation_failed error on exception."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Use operation context with exception
        with self.assertRaises(RuntimeError):
            with logger.start_operation("failing_operation"):
                time.sleep(0.01)
                raise RuntimeError("Operation failed")

        # Read and verify error event
        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        
        # Verify error event structure
        self.assertEqual(event["event_type"], "error")
        self.assertEqual(event["level"], "ERROR")
        self.assertEqual(event["error_type"], "operation_failed")
        
        # Verify exception metadata
        self.assertEqual(event["exception_type"], "RuntimeError")
        self.assertEqual(event["exception_message"], "Operation failed")
        
        # Verify message includes operation name and duration
        self.assertIn("failing_operation", event["message"])
        self.assertIn("failed after", event["message"])

    def test_operation_context_stack_management(self):
        """Test that operation stack is properly managed."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Initial stack should be empty
        self.assertEqual(len(logger.operation_stack), 0)

        # Enter operation
        with logger.start_operation("outer_op"):
            self.assertEqual(len(logger.operation_stack), 1)
            self.assertEqual(logger.operation_stack[0], "outer_op")
            
            # Nested operation
            with logger.start_operation("inner_op"):
                self.assertEqual(len(logger.operation_stack), 2)
                self.assertEqual(logger.operation_stack[1], "inner_op")
            
            # After inner completes
            self.assertEqual(len(logger.operation_stack), 1)

        # After all complete
        self.assertEqual(len(logger.operation_stack), 0)

    def test_operation_context_stack_cleanup_on_error(self):
        """Test that operation stack is cleaned up even on error."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Stack should be cleaned up even on error
        with self.assertRaises(ValueError):
            with logger.start_operation("error_op"):
                raise ValueError("Test error")

        # Stack should be empty after error
        self.assertEqual(len(logger.operation_stack), 0)

    def test_get_logger_singleton_behavior(self):
        """Test that get_logger returns same instance on repeated calls."""
        # First call creates logger
        logger1 = get_logger(
            log_path=self.temp_log_path,
            console_level="ERROR"
        )
        
        self.assertIsInstance(logger1, ChelationLogger)
        
        # Second call returns same instance
        logger2 = get_logger(
            log_path=Path(self.temp_dir) / "different.jsonl",  # Different path
            console_level="DEBUG"  # Different level
        )
        
        # Should be the exact same object
        self.assertIs(logger1, logger2)
        
        # Third call also returns same instance
        logger3 = get_logger()
        self.assertIs(logger1, logger3)

    def test_get_logger_uses_first_call_configuration(self):
        """Test that get_logger uses configuration from first call only."""
        # Reset global logger
        chelation_logger._global_logger = None
        
        # First call with specific configuration
        logger1 = get_logger(
            log_path=self.temp_log_path,
            console_level="ERROR"
        )
        
        # Log an event
        logger1.log_event("test", "First event", level="INFO")
        
        # Get logger again with different path
        different_path = Path(self.temp_dir) / "ignored.jsonl"
        logger2 = get_logger(
            log_path=different_path,
            console_level="DEBUG"
        )
        
        # Log another event
        logger2.log_event("test", "Second event", level="INFO")
        
        # Both events should be in the FIRST log path
        events = self._read_json_events(self.temp_log_path)
        self.assertEqual(len(events), 2)
        
        # The different_path file should not exist
        self.assertFalse(different_path.exists())

    def test_multiple_events_in_sequence(self):
        """Test logging multiple events creates valid JSON lines."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Log multiple events
        for i in range(5):
            logger.log_event(
                event_type="sequence_test",
                message=f"Event {i}",
                level="INFO",
                index=i
            )

        # Read all events
        events = self._read_json_events(self.temp_log_path)
        
        # Should have all 5 events
        self.assertEqual(len(events), 5)
        
        # Verify sequence
        for i, event in enumerate(events):
            self.assertEqual(event["event_type"], "sequence_test")
            self.assertEqual(event["message"], f"Event {i}")
            self.assertEqual(event["index"], i)

    def test_mixed_log_content_handling(self):
        """Test robust handling of mixed JSON and non-JSON log lines."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Log a proper event
        logger.log_event("test", "First event", level="INFO")
        
        # Manually write some non-JSON content (simulating FileHandler output)
        with open(self.temp_log_path, 'a', encoding='utf-8') as f:
            f.write("This is not JSON\n")
            f.write("Another non-JSON line\n")
            f.write("\n")  # Empty line
        
        # Log another proper event
        logger.log_event("test", "Second event", level="INFO")

        # Should successfully read only the JSON events
        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["message"], "First event")
        self.assertEqual(events[1]["message"], "Second event")

    def test_log_performance_directly(self):
        """Test direct use of log_performance method."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Log performance metrics
        logger.log_performance(
            operation="batch_processing",
            duration_seconds=2.5,
            batch_size=100,
            throughput=40.0
        )

        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 1)
        event = events[0]
        
        self.assertEqual(event["event_type"], "performance")
        self.assertEqual(event["operation"], "batch_processing")
        self.assertEqual(event["duration_seconds"], 2.5)
        self.assertEqual(event["batch_size"], 100)
        self.assertEqual(event["throughput"], 40.0)

    def test_different_log_levels(self):
        """Test logging at different levels."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        for level in levels:
            logger.log_event(
                event_type="level_test",
                message=f"Message at {level}",
                level=level
            )

        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 4)
        
        for i, level in enumerate(levels):
            self.assertEqual(events[i]["level"], level)
            self.assertEqual(events[i]["message"], f"Message at {level}")

    def test_elapsed_seconds_increases(self):
        """Test that elapsed_seconds increases over time."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Log first event
        logger.log_event("test", "Event 1", level="INFO")
        
        # Small delay
        time.sleep(0.05)
        
        # Log second event
        logger.log_event("test", "Event 2", level="INFO")

        events = self._read_json_events(self.temp_log_path)
        
        self.assertEqual(len(events), 2)
        
        # Second event should have larger elapsed_seconds
        self.assertGreater(
            events[1]["elapsed_seconds"],
            events[0]["elapsed_seconds"]
        )

    def test_custom_kwargs_passthrough(self):
        """Test that custom kwargs are passed through correctly."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR",
            file_level="DEBUG"
        )

        # Log event with various custom fields
        logger.log_event(
            event_type="custom_test",
            message="Testing kwargs",
            level="INFO",
            string_field="test_string",
            int_field=42,
            float_field=3.14,
            bool_field=True,
            list_field=[1, 2, 3],
            dict_field={"key": "value"}
        )

        events = self._read_json_events(self.temp_log_path)
        event = events[0]
        
        # Verify all custom fields
        self.assertEqual(event["string_field"], "test_string")
        self.assertEqual(event["int_field"], 42)
        self.assertEqual(event["float_field"], 3.14)
        self.assertEqual(event["bool_field"], True)
        self.assertEqual(event["list_field"], [1, 2, 3])
        self.assertEqual(event["dict_field"], {"key": "value"})


class TestOperationContextStandalone(unittest.TestCase):
    """Test OperationContext as standalone functionality."""

    def setUp(self):
        """Set up test fixtures."""
        _close_chelatedai_handlers()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_log_path = Path(self.temp_dir) / "test_log.jsonl"
        chelation_logger._global_logger = None

    def tearDown(self):
        """Clean up test fixtures."""
        _close_chelatedai_handlers()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        chelation_logger._global_logger = None

    def test_operation_context_initialization(self):
        """Test OperationContext initialization."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR"
        )
        
        ctx = OperationContext(logger, "test_op")
        
        self.assertEqual(ctx.logger, logger)
        self.assertEqual(ctx.operation_name, "test_op")
        self.assertIsNone(ctx.start_time)

    def test_operation_context_start_time_set(self):
        """Test that start_time is set on context entry."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR"
        )
        
        with logger.start_operation("test_op") as ctx:
            self.assertIsNotNone(ctx.start_time)
            self.assertIsInstance(ctx.start_time, float)


class TestLoggerMethods(unittest.TestCase):
    """Test specialized logging methods."""

    def setUp(self):
        """Set up test fixtures."""
        _close_chelatedai_handlers()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_log_path = Path(self.temp_dir) / "test_log.jsonl"
        chelation_logger._global_logger = None

    def tearDown(self):
        """Clean up test fixtures."""
        _close_chelatedai_handlers()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        chelation_logger._global_logger = None

    def _read_json_events(self, log_path):
        """Read JSON events from log file."""
        events = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events

    def test_log_query_method(self):
        """Test log_query specialized method."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR"
        )

        logger.log_query(
            query_text="What is the capital of France?",
            variance=0.00042,
            action="FAST",
            top_ids=[1, 5, 10, 15, 20],
            jaccard=0.87
        )

        events = self._read_json_events(self.temp_log_path)
        event = events[0]
        
        self.assertEqual(event["event_type"], "query")
        self.assertEqual(event["action"], "FAST")
        self.assertEqual(event["global_variance"], 0.00042)
        self.assertEqual(event["jaccard_similarity"], 0.87)
        self.assertEqual(len(event["top_10_ids"]), 5)

    def test_log_checkpoint_method(self):
        """Test log_checkpoint specialized method."""
        logger = ChelationLogger(
            log_path=self.temp_log_path,
            console_level="ERROR"
        )

        checkpoint_path = Path("/path/to/checkpoint.pt")
        logger.log_checkpoint(
            checkpoint_type="save",
            checkpoint_path=checkpoint_path,
            epoch=5
        )

        events = self._read_json_events(self.temp_log_path)
        event = events[0]
        
        self.assertEqual(event["event_type"], "checkpoint")
        self.assertEqual(event["checkpoint_type"], "save")
        self.assertEqual(event["epoch"], 5)


if __name__ == "__main__":
    unittest.main()
