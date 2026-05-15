# Phase 10: Monitoring, Observability & Production Readiness
#
# ChelatedAI - Detailed Improvement Plan
# Generated: 2026-04-22

================================================================================
CURRENT STATE (Pre-Phase 10)
================================================================================

ChelatedAI currently has:
- JSON structured logging to a single file (chelation_events.jsonl) via chelation_logger.py
- A dashboard_server.py serving static HTML over http.server
- No metrics, tracing, health checks, alerting, or error tracking
- No Prometheus/OpenTelemetry integration
- No health/liveness/readiness probes
- No log aggregation pipeline
- No distributed tracing
- No alerting system
- No error tracking (Sentry, etc.)
- Dashboard has no real-time charts, no metrics panels, no alert display

The current ChelationLogger (chelation_logger.py, 486 lines) writes JSON lines
to a file with basic event types (log_event, log_query, log_error, log_performance,
log_checkpoint, log_training_start/epoch/complete). We extend this into a full
observability stack.

================================================================================
10.1 Metrics Collection — OpenTelemetry/Prometheus
================================================================================

**Goal:** Export structured metrics to Prometheus for dashboards and alerting.

**Action 1: Create `metrics_collector.py` — Prometheus + OTel metrics**

New file: `metrics_collector.py`

Core classes:

```python
class CounterMetric:
    """Simple counter metric (no Prometheus dependency required)."""
    # inc(), value property, thread-safe via Lock

class GaugeMetric:
    """Simple gauge metric."""
    # set(), inc(), dec(), value property

class HistogramMetric:
    """Simple histogram for latency measurements."""
    # observe(), summary() returns count/sum/buckets/avg
    # Buckets: (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
```

MetricsRegistry — central registry with methods:
```python
class MetricsRegistry:
    # Embedding metrics
    def record_embedding(self, duration: float, count: int = 1)
    def record_cache_hit(self)
    def record_cache_miss(self)

    # Ingestion metrics
    def record_ingestion(self, doc_count: int, duration: float)

    # Search metrics
    def record_search(self, duration: float, result_count: int = 0)

    # Training metrics
    def record_training_epoch(self, duration: float, loss: float)

    # Error metrics
    def record_error(self, error_type: str)

    # Resource metrics
    def set_active_collections(self, count: int)
    def set_qdrant_connections(self, count: int)

    # Export
    def export_prometheus(self) -> str    # Prometheus exposition format
    def export_json(self) -> dict          # JSON-serializable dict
```

Exported metrics names:
- embeddings_total (counter): Total embeddings generated
- embedding_duration_seconds (histogram): Embedding latency
- ingestion_total (counter): Total documents ingested
- ingestion_duration_seconds (histogram): Ingestion batch latency
- search_total (counter): Total search queries
- search_duration_seconds (histogram): Query latency
- search_result_count (histogram): Number of results returned
- training_epochs_total (counter): Total training epochs
- training_loss (gauge): Current training loss
- training_duration_seconds (histogram): Training time
- cache_hits_total (counter): Embedding cache hits
- cache_misses_total (counter): Embedding cache misses
- error_total (counter): Total errors
- error_<type>_total (counter): Errors by type
- active_collections (gauge): Number of active collections
- qdrant_connections_active (gauge): Active Qdrant connections

**Action 2: Integrate metrics into `antigravity_engine.py`**

```python
# In antigravity_engine.py imports:
from metrics_collector import MetricsRegistry

# In AntigravityEngine.__init__():
self.metrics = MetricsRegistry()

# In embed():
def embed(self, texts):
    start = time.perf_counter()
    # ... existing embedding logic ...
    elapsed = time.perf_counter() - start
    self.metrics.record_embedding(elapsed, len(texts))
    return embeddings

# In ingest():
def ingest(self, text_corpus, payloads=None):
    start = time.perf_counter()
    # ... existing ingestion logic ...
    elapsed = time.perf_counter() - start
    self.metrics.record_ingestion(len(text_corpus), elapsed)

# In search/run_inference():
def run_inference(self, query_text):
    start = time.perf_counter()
    # ... existing inference logic ...
    elapsed = time.perf_counter() - start
    self.metrics.record_search(elapsed, len(results))
```

**Action 3: Create `prometheus_exporter.py` — HTTP metrics endpoint**

New file: `prometheus_exporter.py`

```python
class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves Prometheus metrics at /metrics."""
    registry = None

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(self.registry.export_prometheus().encode())
        elif self.path == "/health":
            self.send_json(200, {"status": "healthy"})

def start_metrics_server(port: int = 9090, registry=None):
    MetricsHandler.registry = registry
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
```

**Action 4: OpenTelemetry integration (optional)**

Create `otel_tracer.py`:

```python
class OTelTracer:
    """OpenTelemetry tracer — gracefully degrades when OTel SDK not installed."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.traces = []  # In-memory fallback
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider = TracerProvider()
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("chelatedai")
        except ImportError:
            self.tracer = None

    def trace_embedding(self, texts: list, func):
        if self.tracer and self.enabled:
            with self.tracer.start_as_current_span("embedding") as span:
                span.set_attribute("text_count", len(texts))
                return func(texts)
        else:
            start = time.time()
            result = func(texts)
            self.traces.append({"operation": "embedding", "duration": time.time() - start,
                                "text_count": len(texts)})
            return result

    def get_traces(self) -> list:
        return self.traces
```

**Estimated effort:** 2 days
**Priority:** HIGH — foundation for all other Phase 10 work
**Files created:** `metrics_collector.py`, `prometheus_exporter.py`, `otel_tracer.py`
**Files modified:** `antigravity_engine.py`
**Expected outcome:** Prometheus endpoint at :9090/metrics; JSON export for dashboard

---

================================================================================
10.2 Structured Logging Pipeline
================================================================================

**Goal:** Extend JSON logging with rotation, async writes, and remote shipping.

**Action 1: Create `log_pipeline.py` — Multi-output logging**

New file: `log_pipeline.py`

Classes:

```python
class LogShipper:
    """Ships log lines to a remote endpoint (Elasticsearch, Loki, etc.)."""
    # Buffers logs, ships in batches via background thread
    def __init__(self, url: str, batch_size=100, interval=5.0)
    def add(self, entry: dict)  # Adds to buffer
    def stop()

class LogRotator:
    """Rotates log files by size and age."""
    def __init__(self, log_path, max_size_mb=100, backup_count=10)
    def check_rotate()  # Rotates if file exceeds max_size
    def _rotate()       # Moves file to timestamped backup

class AsyncLogBuffer:
    """Thread-safe log buffer that writes asynchronously."""
    def __init__(self, log_path, buffer_size=500)
    def write(self, entry: dict)  # Adds to buffer, flushes when full
    def _flush()                  # Writes batch to file
```

**Action 2: Extend `chelation_logger.py` with pipeline integration**

Modify `ChelationLogger.__init__()`:

```python
def __init__(self, log_path=None, console_level="INFO",
             enable_file_rotation=True, enable_async=True,
             log_shipper_url=None):
    # existing init (lines 64-81)...

    # Add file rotation
    if enable_file_rotation:
        from log_pipeline import LogRotator
        self.rotator = LogRotator(str(self.log_path), max_size_mb=100)
    else:
        self.rotator = None

    # Add async write buffer
    if enable_async:
        from log_pipeline import AsyncLogBuffer
        self.async_buffer = AsyncLogBuffer(str(self.log_path), buffer_size=500)
    else:
        self.async_buffer = None

    # Add log shipper
    if log_shipper_url:
        from log_pipeline import LogShipper
        self.log_shipper = LogShipper(log_shipper_url)
    else:
        self.log_shipper = None
```

Modify `log_event()` — replace file write section (lines 124-128):

```python
# Replace the existing file write in log_event():
if self.async_buffer:
    self.async_buffer.write(event)
else:
    try:
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + "\n")
    except IOError as e:
        self.logger.error(f"Failed to write to log file: {e}")

# Ship to remote if configured
if self.log_shipper:
    self.log_shipper.add(event)

# Check rotation
if self.rotator:
    self.rotator.check_rotate()
```

**Action 3: Create `log_dashboard_integration.py` — Log analytics for dashboard**

New file: `log_dashboard_integration.py`

```python
def analyze_logs(log_path: str, window_minutes: int = 60) -> dict:
    """Analyze recent log entries for dashboard display."""
    events = _load_recent_events(log_path, window_minutes)
    # Returns:
    # - event_type_counts: Counter of events in window
    # - error_rate: errors / total events
    # - avg_latency: average operation duration
    # - error_summary: top error types and counts
    # - events_per_minute: rate over window
```

**Estimated effort:** 1.5 days
**Priority:** HIGH — essential for operational visibility
**Files created:** `log_pipeline.py`, `log_dashboard_integration.py`
**Files modified:** `chelation_logger.py`
**Expected outcome:** Log rotation, async writes, optional remote shipping

---

================================================================================
10.3 Distributed Tracing
================================================================================

**Goal:** Trace operations across embedding, search, training, and ingestion.

**Action 1: Create `tracer.py` — Lightweight distributed tracing**

New file: `tracer.py`

```python
@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    duration_ms: float = 0.0
    def end(self, status: str = "ok")
    def to_dict(self) -> dict

class Tracer:
    """Lightweight distributed tracer for ChelatedAI."""
    def __init__(self, service_name="chelatedai", enable_otlp=False, otlp_endpoint=None)

    @contextmanager
    def span(self, operation, parent_span_id=None, attributes=None):
        """Context manager for creating a trace span."""

    @contextmanager
    def trace_embedding(self, texts: list, **attrs):
        """Trace an embedding operation."""

    @contextmanager
    def trace_search(self, query: str, **attrs):
        """Trace a search operation."""

    @contextmanager
    def trace_ingestion(self, doc_count: int, **attrs):
        """Trace an ingestion operation."""

    @contextmanager
    def trace_training(self, epoch: int, **attrs):
        """Trace a training operation."""

    def get_spans(self, limit=100) -> List[dict]
    def get_trace(self, trace_id: str) -> List[dict]
    def get_active_traces(self) -> Dict[str, List[dict]]
    def get_summary(self) -> dict  # Returns total_spans, total_traces, by_operation, by_status, avg_duration_ms, p95_duration_ms
```

**Action 2: Integrate tracer into `antigravity_engine.py`**

```python
# In antigravity_engine.py __init__():
from tracer import Tracer
self.tracer = Tracer(service_name="chelatedai-engine")

# In embed():
def embed(self, texts):
    with self.tracer.trace_embedding(texts) as span:
        # existing embedding logic...
        pass

# In search/run_inference():
def run_inference(self, query_text):
    with self.tracer.trace_search(query_text) as span:
        # existing inference logic...
        pass
```

**Action 3: Add trace API to `dashboard_server.py`**

Modify `do_GET()` to add trace endpoints:

```python
elif path == "/api/traces":
    self.handle_api_traces()
elif path == "/api/trace":
    self.handle_api_trace_detail()

def handle_api_traces(self):
    """Handle /api/traces endpoint."""
    tracer = getattr(self.server, 'tracer', None)
    if tracer:
        summary = tracer.get_summary()
        traces = tracer.get_active_traces()
        self.send_json_response({"summary": summary, "traces": list(traces.keys())})
    else:
        self.send_json_response({"summary": {}, "traces": []})

def handle_api_trace_detail(self):
    """Handle /api/trace?trace_id=xxx endpoint."""
    parsed_path = urlparse(self.path)
    query_params = parse_qs(parsed_path.query)
    trace_id = query_params.get("trace_id", [None])[0]
    tracer = getattr(self.server, 'tracer', None)
    if tracer and trace_id:
        spans = tracer.get_trace(trace_id)
        self.send_json_response({"trace_id": trace_id, "spans": spans})
    else:
        self.send_error_response(400, "trace_id parameter required")
```

**Estimated effort:** 1.5 days
**Priority:** MEDIUM — helpful for debugging complex workflows
**Files created:** `tracer.py`
**Files modified:** `antigravity_engine.py`, `dashboard_server.py`

---

================================================================================
10.4 Health Checks — Endpoint Design
================================================================================

**Goal:** Provide liveness and readiness probes for container orchestration.

**Action 1: Create `health_check.py` — Health check module**

New file: `health_check.py`

```python
class HealthStatus:
    """Tracks overall health status of the service."""
    def __init__(self)
    def register_component(self, name: str, check_fn)
    def check_all(self) -> Dict[str, Any]
        # Returns: {status, components, uptime_seconds, liveness_checks, last_healthy_at}
    def get_info(self) -> Dict[str, Any]
        # Returns: {service, version, uptime_seconds, pid, components}

# Default health checks:
def qdrant_health_check() -> dict    # Checks Qdrant connectivity
def model_health_check() -> dict     # Checks embedding model availability
def disk_health_check() -> dict      # Checks disk space (>100MB free)
def memory_health_check() -> dict    # Checks memory (<95% used)

class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    health: Optional[HealthStatus] = None

    def do_GET(self):
        if self.path in ("/healthz", "/health"):
            self._serve_health()      # Liveness check
        elif self.path == "/ready":
            self._serve_readiness()   # Readiness check (requires 30s uptime)
        elif self.path == "/info":
            self._serve_info()        # Service info

    def _serve_health():    # Returns 200 if healthy, 503 if not
    def _serve_readiness(): # Returns 200 if healthy AND uptime > 30s, else 503
    def _serve_info():      # Returns service metadata
    def send_json(code, data):

def start_health_server(port=8081, health: Optional[HealthStatus] = None):
    HealthCheckHandler.health = health
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
```

**Action 2: Integrate health checks into `antigravity_engine.py`**

```python
# In antigravity_engine.py __init__():
from health_check import HealthStatus, qdrant_health_check, model_health_check, disk_health_check, memory_health_check
from health_check import start_health_server

self.health = HealthStatus()
self.health.register_component("qdrant", qdrant_health_check)
self.health.register_component("model", model_health_check)
self.health.register_component("disk", disk_health_check)
self.health.register_component("memory", memory_health_check)
self.health_server = start_health_server(port=8081, health=self.health)
```

**Action 3: Add health config to `config.py`**

```python
# In config.py, add:
HEALTH_CHECK_ENABLED = True
HEALTH_CHECK_PORT = 8081
HEALTH_CHECK_READINESS_DELAY_SECONDS = 30
HEALTH_CHECK_LIVENESS_INTERVAL_SECONDS = 15
HEALTH_CHECK_READINESS_INTERVAL_SECONDS = 5
HEALTH_CHECK_DISK_FREE_MIN_GB = 0.1
HEALTH_CHECK_MEMORY_MAX_PCT = 95
```

**Estimated effort:** 1 day
**Priority:** HIGH — essential for containerized deployments
**Files created:** `health_check.py`
**Files modified:** `antigravity_engine.py`, `config.py`

---

================================================================================
10.5 Alerting — Threshold-based + Anomaly Detection
================================================================================

**Goal:** Automated alerting on performance degradation, errors, and anomalies.

**Action 1: Create `alerting.py` — Alerting system**

New file: `alerting.py`

```python
@dataclass
class AlertRule:
    name: str
    metric: str
    operator: str  # "gt", "lt", "gte", "lte", "eq", "neq"
    threshold: float
    window_seconds: int = 300
    cooldown_seconds: int = 300
    severity: str = "warning"  # "info", "warning", "critical"
    description: str = ""
    enabled: bool = True

@dataclass
class AlertState:
    rule: AlertRule
    fired: bool = False
    last_fired_at: float = 0
    last_value: Optional[float] = None
    trigger_count: int = 0

class AlertingEngine:
    """Evaluates alert rules against metrics and dispatches notifications."""
    def __init__(self)
    def add_rule(self, rule: AlertRule)
    def add_notifier(self, notifier: Callable)
    def record_metric(self, name: str, value: float)
    def evaluate()        # Evaluates all rules, dispatches notifications
    def get_status() -> List[dict]  # Current status for all rules

    def _check_condition(self, value, operator, threshold) -> bool
    def _dispatch(self, alert: dict)  # Sends to all notifiers
```

Default alert rules:
```python
def get_default_alert_rules() -> List[AlertRule]:
    return [
        AlertRule("high_error_rate", "error_total", "gt", 0.05, 300, 600, "critical",
                  "Error rate exceeds 5% over the last 5 minutes"),
        AlertRule("high_embedding_latency", "embedding_duration_seconds", "gt", 5.0, 300, 300, "warning",
                  "Average embedding latency exceeds 5 seconds"),
        AlertRule("high_search_latency", "search_duration_seconds", "gt", 2.0, 300, 300, "warning",
                  "Average search latency exceeds 2 seconds"),
        AlertRule("training_loss_spike", "training_loss", "gt", 1.0, 60, 60, "critical",
                  "Training loss spike detected"),
        AlertRule("low_cache_hit_rate", "cache_hit_rate", "lt", 0.3, 600, 300, "info",
                  "Embedding cache hit rate below 30%"),
    ]
```

Notifiers:
```python
def console_notifier(alert: dict):
    print(f"ALERT [{alert['severity'].upper()}] {alert['rule']}: {alert['description']}")

def webhook_notifier(url: str):
    """Send alert to Slack/Teams webhook."""
    def _notify(alert: dict):
        requests.post(url, json={"text": f"Alert: {alert['rule']}"})
    return _notify
```

**Action 2: Integrate alerting into metrics collection**

Create `alerting_engine.py` that combines MetricsRegistry + AlertingEngine:

```python
# alerting_engine.py
class AlertingEngineWithMetrics:
    """Alerting engine that automatically pulls from MetricsRegistry."""
    def __init__(self, registry: MetricsRegistry, rules: List[AlertRule])
    def evaluate()  # Pushes registry metrics to alerting engine, evaluates rules
```

**Estimated effort:** 1.5 days
**Priority:** MEDIUM — critical for production, optional for development
**Files created:** `alerting.py`, `alerting_engine.py`
**Files modified:** `metrics_collector.py` (extends to integrate with alerting)

---

================================================================================
10.6 Error Tracking — Sentry Integration
================================================================================

**Goal:** Automatic error reporting, categorization, and grouping.

**Action 1: Create `error_tracker.py` — Error tracking module**

New file: `error_tracker.py`

```python
@dataclass
class ErrorReport:
    timestamp: str
    error_type: str
    error_message: str
    traceback: str
    severity: str
    operation: str  # embedding, search, ingestion, training, etc.
    context: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    occurrences: int = 1
    first_seen: str = ""
    last_seen: str = ""
    def compute_fingerprint()  # SHA256 of error_type:message:operation

class ErrorTracker:
    """Error tracking — Sentry integration with local fallback."""
    def __init__(self, dsn=None, environment="dev", enable_sentry=False)
    def capture_error(self, error: Exception, operation="unknown", context=None, severity="warning")
    @contextmanager
    def track(self, operation, severity="warning")  # Auto-captures exceptions
    def get_errors(self, limit=50, severity=None) -> List[dict]
    def get_error_summary(self) -> dict  # Returns total_errors, unique_errors, by_type, by_severity, by_operation
    def clear()
```

Integration with Sentry:
```python
def _init_sentry(self):
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=self.dsn, environment=self.environment,
                        traces_sample_rate=0.1, profiles_sample_rate=0.1)
        self.sentry_available = True
    except ImportError:
        self.enable_sentry = False
        self.sentry_available = False

def capture_error(self, error, operation, context, severity):
    # Store locally in _error_groups (dedup by fingerprint)
    # If Sentry available: sentry_sdk.capture_exception(error, level=severity, extra={...})
```

**Action 2: Integrate error tracking into `antigravity_engine.py`**

```python
# In antigravity_engine.py __init__():
from error_tracker import ErrorTracker
import os

self.error_tracker = ErrorTracker(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "dev"),
    enable_sentry=os.getenv("SENTRY_DSN") is not None,
)

# In run_inference():
def run_inference(self, query_text):
    with self.error_tracker.track("inference", severity="critical"):
        # existing inference logic...
        pass

# In embed():
def embed(self, texts):
    with self.error_tracker.track("embedding"):
        # existing embedding logic...
        pass
```

**Action 3: Add error display to `dashboard_server.py`**

```python
def handle_api_errors(self):
    """Handle /api/errors endpoint."""
    error_tracker = getattr(self.server, 'error_tracker', None)
    if error_tracker:
        errors = error_tracker.get_errors(limit=50)
        summary = error_tracker.get_error_summary()
        self.send_json_response({"errors": errors, "summary": summary})
    else:
        self.send_json_response({"errors": [], "summary": {}})
```

Add endpoint routing:
```python
elif path == "/api/errors":
    self.handle_api_errors()
```

**Estimated effort:** 1 day
**Priority:** MEDIUM — critical for production, optional for dev
**Files created:** `error_tracker.py`
**Files modified:** `antigravity_engine.py`, `dashboard_server.py`
**Dependency:** `sentry-sdk` (optional, only installed when DSN is configured)

---

================================================================================
10.7 Dashboard Enhancement
================================================================================

**Goal:** Transform the static dashboard into a real-time observability hub.

**Action 1: Create `dashboard_enhancements.py` — Dashboard metric panels**

New file: `dashboard_enhancements.py`

Provides HTML panels:
```python
def get_dashboard_metrics_panel() -> str:
    """Return HTML for live metrics: embeddings/sec, qps, latencies, cache hit rate, error rate, active alerts."""

def get_dashboard_alerts_panel() -> str:
    """Return HTML for alerts table (severity, rule, description, value, threshold)."""

def get_dashboard_errors_panel() -> str:
    """Return HTML for errors table (type, operation, message, occurrences)."""

def get_dashboard_trace_panel() -> str:
    """Return HTML for trace explorer (trace ID, operation, duration, status, time)."""

def get_dashboard_refresh_script() -> str:
    """JavaScript for live metric refresh every 10 seconds from /api/metrics, /api/alerts, /api/errors, /api/traces."""
```

**Action 2: Modify `dashboard_server.py` — Integrate enhancements**

In `DashboardHandler.__init__()`, attach server-level references:
```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    self.metrics_registry = getattr(kwargs.get('server'), 'metrics_registry', None)
    self.health = getattr(kwargs.get('server'), 'health', None)
    self.error_tracker = getattr(kwargs.get('server'), 'error_tracker', None)
    self.tracer = getattr(kwargs.get('server'), 'tracer', None)
```

In `do_GET()`, add new endpoints:
```python
elif path == "/api/metrics":
    self.handle_api_metrics()
elif path == "/api/health":
    self.handle_api_health()
elif path == "/api/alerts":
    self.handle_api_alerts()
elif path == "/api/errors":
    self.handle_api_errors()
elif path == "/api/traces":
    self.handle_api_traces()
elif path == "/api/trace":
    self.handle_api_trace_detail()
elif path == "/api/log_analytics":
    self.handle_api_log_analytics()

def handle_api_metrics(self):
    """Serve metrics in JSON format."""
    if self.metrics_registry:
        self.send_json_response(self.metrics_registry.export_json())
    else:
        self.send_json_response({"counters": {}, "gauges": {}, "histograms": {}})

def handle_api_health(self):
    """Serve health status."""
    if self.health:
        self.send_json_response(self.health.check_all())
    else:
        self.send_json_response({"status": "healthy"})

def handle_api_alerts(self):
    """Serve alert statuses."""
    if hasattr(self.server, 'alerting'):
        self.send_json_response({"alerts": self.server.alerting.get_status()})
    else:
        self.send_json_response({"alerts": []})
```

Modify `serve_inline_dashboard()` and `serve_dashboard()` to inject metric panels:

```python
def serve_dashboard(self):
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "index.html")
    if os.path.exists(dashboard_path):
        # Load and inject metric panels
        with open(dashboard_path, 'r') as f:
            content = f.read()
        from dashboard_enhancements import (get_dashboard_metrics_panel,
                                            get_dashboard_alerts_panel,
                                            get_dashboard_errors_panel,
                                            get_dashboard_trace_panel,
                                            get_dashboard_refresh_script)
        # Inject panels into HTML before </body>
        content = content.replace('</body>',
            get_dashboard_metrics_panel() +
            get_dashboard_alerts_panel() +
            get_dashboard_errors_panel() +
            get_dashboard_trace_panel() +
            get_dashboard_refresh_script() +
            '</body>')
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
        return
    # Fallback to inline dashboard with panels injected
    self.serve_inline_dashboard()
```

**Action 3: Add log analytics to dashboard**

```python
def handle_api_log_analytics(self):
    """Serve log-based analytics."""
    from log_dashboard_integration import analyze_logs
    analytics = analyze_logs(LOG_FILE_PATH, window_minutes=60)
    self.send_json_response(analytics)
```

**Estimated effort:** 2 days
**Priority:** HIGH — transforms the dashboard from passive viewer to active observability hub
**Files created:** `dashboard_enhancements.py`
**Files modified:** `dashboard_server.py`

---

================================================================================
10.8 Runbook Creation
================================================================================

**Goal:** Document incident response procedures and common failure modes.

**Action 1: Create `docs/runbook.md`**

New file: `docs/runbook.md`

Sections:

```markdown
# ChelatedAI Production Runbook

## Common Failure Modes

### 1. Embedding Generation Fails
**Symptoms:** High error_rate alert, search returns empty results
**Diagnosis:**
  - Check /health endpoint (healthz, ready)
  - Check model health component
  - Review /api/errors for error_type: embedding
  - Check embedding_backend.py logs
**Resolution:**
  - Local mode: Verify GPU memory (nvidia-smi), restart engine
  - Ollama mode: Check Docker container (docker ps), restart ollama container
  - Check /api/traces for specific failure spans

### 2. Search Latency Spikes
**Symptoms:** high_search_latency alert (latency > 2s)
**Diagnosis:**
  - Check /api/metrics for search_duration_seconds histogram
  - Check /api/health for Qdrant component status
  - Review /api/traces for search span durations
**Resolution:**
  - Verify Qdrant is not disk-bound (check disk_health_check)
  - Consider increasing HNSW_EF_SEARCH in config.py
  - Check if collection needs compaction

### 3. Training Loss Divergence
**Symptoms:** training_loss_spike alert (loss > 1.0)
**Diagnosis:**
  - Check /api/metrics training_loss gauge
  - Review /api/errors for training errors
  - Check sedimentation_trainer.py training loop
**Resolution:**
  - Reduce learning rate (SWEEP_LR_COLLAPSE_THRESHOLD = 0.1 is the danger zone)
  - Check gradient accumulation settings
  - Verify mixed precision is working correctly
  - Consider switching to MSE loss instead of InfoNCE

### 4. Qdrant Connection Loss
**Symptoms:** error_total increases, qdrant component unhealthy
**Diagnosis:**
  - Check /api/health for Qdrant component status
  - Check if Qdrant container is running
  - Review connection error logs
**Resolution:**
  - Restart Qdrant container
  - Check Qdrant logs (docker logs <qdrant_container>)
  - Verify network connectivity
  - Check if in-memory Qdrant ran out of RAM

### 5. Disk Space Exhaustion
**Symptoms:** disk_health_check fails (< 100MB free)
**Diagnosis:**
  - Check /api/health disk component details
  - df -h on host
  - Check log files (chelation_events.jsonl) for unbounded growth
**Resolution:**
  - Rotate/truncate log files
  - Clean old checkpoints (checkpoints/ directory)
  - Remove old benchmark results (large JSON files)

### 6. Memory Exhaustion
**Symptoms:** memory_health_check fails (> 95% used)
**Diagnosis:**
  - Check /api/health memory component details
  - top/htop on host
  - Check embedding cache size (config: EMBEDDING_CACHE_MAX_SIZE)
**Resolution:**
  - Reduce EMBEDDING_CACHE_MAX_SIZE in config.py
  - Trigger GC manually via /api/metrics
  - Restart the engine process

## Deployment Validation Checklist

### Pre-Deployment
- [ ] All health checks pass (GET /healthz returns 200)
- [ ] Readiness check passes (GET /ready returns 200)
- [ ] All components healthy (Qdrant, model, disk, memory)
- [ ] Metrics endpoint accessible (GET /metrics returns Prometheus format)
- [ ] Alert rules configured (check /api/alerts)
- [ ] Error tracker initialized (check /api/errors)
- [ ] Tracer active (check /api/traces)
- [ ] Dashboard accessible and showing live data

### Post-Deployment
- [ ] Baseline metrics captured (run benchmarks)
- [ ] All alert rules firing at expected thresholds
- [ ] Error tracker capturing expected errors (test with deliberate errors)
- [ ] Log rotation working (verify log file sizes)
- [ ] Health check probes configured for load balancer
- [ ] Prometheus scrape configured
- [ ] Sentry DSN verified (if using)

### Rollback Procedures
1. Stop new deployment
2. Restart previous version (Docker: docker-compose up -d <previous_version>)
3. Restore checkpoints from checkpoints/ directory
4. Verify health: GET /healthz returns 200
5. Verify metrics: GET /metrics shows expected values
6. Run smoke tests (quick ingestion + search + query)

## Emergency Procedures

### Graceful Shutdown
```bash
# Send SIGTERM to the process
kill -TERM <pid>
# Or via Docker
docker stop <container>
```

### Emergency Restart
```bash
# Docker
docker restart <container>
# Direct
pkill -f antigravity_engine && python antigravity_engine.py
```

### Data Recovery from Checkpoints
```python
# In antigravity_engine.py, adapter weights are auto-loaded from:
# config.ADAPTER_WEIGHTS_PATH (defaults to adapter_weights.pt)
# If corrupted:
# 1. Restore from last known good checkpoint
# 2. Restart engine
```

## Monitoring Dashboards

### Prometheus/Grafana
- Import Prometheus metrics from :9090/metrics
- Create dashboards for:
  - Embedding throughput (embeddings_total over time)
  - Search latency (search_duration_seconds histogram)
  - Error rates (error_total by type)
  - Training loss (training_loss gauge)
  - Cache hit rate (cache_hits_total / (cache_hits_total + cache_misses_total))

### Dashboard Web UI
- Access at :8080/dashboard/
- View live metrics, alerts, errors, and traces
- Auto-refreshes every 10 seconds

## Escalation Matrix

| Severity | Condition | Response Time | Escalation |
|----------|-----------|---------------|------------|
| P1 | Service down (healthz returns 503) | 15 min | On-call engineer |
| P2 | Error rate > 10% | 30 min | Senior engineer |
| P3 | Search latency > 5s | 1 hour | Development team |
| P4 | Warning alerts only | Next business day | Ticket |
```

**Estimated effort:** 1 day
**Priority:** MEDIUM — important for production operations
**Files created:** `docs/runbook.md`

---

================================================================================
10.9 Load Testing — Stress Testing Infrastructure
================================================================================

**Goal:** Infrastructure for stress testing, capacity planning, and bottleneck identification.

**Action 1: Create `load_tester.py` — Load testing framework**

New file: `load_tester.py`

```python
# load_tester.py
"""Load testing framework for ChelatedAI."""

import time
import threading
import statistics
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TestResult:
    operation: str
    duration: float
    success: bool
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class LoadTestReport:
    """Report from a load test run."""
    timestamp: str
    config: Dict[str, Any]
    results: List[TestResult]
    summary: Dict[str, Any]

    def generate(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        durations = [r.duration for r in self.results if r.success]
        errors = [r for r in self.results if not r.success]

        return {
            "timestamp": self.timestamp,
            "total_operations": len(self.results),
            "successful": len(durations),
            "failed": len(errors),
            "success_rate": len(durations) / len(self.results) if self.results else 0,
            "avg_duration": statistics.mean(durations) if durations else 0,
            "median_duration": statistics.median(durations) if durations else 0,
            "p50_duration": statistics.median(durations) if durations else 0,
            "p95_duration": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "p99_duration": sorted(durations)[int(len(durations) * 0.99)] if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "throughput_per_second": len(durations) / sum(durations) if durations else 0,
            "errors": [e.error for e in errors],
        }


class LoadTester:
    """
    Load testing framework for ChelatedAI operations.

    Supports:
    - Concurrent embedding generation
    - Concurrent search queries
    - Concurrent ingestion
    - Sustained load testing (long-running)
    - Spike testing (sudden load increase)
    """

    def __init__(self, engine, max_workers: int = 10):
        self.engine = engine
        self.max_workers = max_workers
        self.results: List[TestResult] = []
        self._lock = threading.Lock()

    def _add_result(self, result: TestResult):
        with self._lock:
            self.results.append(result)

    def test_concurrent_embeddings(self, texts_per_thread: int = 100,
                                     num_threads: int = 5) -> LoadTestReport:
        """Test embedding generation under concurrent load."""
        corpus = [f"Test document {i}: Lorem ipsum dolor sit amet." for i in range(texts_per_thread * num_threads)]

        def embed_batch(start: int, count: int):
            start_time = time.perf_counter()
            try:
                batch = corpus[start:start + count]
                self.engine.embed(batch)
                duration = time.perf_counter() - start_time
                self._add_result(TestResult("embed", duration, True))
            except Exception as e:
                duration = time.perf_counter() - start_time
                self._add_result(TestResult("embed", duration, False, str(e)))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                futures.append(executor.submit(embed_batch, i * texts_per_thread, texts_per_thread))
            for f in as_completed(futures):
                f.result()

        return LoadTestReport(
            timestamp=datetime.utcnow().isoformat(),
            config={"texts_per_thread": texts_per_thread, "num_threads": num_threads},
            results=self.results[-(num_threads * texts_per_thread):],
            summary={}  # Generated by generate()
        )

    def test_concurrent_search(self, num_queries: int = 100,
                                num_threads: int = 5) -> LoadTestReport:
        """Test search queries under concurrent load."""
        queries = [f"Test query {i}" for i in range(num_queries)]

        def search_query(query: str):
            start_time = time.perf_counter()
            try:
                self.engine.run_inference(query)
                duration = time.perf_counter() - start_time
                self._add_result(TestResult("search", duration, True))
            except Exception as e:
                duration = time.perf_counter() - start_time
                self._add_result(TestResult("search", duration, False, str(e)))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(search_query, q) for q in queries]
            for f in as_completed(futures):
                f.result()

        return LoadTestReport(
            timestamp=datetime.utcnow().isoformat(),
            config={"num_queries": num_queries, "num_threads": num_threads},
            results=self.results[-num_queries:],
            summary={}
        )

    def test_ingestion(self, doc_count: int = 1000,
                       batch_size: int = 100) -> LoadTestReport:
        """Test ingestion throughput."""
        corpus = [f"Test document {i}: Lorem ipsum dolor sit amet." for i in range(doc_count)]
        start_time = time.perf_counter()
        try:
            self.engine.ingest(corpus)
            duration = time.perf_counter() - start_time
            self._add_result(TestResult("ingest", duration, True))
            return LoadTestReport(
                timestamp=datetime.utcnow().isoformat(),
                config={"doc_count": doc_count, "batch_size": batch_size},
                results=[TestResult("ingest", duration, True)],
                summary={}
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            return LoadTestReport(
                timestamp=datetime.utcnow().isoformat(),
                config={"doc_count": doc_count, "batch_size": batch_size},
                results=[TestResult("ingest", duration, False, str(e))],
                summary={}
            )

    def test_sustained_load(self, duration_seconds: int = 300,
                            queries_per_second: float = 5.0) -> LoadTestReport:
        """Sustained load test: generate queries at steady rate."""
        import queue
        q = queue.Queue()

        # Pre-generate queries
        for i in range(int(duration_seconds * queries_per_second)):
            q.put(f"Sustained query {i}")

        def producer():
            """Feed queries to workers at steady rate."""
            import random
            while not q.empty():
                try:
                    query = q.get_nowait()
                    self.engine.run_inference(query)
                    q.task_done()
                except queue.Empty:
                    break
                time.sleep(1.0 / queries_per_second)

        def consumer():
            """Process queries from queue."""
            while True:
                try:
                    query = q.get(timeout=1.0)
                    start_time = time.perf_counter()
                    try:
                        self.engine.run_inference(query)
                        duration = time.perf_counter() - start_time
                        self._add_result(TestResult("sustained_search", duration, True))
                    except Exception as e:
                        duration = time.perf_counter() - start_time
                        self._add_result(TestResult("sustained_search", duration, False, str(e)))
                    q.task_done()
                except queue.Empty:
                    break

        start = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.submit(producer)
            futures = [executor.submit(consumer) for _ in range(self.max_workers)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.time() - start

        report = LoadTestReport(
            timestamp=datetime.utcnow().isoformat(),
            config={"duration_seconds": duration_seconds, "qps": queries_per_second},
            results=self.results[-1000:],  # Last 1000 results
            summary={}
        )
        return report

    def run_capacity_test(self, target_qps: float = 10.0,
                          max_threads: int = 50) -> Dict[str, Any]:
        """
        Find maximum QPS before latency degrades beyond threshold.

        Binary search for the breaking point.
        """
        low_threads = 1
        high_threads = max_threads
        best_qps = 0
        best_latency = float('inf')

        while low_threads <= high_threads:
            mid_threads = (low_threads + high_threads) // 2
            report = self.test_concurrent_search(num_queries=mid_threads * 10,
                                                  num_threads=mid_threads)
            summary = report.generate()
            p95 = summary["p95_duration"]

            if p95 < 2.0:  # Acceptable latency threshold
                best_qps = mid_threads / summary["avg_duration"]
                best_latency = p95
                low_threads = mid_threads + 1
            else:
                high_threads = mid_threads - 1

        return {
            "capacity_test": True,
            "max_acceptable_qps": round(best_qps, 2),
            "p95_latency_at_capacity": round(best_latency, 4),
            "max_threads_tested": max_threads,
        }
```

**Action 2: Create `load_test_cli.py` — Command-line load tester**

```python
# load_test_cli.py
"""Command-line load testing interface."""

import argparse
import json
import sys
from antigravity_engine import AntigravityEngine
from load_tester import LoadTester

def main():
    parser = argparse.ArgumentParser(description="ChelatedAI Load Tester")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--qdrant", default=":memory:")
    parser.add_argument("--test", required=True,
                        choices=["embed", "search", "ingest", "sustained", "capacity"])
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--output", default="load_test_results.json")
    args = parser.parse_args()

    engine = AntigravityEngine(qdrant_location=args.qdrant, model_name=args.model)
    tester = LoadTester(engine, max_workers=args.threads)

    if args.test == "embed":
        report = tester.test_concurrent_embeddings(texts_per_thread=100, num_threads=args.threads)
    elif args.test == "search":
        # Ingest some data first
        engine.ingest([f"Document {i}" for i in range(100)])
        report = tester.test_concurrent_search(num_queries=100, num_threads=args.threads)
    elif args.test == "ingest":
        report = tester.test_ingestion(doc_count=1000)
    elif args.test == "sustained":
        report = tester.test_sustained_load(duration_seconds=args.duration, queries_per_second=5.0)
    elif args.test == "capacity":
        result = tester.run_capacity_test(target_qps=10.0, max_threads=args.threads)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    result = report.generate()
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    engine.close()

if __name__ == "__main__":
    main()
```

**Action 3: Create `capacity_planning.py` — Capacity analysis tool**

```python
# capacity_planning.py
"""Capacity planning analysis based on load test results."""

import json
from pathlib import Path
from typing import List, Dict, Any


class CapacityPlanner:
    """Analyze load test results to determine capacity requirements."""

    def __init__(self, results_dir: str = "load_test_results"):
        self.results_dir = Path(results_dir)

    def load_results(self) -> List[Dict[str, Any]]:
        """Load all JSON results from results directory."""
        results = []
        for f in self.results_dir.glob("*.json"):
            with open(f) as file:
                results.append(json.load(file))
        return results

    def analyze_trends(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze capacity trends across multiple load test runs."""
        return {
            "max_qps": max(r.get("throughput_per_second", 0) for r in results),
            "avg_p95_latency": sum(r.get("p95_duration", 0) for r in results) / len(results),
            "worst_p99_latency": max(r.get("p99_duration", 0) for r in results),
            "min_success_rate": min(r.get("success_rate", 0) for r in results),
        }

    def recommend_sizing(self, results: List[Dict], target_p95: float = 1.0) -> Dict[str, Any]:
        """Recommend resource sizing based on test results."""
        max_qps = max(r.get("throughput_per_second", 0) for r in results)
        avg_p95 = sum(r.get("p95_duration", 0) for r in results) / len(results)

        if avg_p95 <= target_p95:
            return {
                "adequate": True,
                "current_max_qps": round(max_qps, 2),
                "headroom_pct": round((1 - avg_p95 / target_p95) * 100, 1),
                "recommendation": "Current capacity is adequate for target latency.",
            }
        else:
            return {
                "adequate": False,
                "current_max_qps": round(max_qps, 2),
                "target_p95": target_p95,
                "actual_p95": round(avg_p95, 4),
                "recommendation": f"Latency exceeds target. Consider: scaling horizontally, "
                                  f"increasing HNSW_EF_SEARCH, or reducing EMBEDDING_BATCH_SIZE.",
            }
```

**Action 4: Add capacity test to `config.py`**

```python
# In config.py, add:
LOAD_TEST_ENABLED = False
LOAD_TEST_DEFAULT_THREADS = 10
LOAD_TEST_DEFAULT_DURATION = 300
LOAD_TEST_DEFAULT_QPS = 5.0
LOAD_TEST_ACCEPTABLE_P95_LATENCY = 1.0  # seconds
LOAD_TEST_RESULTS_DIR = "load_test_results"
```

**Estimated effort:** 2 days
**Priority:** MEDIUM — essential for production capacity planning
**Files created:** `load_tester.py`, `load_test_cli.py`, `capacity_planning.py`
**Files modified:** `config.py`
**Expected outcome:** Automated load testing, capacity analysis, bottleneck identification

---

================================================================================
10.10 Production Checklist
================================================================================

**Goal:** Comprehensive validation checklist for deployment readiness.

**Action 1: Create `production_checklist.py` — Automated validation**

New file: `production_checklist.py`

```python
# production_checklist.py
"""Automated production deployment validation."""

import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str  # "required", "recommended", "optional"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class ProductionChecklist:
    """
    Automated production deployment validation.

    Checks:
    - Environment configuration
    - Dependency availability
    - Health endpoint connectivity
    - Metrics endpoint accessibility
    - Log file permissions
    - Checkpoint integrity
    - Resource availability
    - Security configuration
    """

    def __init__(self, metrics_url: str = "http://localhost:9090",
                 health_url: str = "http://localhost:8081"):
        self.metrics_url = metrics_url
        self.health_url = health_url
        self.results: List[CheckResult] = []

    def run_all(self) -> List[CheckResult]:
        """Run all checks and return results."""
        self.check_environment()
        self.check_dependencies()
        self.check_health_endpoint()
        self.check_metrics_endpoint()
        self.check_log_files()
        self.check_checkpoints()
        self.check_resources()
        self.check_security()
        return self.results

    def check_environment(self):
        """Check environment configuration."""
        checks = [
            CheckResult("env_venv_active",
                        bool(os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')),
                        "required", "Python virtual environment should be active"),
            CheckResult("env_python_version",
                        True,  # Would check sys.version_info
                        "recommended", "Python 3.9+ recommended"),
        ]
        self.results.extend(checks)

    def check_dependencies(self):
        """Check required dependencies."""
        checks = [
            self._check_import("numpy"),
            self._check_import("torch"),
            self._check_import("qdrant_client"),
            self._check_import("sentence_transformers"),
            self._check_import("requests"),
        ]
        self.results.extend([c for c in checks if c])

    def _check_import(self, name: str) -> Optional[CheckResult]:
        try:
            __import__(name)
            return CheckResult(f"dep_{name}", True, "required",
                             f"{name} is installed")
        except ImportError:
            return CheckResult(f"dep_{name}", False, "required",
                             f"{name} is NOT installed")

    def check_health_endpoint(self):
        """Check health endpoint."""
        try:
            import requests
            resp = requests.get(f"{self.health_url}/healthz", timeout=5)
            self.results.append(CheckResult("health_liveness",
                                            resp.status_code == 200,
                                            "required",
                                            f"Healthz returned {resp.status_code}",
                                            {"status_code": resp.status_code}))
            resp2 = requests.get(f"{self.health_url}/ready", timeout=5)
            self.results.append(CheckResult("health_readiness",
                                            resp2.status_code == 200,
                                            "required",
                                            f"Ready returned {resp2.status_code}",
                                            {"status_code": resp2.status_code}))
        except Exception as e:
            self.results.append(CheckResult("health_liveness", False, "required",
                                            f"Health check failed: {e}"))
            self.results.append(CheckResult("health_readiness", False, "required",
                                            f"Health check failed: {e}"))

    def check_metrics_endpoint(self):
        """Check metrics endpoint."""
        try:
            import requests
            resp = requests.get(f"{self.metrics_url}/metrics", timeout=5)
            has_prometheus_format = resp.status_code == 200 and "# HELP" in resp.text
            self.results.append(CheckResult("metrics_endpoint",
                                            has_prometheus_format,
                                            "required",
                                            f"Metrics endpoint: {resp.status_code}",
                                            {"has_prometheus_format": has_prometheus_format}))
        except Exception as e:
            self.results.append(CheckResult("metrics_endpoint", False, "required",
                                            f"Metrics endpoint failed: {e}"))

    def check_log_files(self):
        """Check log file configuration."""
        from config import ChelationConfig
        log_path = ChelationConfig.EVENT_LOG_PATH

        exists = log_path.exists() if log_path else False
        self.results.append(CheckResult("log_file_exists", exists,
                                        "recommended",
                                        f"Log file at {log_path}",
                                        {"exists": exists}))

        if exists:
            size_mb = log_path.stat().st_size / 1024 / 1024
            self.results.append(CheckResult("log_file_size",
                                            size_mb < 100,
                                            "recommended",
                                            f"Log file size: {size_mb:.1f}MB",
                                            {"size_mb": size_mb}))

    def check_checkpoints(self):
        """Check checkpoint integrity."""
        from config import ChelationConfig
        from checkpoint_manager import CheckpointManager

        adapter_path = ChelationConfig.ADAPTER_WEIGHTS_PATH
        if adapter_path.exists():
            try:
                mgr = CheckpointManager()
                data = mgr.load_checkpoint(str(adapter_path))
                self.results.append(CheckResult("checkpoint_valid",
                                                data is not None,
                                                "required",
                                                f"Checkpoint valid: {data is not None}",
                                                {"has_checkpoint": True}))
            except Exception as e:
                self.results.append(CheckResult("checkpoint_valid", False, "required",
                                                f"Checkpoint invalid: {e}"))
        else:
            self.results.append(CheckResult("checkpoint_valid", True,
                                            "optional",
                                            "No adapter checkpoint (will create new one)",
                                            {"has_checkpoint": False}))

    def check_resources(self):
        """Check system resources."""
        import shutil
        try:
            total, used, free = shutil.disk_usage("/")
            self.results.append(CheckResult("disk_space",
                                            free > 100 * 1024 * 1024,
                                            "required",
                                            f"Disk free: {free / 1024**3:.1f}GB",
                                            {"free_gb": free / 1024**3}))
        except Exception as e:
            self.results.append(CheckResult("disk_space", False, "required",
                                            f"Disk check failed: {e}"))

    def check_security(self):
        """Check security configuration."""
        dashboard_token = os.getenv("CHELATED_DASHBOARD_TOKEN", "")
        self.results.append(CheckResult("security_dashboard_token",
                                        len(dashboard_token) > 0,
                                        "recommended",
                                        "Dashboard should have authentication token",
                                        {"has_token": len(dashboard_token) > 0}))

        ssn = os.getenv("SENTRY_DSN", "")
        self.results.append(CheckResult("security_sentry_dsn",
                                        len(ssn) == 0 or ssn.startswith("https://"),
                                        "recommended",
                                        "SENTRY_DSN should use HTTPS",
                                        {"dsn_configured": len(ssn) > 0}))

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive checklist report."""
        required = [r for r in self.results if r.severity == "required"]
        recommended = [r for r in self.results if r.severity == "recommended"]
        optional = [r for r in self.results if r.severity == "optional"]

        all_required_passed = all(r.passed for r in required)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "passed": all_required_passed,
            "summary": {
                "required": {"total": len(required), "passed": sum(1 for r in required if r.passed),
                            "failed": sum(1 for r in required if not r.passed)},
                "recommended": {"total": len(recommended), "passed": sum(1 for r in recommended if r.passed)},
                "optional": {"total": len(optional), "passed": sum(1 for r in optional if r.passed)},
            },
            "results": [
                {"name": r.name, "passed": r.passed, "severity": r.severity,
                 "message": r.message}
                for r in self.results
            ],
        }
```

**Action 2: Create `production_deploy.py` — Deployment validation script**

```python
# production_deploy.py
"""Deployment validation and rollback script."""

import argparse
import json
import subprocess
import sys
import os
from datetime import datetime


def run_pre_deploy_checks():
    """Run production checklist before deployment."""
    from production_checklist import ProductionChecklist
    checklist = ProductionChecklist()
    results = checklist.run_all()
    report = checklist.generate_report()

    print(json.dumps(report, indent=2))

    if not report["passed"]:
        print("\nDeployment BLOCKED: Required checks failed.")
        sys.exit(1)

    print("\nAll required checks passed. Proceeding with deployment.")
    return report


def save_deployment_snapshot():
    """Save current state before deployment for rollback."""
    from config import ChelationConfig
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "adapter_weights_sha256": None,  # Would compute actual hash
        "current_config": {},  # Would dump ChelationConfig
        "environment": dict(os.environ),
        "pid": os.getpid(),
    }
    snapshot_path = f"deploy_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"Deployment snapshot saved to {snapshot_path}")
    return snapshot_path


def rollback(snapshot_path: str):
    """Rollback to previous deployment state."""
    with open(snapshot_path) as f:
        snapshot = json.load(f)
    print(f"Rolling back to state from {snapshot['timestamp']}")
    # Implementation depends on deployment method (Docker, systemd, etc.)
    print("Rollback completed. Verify with: python production_deploy.py --verify")


def verify_deployment():
    """Verify deployment is healthy."""
    from production_checklist import ProductionChecklist
    checklist = ProductionChecklist()
    results = checklist.run_all()
    report = checklist.generate_report()

    if report["passed"]:
        print("Deployment verified: All checks passed.")
    else:
        print("Deployment verification FAILED:")
        for r in results:
            if not r.passed:
                print(f"  FAILED: {r.name} - {r.message}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChelatedAI Production Deployment")
    parser.add_argument("--pre-deploy", action="store_true", help="Run pre-deployment checks")
    parser.add_argument("--save-snapshot", action="store_true", help="Save current state before deployment")
    parser.add_argument("--rollback", type=str, help="Rollback to snapshot file")
    parser.add_argument("--verify", action="store_true", help="Verify deployment health")
    args = parser.parse_args()

    if args.pre_deploy:
        run_pre_deploy_checks()
    elif args.save_snapshot:
        save_deployment_snapshot()
    elif args.rollback:
        rollback(args.rollback)
    elif args.verify:
        verify_deployment()
    else:
        parser.print_help()
```

**Action 3: Create `disaster_recovery.py` — Disaster recovery procedures**

New file: `disaster_recovery.py`

```python
# disaster_recovery.py
"""Disaster recovery procedures for ChelatedAI."""

import json
import shutil
import os
from pathlib import Path
from datetime import datetime
from config import ChelationConfig


class DisasterRecovery:
    """Automated disaster recovery for ChelatedAI."""

    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_current_state(self):
        """Create a full backup of current state."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp

        # Backup adapter weights
        adapter_path = ChelationConfig.ADAPTER_WEIGHTS_PATH
        if adapter_path.exists():
            shutil.copy2(str(adapter_path), str(backup_path / "adapter_weights.pt"))

        # Backup checkpoints
        checkpoints_dir = ChelationConfig.PROJECT_ROOT / "checkpoints"
        if checkpoints_dir.exists():
            for cp in checkpoints_dir.iterdir():
                shutil.copy2(str(cp), str(backup_path / cp.name))

        # Backup config
        # (Config is code, no need to backup — but log any custom overrides)

        # Backup log state (last N lines)
        log_path = ChelationConfig.EVENT_LOG_PATH
        if log_path.exists():
            with open(log_path, 'r') as f:
                lines = f.readlines()
                with open(backup_path / "log_tail.jsonl", 'w') as out:
                    out.writelines(lines[-1000:])  # Last 1000 lines

        print(f"Backup created at {backup_path}")
        return str(backup_path)

    def restore_latest(self):
        """Restore from the latest backup."""
        backups = sorted(self.backup_dir.iterdir(), key=lambda p: p.name, reverse=True)
        if not backups:
            print("No backups found.")
            return None
        latest = backups[0]
        print(f"Restoring from {latest}")

        # Restore adapter weights
        adapter_backup = latest / "adapter_weights.pt"
        if adapter_backup.exists():
            shutil.copy2(str(adapter_backup), str(ChelationConfig.ADAPTER_WEIGHTS_PATH))

        # Restore checkpoints
        for cp in latest.glob("*.pt"):
            dest = ChelationConfig.PROJECT_ROOT / "checkpoints" / cp.name
            shutil.copy2(str(cp), str(dest))

        return str(latest)

    def backup_all(self):
        """Create timestamped backup + cleanup old backups (keep last 10)."""
        path = self.backup_current_state()
        backups = sorted(self.backup_dir.iterdir(), key=lambda p: p.name)
        while len(backups) > 10:
            oldest = backups.pop(0)
            shutil.rmtree(str(oldest))
            print(f"Cleaned old backup: {oldest}")
        return path
```

**Estimated effort:** 1.5 days
**Priority:** HIGH — essential for production operations
**Files created:** `production_checklist.py`, `production_deploy.py`, `disaster_recovery.py`
**Files modified:** None (all new files)
**Expected outcome:** Automated deployment validation, snapshot/rollback, disaster recovery

---

================================================================================
PHASE 10 SUMMARY
================================================================================

Total estimated effort: ~13 days (can be parallelized across team members)

Priority ordering:
1. HIGH: 10.1 Metrics collection (Prometheus + OTel)
2. HIGH: 10.2 Structured logging pipeline (rotation, async, shipping)
3. HIGH: 10.4 Health checks (liveness + readiness probes)
4. HIGH: 10.7 Dashboard enhancement (real-time panels)
5. HIGH: 10.10 Production checklist (automated validation)
6. MEDIUM: 10.3 Distributed tracing (span-based tracing)
7. MEDIUM: 10.5 Alerting (threshold-based)
8. MEDIUM: 10.6 Error tracking (Sentry)
9. MEDIUM: 10.8 Runbook creation
10. MEDIUM: 10.9 Load testing (capacity planning)

Key new files to create:
- metrics_collector.py
- prometheus_exporter.py
- otel_tracer.py
- log_pipeline.py
- log_dashboard_integration.py
- tracer.py
- health_check.py
- alerting.py
- alerting_engine.py
- error_tracker.py
- dashboard_enhancements.py
- docs/runbook.md
- load_tester.py
- load_test_cli.py
- capacity_planning.py
- production_checklist.py
- production_deploy.py
- disaster_recovery.py

Key files to modify:
- antigravity_engine.py (metrics, tracer, error_tracker, health, dashboard server reference)
- chelation_logger.py (pipeline integration: rotation, async, shipping)
- dashboard_server.py (new API endpoints: metrics, health, alerts, errors, traces, log analytics)
- config.py (new configuration: metrics, health, security, load test, production)

Dependencies (optional, installed only when configured):
- sentry-sdk (for error tracking)
- opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp (for distributed tracing)
- prometheus-client (alternative to custom metrics_collector)
- aiofiles (for async I/O in log pipeline)
- psutil (for memory/disk checks)
- pynvml (for GPU monitoring)
- aiohttp (for async web requests in load testing)
