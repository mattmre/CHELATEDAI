#!/usr/bin/env python3
"""
Dashboard Server for Chelation Log Visualization

A lightweight HTTP server using only Python standard library to serve
a web dashboard for visualizing chelation_events.jsonl logs.

Usage:
    python dashboard_server.py [--host HOST] [--port PORT] [--log-file PATH]

Example:
    python dashboard_server.py --host localhost --port 8080 --log-file chelation_events.jsonl
"""

import argparse
import hmac
import json
import os
from collections import Counter
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


# Global configuration
LOG_FILE_PATH = "chelation_events.jsonl"
DASHBOARD_TOKEN = os.getenv("CHELATED_DASHBOARD_TOKEN", "").strip()
DASHBOARD_CORS_ORIGIN = os.getenv("CHELATED_DASHBOARD_CORS_ORIGIN", "").strip()


def get_inline_dashboard_html():
    """Return inline dashboard HTML for when dashboard/index.html doesn't exist."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chelation Events Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        h1 {
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 14px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .metric-label {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #333;
            font-size: 36px;
            font-weight: bold;
        }

        .metric-breakdown {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }

        .breakdown-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            color: #666;
            font-size: 14px;
        }

        .breakdown-value {
            font-weight: bold;
            color: #667eea;
        }

        .events-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        h2 {
            color: #333;
            font-size: 24px;
        }

        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        select, button {
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            cursor: pointer;
            background: white;
        }

        button {
            background: #667eea;
            color: white;
            border: none;
            font-weight: 500;
            transition: background 0.2s;
        }

        button:hover {
            background: #5568d3;
        }

        .table-container {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        thead {
            background: #f8f9fa;
        }

        th {
            text-align: left;
            padding: 12px;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #e9ecef;
        }

        td {
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
            color: #666;
            font-size: 14px;
        }

        tr:hover {
            background: #f8f9fa;
        }

        .action-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .action-fast {
            background: #d4edda;
            color: #155724;
        }

        .action-adapt {
            background: #fff3cd;
            color: #856404;
        }

        .action-deep {
            background: #cce5ff;
            color: #004085;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
            font-style: italic;
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #f5c6cb;
        }

        .time-range {
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }

        .refresh-status {
            color: #666;
            font-size: 12px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Chelation Events Dashboard</h1>
            <p class="subtitle">Real-time visualization of chelation_events.jsonl</p>
        </header>

        <div id="error-container"></div>

        <div class="metrics-grid" id="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Events</div>
                <div class="metric-value" id="total-events">-</div>
                <div class="time-range" id="time-range"></div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Query Events</div>
                <div class="metric-value" id="query-count">-</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Error Events</div>
                <div class="metric-value" id="error-count">-</div>
            </div>

            <div class="metric-card">
                <div class="metric-label">Action Breakdown</div>
                <div class="metric-breakdown" id="action-breakdown">
                    <div class="loading">Loading...</div>
                </div>
            </div>
        </div>

        <div class="events-section">
            <div class="section-header">
                <h2>Recent Events</h2>
                <div class="controls">
                    <select id="event-filter">
                        <option value="">All Events</option>
                        <option value="query">Query Events</option>
                        <option value="error">Error Events</option>
                    </select>
                    <select id="limit-select">
                        <option value="10">10 events</option>
                        <option value="25" selected>25 events</option>
                        <option value="50">50 events</option>
                        <option value="100">100 events</option>
                    </select>
                    <button onclick="refreshData()">🔄 Refresh</button>
                </div>
            </div>

            <div class="refresh-status" id="refresh-status"></div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Query</th>
                            <th>Action</th>
                            <th>Variance</th>
                            <th>Top IDs</th>
                        </tr>
                    </thead>
                    <tbody id="events-tbody">
                        <tr>
                            <td colspan="5" class="loading">Loading events...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let currentData = {
            summary: null,
            events: null
        };

        function formatTimestamp(timestamp) {
            if (!timestamp) return '-';
            const date = new Date(timestamp * 1000);
            return date.toLocaleString();
        }

        function formatNumber(num) {
            if (num === null || num === undefined) return '-';
            return num.toLocaleString();
        }

        function getActionBadgeClass(action) {
            if (!action) return '';
            const actionLower = action.toLowerCase();
            if (actionLower === 'fast') return 'action-fast';
            if (actionLower === 'adapt') return 'action-adapt';
            if (actionLower === 'deep') return 'action-deep';
            return '';
        }

        async function loadSummary() {
            try {
                const response = await fetch('/api/summary');
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                const data = await response.json();
                currentData.summary = data;
                updateSummaryUI(data);
                clearError();
            } catch (error) {
                showError(`Failed to load summary: ${error.message}`);
            }
        }

        async function loadEvents() {
            try {
                const eventType = document.getElementById('event-filter').value;
                const limit = document.getElementById('limit-select').value;
                
                let url = `/api/events?limit=${limit}`;
                if (eventType) {
                    url += `&event_type=${eventType}`;
                }

                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                const data = await response.json();
                currentData.events = data.events;
                updateEventsUI(data.events);
                clearError();
                updateRefreshStatus();
            } catch (error) {
                showError(`Failed to load events: ${error.message}`);
            }
        }

        function updateSummaryUI(summary) {
            document.getElementById('total-events').textContent = formatNumber(summary.total_events);
            document.getElementById('query-count').textContent = formatNumber(summary.query_count);
            document.getElementById('error-count').textContent = formatNumber(summary.error_count);

            const timeRange = document.getElementById('time-range');
            if (summary.time_range && summary.time_range.earliest && summary.time_range.latest) {
                const earliest = formatTimestamp(summary.time_range.earliest);
                const latest = formatTimestamp(summary.time_range.latest);
                timeRange.textContent = `${earliest} → ${latest}`;
            } else {
                timeRange.textContent = '';
            }

            const breakdownDiv = document.getElementById('action-breakdown');
            breakdownDiv.replaceChildren();
            if (summary.action_breakdown && Object.keys(summary.action_breakdown).length > 0) {
                for (const [action, count] of Object.entries(summary.action_breakdown)) {
                    const item = document.createElement('div');
                    item.className = 'breakdown-item';
                    const label = document.createElement('span');
                    label.textContent = action;
                    const value = document.createElement('span');
                    value.className = 'breakdown-value';
                    value.textContent = formatNumber(count);
                    item.appendChild(label);
                    item.appendChild(value);
                    breakdownDiv.appendChild(item);
                }
            } else {
                const noData = document.createElement('div');
                noData.className = 'loading';
                noData.textContent = 'No actions recorded';
                breakdownDiv.appendChild(noData);
            }
        }

        function updateEventsUI(events) {
            const tbody = document.getElementById('events-tbody');
            tbody.replaceChildren();

            if (!events || events.length === 0) {
                const tr = document.createElement('tr');
                const td = document.createElement('td');
                td.colSpan = 5;
                td.className = 'loading';
                td.textContent = 'No events found';
                tr.appendChild(td);
                tbody.appendChild(tr);
                return;
            }

            for (const event of events) {
                const tr = document.createElement('tr');

                const tdTime = document.createElement('td');
                tdTime.textContent = formatTimestamp(event.timestamp);
                tr.appendChild(tdTime);

                const tdQuery = document.createElement('td');
                tdQuery.textContent = event.query_snippet || '-';
                tr.appendChild(tdQuery);

                const tdAction = document.createElement('td');
                const badge = document.createElement('span');
                const action = event.action || '-';
                badge.className = 'action-badge ' + getActionBadgeClass(action);
                badge.textContent = action;
                tdAction.appendChild(badge);
                tr.appendChild(tdAction);

                const tdVariance = document.createElement('td');
                tdVariance.textContent = event.global_variance ? event.global_variance.toFixed(6) : '-';
                tr.appendChild(tdVariance);

                const tdIds = document.createElement('td');
                tdIds.textContent = event.top_10_ids ? event.top_10_ids.slice(0, 5).join(', ') + '...' : '-';
                tr.appendChild(tdIds);

                tbody.appendChild(tr);
            }
        }

        function showError(message) {
            const container = document.getElementById('error-container');
            container.replaceChildren();
            const div = document.createElement('div');
            div.className = 'error';
            const strong = document.createElement('strong');
            strong.textContent = 'Error:';
            div.appendChild(strong);
            div.appendChild(document.createTextNode(' ' + message));
            container.appendChild(div);
        }

        function clearError() {
            document.getElementById('error-container').replaceChildren();
        }

        function updateRefreshStatus() {
            const status = document.getElementById('refresh-status');
            const now = new Date().toLocaleTimeString();
            status.textContent = `Last updated: ${now}`;
        }

        async function refreshData() {
            await Promise.all([loadSummary(), loadEvents()]);
        }

        document.getElementById('event-filter').addEventListener('change', loadEvents);
        document.getElementById('limit-select').addEventListener('change', loadEvents);

        refreshData();
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
"""


def load_events(log_file: str = LOG_FILE_PATH) -> List[Dict[str, Any]]:
    """
    Load all events from a JSONL log file.

    Args:
        log_file: Path to the JSONL log file

    Returns:
        List of event dictionaries

    Raises:
        FileNotFoundError: If log file doesn't exist
        json.JSONDecodeError: If log file contains invalid JSON
    """
    events = []
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_num}: {e.msg}",
                    e.doc,
                    e.pos
                )
    
    return events


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from a list of events.

    Args:
        events: List of event dictionaries

    Returns:
        Dictionary containing summary statistics:
        - total_events: Total number of events
        - query_count: Number of query events
        - error_count: Number of error events
        - action_breakdown: Count of events by action type
        - time_range: Earliest and latest timestamps
    """
    if not events:
        return {
            "total_events": 0,
            "query_count": 0,
            "error_count": 0,
            "action_breakdown": {},
            "time_range": {"earliest": None, "latest": None}
        }
    
    total_events = len(events)
    query_count = sum(1 for e in events if "query_snippet" in e)
    error_count = sum(1 for e in events if e.get("event_type") == "error" or "error" in e)
    
    # Count actions
    actions = [e.get("action") for e in events if "action" in e]
    action_breakdown = dict(Counter(actions))
    
    # Get time range
    timestamps = [e.get("timestamp") for e in events if "timestamp" in e]
    time_range = {
        "earliest": min(timestamps) if timestamps else None,
        "latest": max(timestamps) if timestamps else None
    }
    
    return {
        "total_events": total_events,
        "query_count": query_count,
        "error_count": error_count,
        "action_breakdown": action_breakdown,
        "time_range": time_range
    }


def filter_events(
    events: List[Dict[str, Any]],
    event_type: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter events by type and optionally limit the number of results.

    Args:
        events: List of event dictionaries
        event_type: Filter by event type (e.g., "query", "error")
        limit: Maximum number of events to return (most recent first)

    Returns:
        Filtered list of events
    """
    filtered = events
    
    # Filter by event type
    if event_type:
        if event_type == "query":
            filtered = [e for e in filtered if "query_snippet" in e]
        elif event_type == "error":
            filtered = [e for e in filtered if e.get("event_type") == "error" or "error" in e]
        else:
            # Generic event_type field filter
            filtered = [e for e in filtered if e.get("event_type") == event_type]
    
    # Sort by timestamp (most recent first)
    filtered = sorted(
        filtered,
        key=lambda e: e.get("timestamp", 0),
        reverse=True
    )
    
    # Apply limit (0 means no rows)
    if limit is not None:
        filtered = filtered[:max(0, limit)]
    
    return filtered


class DashboardHandler(SimpleHTTPRequestHandler):
    """
    HTTP request handler for the dashboard server.
    
    Serves static files and provides JSON API endpoints.
    """
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve static files from
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def do_GET(self):
        """Handle GET requests for API endpoints and static files."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)

        if DASHBOARD_TOKEN and not self._is_api_authorized():
            self.send_error_response(401, "Unauthorized")
            return
        
        # API endpoints
        if path == "/api/events":
            self.handle_api_events(query_params)
        elif path == "/api/summary":
            self.handle_api_summary()
        elif path == "/api/sweep_results":
            self.handle_api_sweep_results(query_params)
        elif path == "/api/test_results":
            self.handle_api_test_results()
        elif path == "/" or path == "/dashboard" or path == "/dashboard/":
            # Redirect to dashboard page
            self.serve_dashboard()
        else:
            # Serve static files
            super().do_GET()

    def _is_api_authorized(self) -> bool:
        """Validate API access token when dashboard token auth is configured."""
        if not DASHBOARD_TOKEN:
            return True
        auth_header = ""
        if hasattr(self, "headers") and self.headers is not None:
            auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        token = auth_header.split(" ", 1)[1].strip()
        return hmac.compare_digest(token, DASHBOARD_TOKEN)
    
    def handle_api_events(self, query_params: Dict[str, List[str]]):
        """
        Handle /api/events endpoint.
        
        Query parameters:
        - limit: Maximum number of events to return
        - event_type: Filter by event type (e.g., "query", "error")
        """
        try:
            events = load_events(LOG_FILE_PATH)
            
            # Extract query parameters
            limit = None
            if "limit" in query_params:
                try:
                    limit = int(query_params["limit"][0])
                except (ValueError, IndexError):
                    limit = None
            
            event_type = None
            if "event_type" in query_params:
                try:
                    event_type = query_params["event_type"][0]
                except IndexError:
                    event_type = None
            
            # Filter events
            filtered = filter_events(events, event_type=event_type, limit=limit)
            
            # Send response
            self.send_json_response({"events": filtered})
        
        except FileNotFoundError as e:
            self.send_error_response(404, str(e))
        except Exception as e:
            self.send_error_response(500, f"Internal server error: {str(e)}")
    
    def handle_api_summary(self):
        """Handle /api/summary endpoint."""
        try:
            events = load_events(LOG_FILE_PATH)
            summary = summarize_events(events)
            self.send_json_response(summary)
        
        except FileNotFoundError as e:
            self.send_error_response(404, str(e))
        except Exception as e:
            self.send_error_response(500, f"Internal server error: {str(e)}")
            
    def handle_api_sweep_results(self, query_params: Dict[str, List[str]]):
        """Handle /api/sweep_results endpoint by returning the large sweep JSON data."""
        sweep_file = "large_sweep_results.json"
        try:
            if not os.path.exists(sweep_file):
                self.send_json_response({"results": []})
                return
            
            with open(sweep_file, 'r') as f:
                results = json.load(f)
            self.send_json_response({"results": results})
        except Exception as e:
            self.send_error_response(500, f"Error reading sweep results: {str(e)}")

    def handle_api_test_results(self):
        """Handle /api/test_results endpoint."""
        test_file = ".report.json"
        try:
            if not os.path.exists(test_file):
                self.send_json_response({"summary": None, "tests": []})
                return
            with open(test_file, 'r') as f:
                report = json.load(f)
            self.send_json_response(report)
        except Exception as e:
            self.send_error_response(500, f"Error reading test results: {str(e)}")
            
    def serve_dashboard(self):
        """Serve the dashboard HTML page."""
        dashboard_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dashboard",
            "index.html"
        )
        
        # Try to load from file first
        if os.path.exists(dashboard_path):
            try:
                with open(dashboard_path, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                self.send_error_response(500, f"Error serving dashboard: {str(e)}")
                return
        
        # Fallback to inline HTML if file doesn't exist
        self.serve_inline_dashboard()
    
    def serve_inline_dashboard(self):
        """Serve inline dashboard HTML as fallback."""
        html_content = get_inline_dashboard_html()
        content_bytes = html_content.encode('utf-8')
        
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content_bytes)))
        self.end_headers()
        self.wfile.write(content_bytes)
    
    def send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send a JSON response."""
        response = json.dumps(data, indent=2)
        response_bytes = response.encode('utf-8')
        
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        if DASHBOARD_CORS_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", DASHBOARD_CORS_ORIGIN)
        elif not DASHBOARD_TOKEN:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)
    
    def send_error_response(self, status_code: int, message: str):
        """Send an error response."""
        self.send_json_response({"error": message}, status_code)
    
    def log_message(self, format, *args):
        """Override to customize logging format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")


def _is_loopback_host(host: str) -> bool:
    """Return True if host is loopback/local-only."""
    return host in {"localhost", "127.0.0.1", "::1"}


def run_server(host: str = "127.0.0.1", port: int = 8080, log_file: str = LOG_FILE_PATH):
    """
    Run the dashboard HTTP server.

    Args:
        host: Host address to bind to
        port: Port number to listen on
        log_file: Path to the log file to visualize
    """
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_file

    if not _is_loopback_host(host) and not DASHBOARD_TOKEN:
        raise ValueError(
            "Refusing non-local dashboard bind without CHELATED_DASHBOARD_TOKEN."
        )
    
    server_address = (host, port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print("Dashboard server starting...")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Log file: {log_file}")
    print(f"  Token auth: {'enabled' if DASHBOARD_TOKEN else 'disabled'}")
    print(f"  Dashboard URL: http://{host}:{port}/dashboard/")
    print("\nAPI Endpoints:")
    print(f"  GET http://{host}:{port}/api/events?limit=N")
    print(f"  GET http://{host}:{port}/api/summary")
    print(f"  GET http://{host}:{port}/api/events?event_type=query")
    print("\nPress Ctrl+C to stop the server.")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
        print("Server stopped.")


def main():
    """Parse command-line arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="Dashboard server for Chelation log visualization"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number to listen on (default: 8080)"
    )
    parser.add_argument(
        "--log-file",
        default="chelation_events.jsonl",
        help="Path to the log file (default: chelation_events.jsonl)"
    )
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, log_file=args.log_file)


if __name__ == "__main__":
    main()
