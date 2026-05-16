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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, urlparse

from plan_evidence_artifact_cleanup import plan_evidence_artifact_cleanup
from computational_storage_poc.disk_llm_estimator import (
    DEFAULT_HARDWARE_PROFILES,
    estimate_disk_llm,
)


# Global configuration
LOG_FILE_PATH = "chelation_events.jsonl"
DASHBOARD_TOKEN = os.getenv("CHELATED_DASHBOARD_TOKEN", "").strip()
DASHBOARD_CORS_ORIGIN = os.getenv("CHELATED_DASHBOARD_CORS_ORIGIN", "").strip()
CAMPAIGN_HISTORY_ROOT = "experiment_runs"
VALIDATION_HISTORY_ROOT = "experiment_runs"
PREFLIGHT_HISTORY_ROOT = "experiment_runs"
EVIDENCE_INDEX_PATH = "evidence_index.json"
EVIDENCE_CHAIN_HISTORY_ROOT = "experiment_runs"
EVIDENCE_CLEANUP_ROOT = "experiment_runs"
PHASE_C_RESULTS_PATH = "phase_c_results.json"
PHASE_C_ANALYSIS_PATH = "phase_c_analysis.json"
MODEL_SCOPE_ARTIFACT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiment_runs", "model_scope"
)

# TTS pipeline state — populated by AntigravityEngine.enable_tts() and run_inference()
_TTS_DASHBOARD_STATE: Dict[str, Any] = {
    "enabled": False,
    "config": {},
    "last_result": None,
}


def update_tts_dashboard_state(
    enabled: bool,
    config: Optional[Dict[str, Any]] = None,
    last_result: Optional[Dict[str, Any]] = None,
) -> None:
    """Update TTS dashboard state. Called by AntigravityEngine when TTS is active."""
    _TTS_DASHBOARD_STATE["enabled"] = bool(enabled)
    if config is not None:
        _TTS_DASHBOARD_STATE["config"] = dict(config)
    if last_result is not None:
        _TTS_DASHBOARD_STATE["last_result"] = dict(last_result)


def _tts_result_to_dict(tts_result: Any) -> Dict[str, Any]:
    """Convert a TTSResult object to a JSON-serialisable dict for dashboard API."""
    import numpy as _np

    norm_before = float(_np.linalg.norm(tts_result.original))
    norm_after = float(_np.linalg.norm(tts_result.after_steering))
    translation_offset_norm: Optional[float] = None
    transport_weight_used: Optional[float] = None
    steering_delta_norm: Optional[float] = None

    if tts_result.translation_result is not None:
        translation_offset_norm = float(tts_result.translation_result.offset_norm)
    if tts_result.transport_result is not None:
        transport_weight_used = float(tts_result.transport_result.weight_used)
    if tts_result.steering_meta is not None:
        steering_delta_norm = float(tts_result.steering_meta.get("total_delta_norm", 0.0))

    return {
        "last_result_available": True,
        "norm_before": norm_before,
        "norm_after": norm_after,
        "total_delta_norm": float(tts_result.total_delta_norm),
        "stages_applied": list(tts_result.stages_applied),
        "translation_offset_norm": translation_offset_norm,
        "transport_weight_used": transport_weight_used,
        "steering_delta_norm": steering_delta_norm,
    }


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
                <h2>Phase C Evaluation — Four-Candidate Results</h2>
            </div>
            <div id="phase-c-panel">
                <div class="loading" id="phase-c-loading">Loading Phase C data…</div>
                <div id="phase-c-recommendation" style="margin-bottom:12px;display:none;"></div>
                <div id="phase-c-table-wrap" style="display:none;">
                    <div class="table-container">
                        <table id="phase-c-table">
                            <thead>
                                <tr>
                                    <th>Candidate</th>
                                    <th>NDCG@10 (mean)</th>
                                    <th>Δ vs Baseline</th>
                                    <th>Win Rate</th>
                                    <th>Gate%</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="phase-c-tbody"></tbody>
                        </table>
                    </div>
                </div>
                <div id="phase-c-empty" style="display:none;color:#888;padding:12px;">
                    Phase C results not yet generated — run <code>run_phase_c_eval.py</code> then <code>analyze_phase_c_results.py</code>.
                </div>
            </div>
        </div>

        <div class="events-section" id="model-scope-panel">
            <div class="section-header">
                <h2>Model Scope Observations</h2>
            </div>
            <div id="ms-loading" class="loading">Loading Model Scope data…</div>
            <div id="ms-content" style="display:none;">
                <div class="metrics-grid" style="margin-bottom:16px;">
                    <div class="metric-card">
                        <div class="metric-label">Observation Events</div>
                        <div class="metric-value" id="ms-event-count">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Intervention Records</div>
                        <div class="metric-value" id="ms-intervention-count">-</div>
                    </div>
                </div>
                <h3 style="color:#333;font-size:18px;margin-bottom:12px;">Last Feature Event</h3>
                <div class="table-container" style="margin-bottom:20px;">
                    <table id="ms-features-table">
                        <thead>
                            <tr>
                                <th>Layer ID</th>
                                <th>Feature Count</th>
                                <th>Nonzero Count</th>
                                <th>Extracted At</th>
                            </tr>
                        </thead>
                        <tbody id="ms-features-tbody">
                        </tbody>
                    </table>
                </div>
                <h3 style="color:#333;font-size:18px;margin-bottom:12px;">Last Intervention</h3>
                <div id="ms-last-intervention" style="background:#f8f9fa;border-radius:6px;padding:12px;font-size:13px;color:#555;font-family:monospace;">
                    No interventions recorded.
                </div>
            </div>
            <div id="ms-empty" style="display:none;color:#888;padding:12px;">
                No Model Scope data yet — run a campaign with model-scope enabled to populate this panel.
            </div>
        </div>

        <div class="events-section" id="tts-panel">
            <div class="section-header">
                <h2>TTS Pipeline (Translation &#x2192; Transport &#x2192; Steering)</h2>
            </div>
            <div id="tts-loading" class="loading">Loading TTS data&#x2026;</div>
            <div id="tts-content" style="display:none;">
                <div class="metrics-grid" style="margin-bottom:16px;">
                    <div class="metric-card">
                        <div class="metric-label">Status</div>
                        <div class="metric-value" id="tts-status">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Translation &#x394; (L2)</div>
                        <div class="metric-value" id="tts-translation-delta">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Transport Weight (t)</div>
                        <div class="metric-value" id="tts-transport-weight">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Steering Magnitude</div>
                        <div class="metric-value" id="tts-steering-magnitude">-</div>
                    </div>
                </div>
                <div class="metrics-grid" style="margin-bottom:16px;">
                    <div class="metric-card">
                        <div class="metric-label">Vector Norm (before TTS)</div>
                        <div class="metric-value" id="tts-norm-before">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Vector Norm (after TTS)</div>
                        <div class="metric-value" id="tts-norm-after">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total &#x394; Norm</div>
                        <div class="metric-value" id="tts-total-delta">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Stages Applied</div>
                        <div class="metric-value" id="tts-stages">-</div>
                    </div>
                </div>
            </div>
            <div id="tts-disabled" style="display:none;color:#888;padding:12px;">
                TTS pipeline not enabled &#x2014; call <code>engine.enable_tts()</code> to activate.
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
            await Promise.all([loadSummary(), loadEvents(), loadPhaseC(), loadModelScope(), loadTTS()]);
        }

        async function loadTTS() {
            try {
                const [statusResp, resultResp] = await Promise.all([
                    fetch('/api/tts/status'),
                    fetch('/api/tts/last_result'),
                ]);
                const status = statusResp.ok ? await statusResp.json() : null;
                const result = resultResp.ok ? await resultResp.json() : null;
                renderTTS(status, result);
            } catch (e) {
                const el = document.getElementById('tts-loading');
                if (el) el.textContent = 'Error loading TTS data: ' + e.message;
            }
        }

        function renderTTS(status, result) {
            const loading = document.getElementById('tts-loading');
            const content = document.getElementById('tts-content');
            const disabled = document.getElementById('tts-disabled');
            if (!loading) return;
            loading.style.display = 'none';
            const enabled = status && status.enabled;
            if (!enabled) {
                if (disabled) disabled.style.display = 'block';
                return;
            }
            if (content) content.style.display = 'block';
            const statusEl = document.getElementById('tts-status');
            if (statusEl) statusEl.textContent = '\u2705 Active';
            if (result && result.last_result_available) {
                const setVal = (id, val, fmt) => {
                    const el = document.getElementById(id);
                    if (el) el.textContent = val != null ? fmt(val) : '-';
                };
                setVal('tts-translation-delta', result.translation_offset_norm, v => v.toFixed(6));
                setVal('tts-transport-weight', result.transport_weight_used, v => v.toFixed(4));
                setVal('tts-steering-magnitude', result.steering_delta_norm, v => v.toFixed(6));
                setVal('tts-norm-before', result.norm_before, v => v.toFixed(4));
                setVal('tts-norm-after', result.norm_after, v => v.toFixed(4));
                setVal('tts-total-delta', result.total_delta_norm, v => v.toFixed(6));
                const stagesEl = document.getElementById('tts-stages');
                if (stagesEl) {
                    const stages = result.stages_applied;
                    stagesEl.textContent = (stages && stages.length > 0) ? stages.join(', ') : 'none';
                }
            }
        }

        async function loadModelScope() {
            try {
                const [eventsResp, featuresResp, interventionsResp] = await Promise.all([
                    fetch('/api/model_scope/events'),
                    fetch('/api/model_scope/features'),
                    fetch('/api/model_scope/interventions'),
                ]);
                const events = eventsResp.ok ? await eventsResp.json() : null;
                const features = featuresResp.ok ? await featuresResp.json() : null;
                const interventions = interventionsResp.ok ? await interventionsResp.json() : null;
                renderModelScope(events, features, interventions);
            } catch (e) {
                document.getElementById('ms-loading').textContent = `Error loading Model Scope data: ${e.message}`;
            }
        }

        function renderModelScope(events, features, interventions) {
            const loading = document.getElementById('ms-loading');
            const content = document.getElementById('ms-content');
            const empty = document.getElementById('ms-empty');

            const eventItems = (events && events.items) ? events.items : [];
            const featureItems = (features && features.items) ? features.items : [];
            const interventionItems = (interventions && interventions.items) ? interventions.items : [];

            const hasData = eventItems.length > 0 || featureItems.length > 0 || interventionItems.length > 0;
            loading.style.display = 'none';
            if (!hasData) {
                empty.style.display = 'block';
                return;
            }
            content.style.display = 'block';

            document.getElementById('ms-event-count').textContent = formatNumber(eventItems.length);
            document.getElementById('ms-intervention-count').textContent = formatNumber(interventionItems.length);

            const tbody = document.getElementById('ms-features-tbody');
            tbody.replaceChildren();
            if (featureItems.length > 0) {
                const last = featureItems[featureItems.length - 1];
                const row = document.createElement('tr');
                row.innerHTML = `<td>${last.layer_id != null ? last.layer_id : '-'}</td>` +
                    `<td>${last.feature_count != null ? last.feature_count : '-'}</td>` +
                    `<td>${last.nonzero_count != null ? last.nonzero_count : '-'}</td>` +
                    `<td>${last.extracted_at != null ? last.extracted_at : '-'}</td>`;
                tbody.appendChild(row);
            } else {
                const row = document.createElement('tr');
                row.innerHTML = '<td colspan="4" style="color:#888;">No feature events recorded.</td>';
                tbody.appendChild(row);
            }

            const lastIntervDiv = document.getElementById('ms-last-intervention');
            if (interventionItems.length > 0) {
                const last = interventionItems[interventionItems.length - 1];
                lastIntervDiv.textContent = JSON.stringify(last, null, 2);
            } else {
                lastIntervDiv.textContent = 'No interventions recorded.';
            }
        }

        async function loadPhaseC() {
            try {
                const [resultsResp, analysisResp] = await Promise.all([
                    fetch('/api/phase_c_results'),
                    fetch('/api/phase_c_analysis'),
                ]);
                const results = resultsResp.ok ? await resultsResp.json() : null;
                const analysis = analysisResp.ok ? await analysisResp.json() : null;
                renderPhaseC(results, analysis);
            } catch (e) {
                document.getElementById('phase-c-loading').textContent = `Error loading Phase C data: ${e.message}`;
            }
        }

        function renderPhaseC(results, analysis) {
            const loading = document.getElementById('phase-c-loading');
            const recDiv = document.getElementById('phase-c-recommendation');
            const tableWrap = document.getElementById('phase-c-table-wrap');
            const tbody = document.getElementById('phase-c-tbody');
            const empty = document.getElementById('phase-c-empty');

            const hasData = results && results.data_status === 'ok' && results.summaries;
            if (!hasData) {
                loading.style.display = 'none';
                empty.style.display = 'block';
                return;
            }
            loading.style.display = 'none';

            // Recommendation badge
            const rec = analysis && analysis.recommendation ? analysis.recommendation : null;
            if (rec) {
                const badge = rec === 'no_default_change'
                    ? `<span style="background:#e8f5e9;color:#388e3c;padding:4px 10px;border-radius:4px;font-weight:600;">✅ ${rec.replace(/_/g,' ')}</span>`
                    : `<span style="background:#fff8e1;color:#f57f17;padding:4px 10px;border-radius:4px;font-weight:600;">⚠️ ${rec.replace(/_/g,' ')}</span>`;
                const base = analysis.baseline_mean_ndcg_at_10 != null
                    ? ` — Baseline NDCG@10: <strong>${analysis.baseline_mean_ndcg_at_10.toFixed(4)}</strong>` : '';
                recDiv.innerHTML = `Recommendation: ${badge}${base}`;
                recDiv.style.display = 'block';
            }

            // Candidate table
            const summaries = results.summaries || {};
            const promotable = (analysis && analysis.promotable_candidates) ? new Set(analysis.promotable_candidates) : new Set();
            const candidateSummaries = (analysis && analysis.candidate_summaries) ? analysis.candidate_summaries : {};
            tbody.replaceChildren();
            for (const [cname, cdata] of Object.entries(summaries)) {
                const ana = candidateSummaries[cname] || {};
                const ndcg = typeof cdata.mean_ndcg_at_10 === 'number' ? cdata.mean_ndcg_at_10.toFixed(4) : '-';
                const delta = typeof ana.mean_delta === 'number' ? (ana.mean_delta >= 0 ? '+' : '') + ana.mean_delta.toFixed(4) : '-';
                const winRate = typeof ana.win_rate === 'number' ? (ana.win_rate * 100).toFixed(1) + '%' : '-';
                const gatePct = typeof cdata.gate_apply_pct === 'number' ? cdata.gate_apply_pct.toFixed(0) + '%' : '-';
                const isPromo = promotable.has(cname);
                const status = cname === 'baseline'
                    ? '<span style="color:#1976d2;font-weight:600;">baseline</span>'
                    : isPromo
                        ? '<span style="color:#388e3c;font-weight:600;">✅ promotable</span>'
                        : '<span style="color:#888;">hold</span>';
                const deltaColor = ana.mean_delta != null ? (ana.mean_delta > 0 ? 'color:#388e3c' : ana.mean_delta < 0 ? 'color:#d32f2f' : '') : '';
                const row = document.createElement('tr');
                row.innerHTML = `<td><code>${cname}</code></td><td>${ndcg}</td><td style="${deltaColor}">${delta}</td><td>${winRate}</td><td>${gatePct}</td><td>${status}</td>`;
                tbody.appendChild(row);
            }
            tableWrap.style.display = 'block';
        }

        document.getElementById('event-filter').addEventListener('change', loadEvents);
        document.getElementById('limit-select').addEventListener('change', loadEvents);

        refreshData();
        setInterval(refreshData, 30000);
        setInterval(loadTTS, 5000);
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
            "adaptive_gate_actions": {},
            "adapter_route_breakdown": {},
            "adaptive_overlay": {
                "summary_count": 0,
                "ready_count": 0,
                "blocked_count": 0,
                "blockers": {},
                "latest_ready_for_broader_validation": None,
                "latest_next_action": None,
            },
            "runtime_diagnostics_count": 0,
            "latency_ms": {"mean": None, "p50": None, "p95": None},
            "time_range": {"earliest": None, "latest": None}
        }
    
    total_events = len(events)
    query_count = sum(1 for e in events if "query_snippet" in e)
    error_count = sum(1 for e in events if e.get("event_type") == "error" or e.get("error") is not None)
    
    # Count actions
    actions = [e.get("action") for e in events if "action" in e]
    action_breakdown = dict(Counter(actions))

    adaptive_actions = []
    for event in events:
        if event.get("event_type") == "adaptive_gate_evaluated":
            adaptive_actions.extend(event.get("actions", []))
        gate = event.get("adaptive_gate")
        if isinstance(gate, dict):
            adaptive_actions.extend(gate.get("actions", []))
    adaptive_gate_actions = dict(Counter(adaptive_actions))

    route_keys = []
    for event in events:
        if event.get("route_key") is not None:
            route_keys.append(event.get("route_key"))
        route = event.get("route")
        if isinstance(route, dict) and route.get("key") is not None:
            route_keys.append(route.get("key"))
    adapter_route_breakdown = dict(Counter(route_keys))

    overlay_summaries = []
    for event in events:
        overlay = event.get("adaptive_overlay_summary")
        if isinstance(overlay, dict):
            overlay_summaries.append(overlay)
        diagnostics = event.get("integrated_diagnostics")
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("adaptive_overlay_summary"), dict):
            overlay_summaries.append(diagnostics["adaptive_overlay_summary"])

    overlay_blockers = []
    ready_count = 0
    for overlay in overlay_summaries:
        if bool(overlay.get("ready_for_broader_validation", False)):
            ready_count += 1
        blockers = overlay.get("blockers", [])
        if isinstance(blockers, list):
            overlay_blockers.extend(str(blocker) for blocker in blockers)
    latest_overlay = overlay_summaries[-1] if overlay_summaries else {}
    adaptive_overlay = {
        "summary_count": len(overlay_summaries),
        "ready_count": ready_count,
        "blocked_count": len(overlay_summaries) - ready_count,
        "blockers": dict(Counter(overlay_blockers)),
        "latest_ready_for_broader_validation": (
            bool(latest_overlay.get("ready_for_broader_validation"))
            if overlay_summaries
            else None
        ),
        "latest_next_action": latest_overlay.get("next_action") if overlay_summaries else None,
    }

    runtime_events = [e for e in events if e.get("event_type") == "runtime_diagnostics" or isinstance(e.get("runtime"), dict)]
    latencies = []
    for event in runtime_events:
        runtime = event.get("runtime", {})
        if isinstance(runtime, dict) and runtime.get("latency_ms") is not None:
            latencies.append(float(runtime["latency_ms"]))
        elif event.get("latency_ms") is not None:
            latencies.append(float(event["latency_ms"]))

    def percentile(values, pct):
        if not values:
            return None
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * pct))
        return float(ordered[index])

    latency_ms = {
        "mean": float(sum(latencies) / len(latencies)) if latencies else None,
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
    }
    
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
        "adaptive_gate_actions": adaptive_gate_actions,
        "adapter_route_breakdown": adapter_route_breakdown,
        "adaptive_overlay": adaptive_overlay,
        "runtime_diagnostics_count": len(runtime_events),
        "latency_ms": latency_ms,
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


def _first_present(payload: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _relative_dashboard_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root.parent) if path.is_relative_to(root.parent) else path
    return relative.as_posix()


def _extract_campaign_record(path: Path, root: Path) -> Dict[str, Any]:
    payload = _load_json_object(path)
    promotion = payload.get("promotion_decision")
    if not isinstance(promotion, dict):
        promotion = payload.get("promotion")
    if not isinstance(promotion, dict):
        promotion = {}

    overlay = payload.get("adaptive_overlay_summary")
    if not isinstance(overlay, dict):
        overlay = payload.get("overlay_summary")
    if not isinstance(overlay, dict):
        overlay = {}

    artifact_card = payload.get("adaptive_overlay_artifact_card")
    if not isinstance(artifact_card, dict):
        artifact_card = payload.get("artifact_card")
    if not isinstance(artifact_card, dict):
        artifact_card = {}

    stat = path.stat()
    return {
        "path": _relative_dashboard_path(path, root),
        "report_name": path.name,
        "run_label": _first_present(payload, ["run_label", "label", "campaign_id"]) or path.parent.name,
        "record_type": payload.get("record_type"),
        "task": _first_present(payload, ["task", "dataset", "dataset_name"]),
        "decision": promotion.get("decision") or promotion.get("status") or payload.get("decision"),
        "default_change_allowed": bool(
            promotion.get("default_change_allowed", payload.get("default_change_allowed", False))
        ),
        "overlay_ready": overlay.get("ready_for_broader_validation"),
        "overlay_next_action": overlay.get("next_action"),
        "artifact_card_id": artifact_card.get("artifact_card_id") or artifact_card.get("id"),
        "data_source": payload.get("data_source", "unknown"),
        "modified_at": stat.st_mtime,
    }


def load_campaign_history(root: str = CAMPAIGN_HISTORY_ROOT, limit: int = 25) -> Dict[str, Any]:
    """Load compact campaign report history for dashboard display."""
    root_path = Path(root)
    if not root_path.exists():
        return {
            "root": root,
            "reports": [],
            "summary": {"total_reports": 0, "promotion_allowed": 0, "overlay_ready": 0},
        }

    report_paths = sorted(
        {
            *root_path.rglob("campaign_report.json"),
            *root_path.rglob("*campaign*report*.json"),
        },
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    reports = []
    for path in report_paths[: max(0, limit)]:
        try:
            reports.append(_extract_campaign_record(path, root_path))
        except (OSError, ValueError):
            continue

    return {
        "root": root,
        "reports": reports,
        "summary": {
            "total_reports": len(reports),
            "promotion_allowed": sum(1 for report in reports if report["default_change_allowed"]),
            "overlay_ready": sum(1 for report in reports if report["overlay_ready"] is True),
        },
    }


def _extract_validation_record(path: Path, root: Path) -> Dict[str, Any]:
    payload = _load_json_object(path)
    stat = path.stat()
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    return {
        "path": _relative_dashboard_path(path, root),
        "report_name": path.name,
        "record_type": payload.get("record_type"),
        "passed": bool(payload.get("passed", False)),
        "command_count": int(payload.get("command_count", len(results)) or 0),
        "failed_commands": [str(item) for item in payload.get("failed_commands", [])],
        "output_dir": payload.get("output_dir"),
        "data_source": payload.get("data_source", "unknown"),
        "modified_at": stat.st_mtime,
    }


def load_validation_history(root: str = VALIDATION_HISTORY_ROOT, limit: int = 10) -> Dict[str, Any]:
    """Load compact validation-bundle summaries for dashboard display."""
    root_path = Path(root)
    if not root_path.exists():
        return {
            "root": root,
            "reports": [],
            "summary": {"total_reports": 0, "passed": 0, "failed": 0, "latest_passed": None},
        }

    report_paths = sorted(root_path.rglob("validation_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    reports = []
    for path in report_paths[: max(0, limit)]:
        try:
            reports.append(_extract_validation_record(path, root_path))
        except (OSError, ValueError):
            continue
    latest = reports[0] if reports else {}
    return {
        "root": root,
        "reports": reports,
        "summary": {
            "total_reports": len(reports),
            "passed": sum(1 for report in reports if report["passed"]),
            "failed": sum(1 for report in reports if not report["passed"]),
            "latest_passed": latest.get("passed") if reports else None,
        },
    }


def _extract_preflight_record(path: Path, root: Path) -> Dict[str, Any]:
    payload = _load_json_object(path)
    stat = path.stat()
    blockers = payload.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    return {
        "path": _relative_dashboard_path(path, root),
        "report_name": path.name,
        "record_type": payload.get("record_type"),
        "review_allowed": bool(payload.get("review_allowed", False)),
        "default_change_allowed": bool(payload.get("default_change_allowed", False)),
        "blockers": [str(blocker) for blocker in blockers],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "modified_at": stat.st_mtime,
    }


def load_preflight_history(root: str = PREFLIGHT_HISTORY_ROOT, limit: int = 10) -> Dict[str, Any]:
    """Load compact default-promotion preflight reports for dashboard display."""
    root_path = Path(root)
    if not root_path.exists():
        return {
            "root": root,
            "reports": [],
            "summary": {"total_reports": 0, "review_allowed": 0, "blocked": 0, "latest_review_allowed": None},
        }

    report_paths_with_mtime = []
    for path in root_path.rglob("*preflight*.json"):
        try:
            report_paths_with_mtime.append((path, path.stat().st_mtime))
        except OSError:
            continue
    report_paths_with_mtime.sort(key=lambda item: item[1], reverse=True)
    report_paths = [path for path, _mtime in report_paths_with_mtime]
    reports = []
    for path in report_paths[: max(0, limit)]:
        try:
            reports.append(_extract_preflight_record(path, root_path))
        except (OSError, ValueError):
            continue
    latest = reports[0] if reports else {}
    return {
        "root": root,
        "reports": reports,
        "summary": {
            "total_reports": len(reports),
            "review_allowed": sum(1 for report in reports if report["review_allowed"]),
            "blocked": sum(1 for report in reports if not report["review_allowed"]),
            "latest_review_allowed": latest.get("review_allowed") if reports else None,
            "latest_blockers": latest.get("blockers", []) if reports else [],
        },
    }


def load_evidence_index(path: str = EVIDENCE_INDEX_PATH) -> Dict[str, Any]:
    """Load the latest compact cross-artifact evidence index for dashboard display."""
    index_path = Path(path)
    if not index_path.exists():
        return {
            "path": path,
            "present": False,
            "summary": {
                "artifact_counts": {},
                "latest_review_allowed": None,
                "latest_preflight_blockers": [],
                "latest_chain_passed": None,
            },
            "artifacts": {},
        }
    try:
        payload = _load_json_object(index_path)
    except (OSError, ValueError):
        payload = {}
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    stat = index_path.stat()
    return {
        "path": str(index_path),
        "present": True,
        "record_type": payload.get("record_type"),
        "root": payload.get("root"),
        "modified_at": stat.st_mtime,
        "summary": {
            "artifact_counts": summary.get("artifact_counts", {}),
            "latest_review_allowed": summary.get("latest_review_allowed"),
            "latest_preflight_blockers": summary.get("latest_preflight_blockers", []),
            "latest_chain_passed": summary.get("latest_chain_passed"),
        },
        "artifacts": artifacts,
    }


def _extract_evidence_chain_record(path: Path, root: Path) -> Dict[str, Any]:
    payload = _load_json_object(path)
    stat = path.stat()
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    blockers = payload.get("preflight_blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    failures = payload.get("command_failures", [])
    if not isinstance(failures, list):
        failures = []
    return {
        "path": _relative_dashboard_path(path, root),
        "report_name": path.name,
        "record_type": payload.get("record_type"),
        "chain_passed": bool(payload.get("chain_passed", False)),
        "review_allowed": bool(payload.get("review_allowed", False)),
        "default_change_allowed": bool(payload.get("default_change_allowed", False)),
        "command_failures": [str(item) for item in failures],
        "preflight_blockers": [str(item) for item in blockers],
        "artifact_count": len(artifacts),
        "modified_at": stat.st_mtime,
    }


def load_evidence_chain_history(root: str = EVIDENCE_CHAIN_HISTORY_ROOT, limit: int = 10) -> Dict[str, Any]:
    """Load compact default-promotion evidence-chain summaries for dashboard display."""
    root_path = Path(root)
    if not root_path.exists():
        return {
            "root": root,
            "reports": [],
            "summary": {"total_reports": 0, "passed": 0, "failed": 0, "latest_chain_passed": None},
        }
    report_paths = sorted(root_path.rglob("evidence_chain_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    reports = []
    for path in report_paths[: max(0, limit)]:
        try:
            reports.append(_extract_evidence_chain_record(path, root_path))
        except (OSError, ValueError):
            continue
    latest = reports[0] if reports else {}
    return {
        "root": root,
        "reports": reports,
        "summary": {
            "total_reports": len(report_paths),
            "loaded_reports": len(reports),
            "passed": sum(1 for report in reports if report["chain_passed"]),
            "failed": sum(1 for report in reports if not report["chain_passed"]),
            "latest_chain_passed": latest.get("chain_passed") if reports else None,
            "latest_review_allowed": latest.get("review_allowed") if reports else None,
            "latest_preflight_blockers": latest.get("preflight_blockers", []) if reports else [],
        },
    }


def load_evidence_cleanup_plan(
    root: str = EVIDENCE_CLEANUP_ROOT,
    keep_latest: int = 1,
    candidate_limit: int = 25,
    evidence_index: Optional[Union[str, Path]] = None,
    freshness_audit: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a read-only dry-run cleanup plan for dashboard display."""

    plan_kwargs: Dict[str, Any] = {"root": root, "keep_latest": keep_latest}
    if evidence_index:
        plan_kwargs["evidence_index"] = evidence_index
    if freshness_audit:
        plan_kwargs["freshness_audit"] = freshness_audit
    plan = plan_evidence_artifact_cleanup(**plan_kwargs)
    candidates = plan.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    retained = plan.get("retained", [])
    if not isinstance(retained, list):
        retained = []
    summary = plan.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    return {
        "record_type": plan.get("record_type"),
        "dry_run": True,
        "root": plan.get("root", root),
        "keep_latest": plan.get("keep_latest", keep_latest),
        "source_artifacts": plan.get("source_artifacts", {}),
        "source_status": plan.get("source_status", {}),
        "summary": {
            "candidate_count": summary.get("candidate_count", len(candidates)),
            "retained_count": summary.get("retained_count", len(retained)),
            "candidate_bytes": summary.get("candidate_bytes", 0),
            "cleanup_review_allowed": summary.get("cleanup_review_allowed"),
            "missing_source_artifacts": summary.get("missing_source_artifacts", []),
            "candidate_types": sorted({str(item.get("artifact_type", "unknown")) for item in candidates if isinstance(item, dict)}),
        },
        "candidates": candidates[: max(0, candidate_limit)],
    }


def load_phase_c_results(path: str = PHASE_C_RESULTS_PATH) -> Dict[str, Any]:
    """Load Phase C four-candidate evaluation results for dashboard display.

    Returns a dashboard-safe payload: if the file is missing, returns a
    ``data_status="not_generated"`` sentinel so the frontend can show a helpful
    prompt rather than an error.
    """
    p = Path(path)
    if not p.exists():
        return {
            "data_status": "not_generated",
            "reason": f"{path} not found — run run_phase_c_eval.py to populate",
            "candidates": [],
            "datasets": [],
            "summaries": {},
        }
    try:
        payload = _load_json_object(p)
    except (OSError, ValueError) as exc:
        return {
            "data_status": "error",
            "reason": f"Failed to load {path}: {exc}",
            "candidates": [],
            "datasets": [],
            "summaries": {},
        }
    summaries = payload.get("summaries")
    if not isinstance(summaries, dict):
        summaries = {}
    return {
        "data_status": "ok",
        "record_type": payload.get("record_type"),
        "run_at": payload.get("run_at"),
        "schema_version": payload.get("schema_version"),
        "candidates": payload.get("candidates") if isinstance(payload.get("candidates"), list) else [],
        "datasets": payload.get("datasets") if isinstance(payload.get("datasets"), list) else [],
        "summaries": summaries,
        "gate_summary": payload.get("gate_summary") if isinstance(payload.get("gate_summary"), dict) else {},
    }


def load_phase_c_analysis(path: str = PHASE_C_ANALYSIS_PATH) -> Dict[str, Any]:
    """Load Phase C analysis report for dashboard display.

    Returns a dashboard-safe payload with a ``data_status`` sentinel when the
    file is missing or malformed.
    """
    p = Path(path)
    if not p.exists():
        return {
            "data_status": "not_generated",
            "reason": f"{path} not found — run analyze_phase_c_results.py to populate",
            "recommendation": None,
            "promotable_candidates": [],
            "candidate_summaries": {},
        }
    try:
        payload = _load_json_object(p)
    except (OSError, ValueError) as exc:
        return {
            "data_status": "error",
            "reason": f"Failed to load {path}: {exc}",
            "recommendation": None,
            "promotable_candidates": [],
            "candidate_summaries": {},
        }
    summaries = payload.get("candidate_summaries")
    if not isinstance(summaries, dict):
        summaries = {}
    promotable = payload.get("promotable_candidates")
    if not isinstance(promotable, list):
        promotable = []
    return {
        "data_status": "ok",
        "analysis_timestamp": payload.get("analysis_timestamp"),
        "schema_version": payload.get("schema_version"),
        "recommendation": payload.get("recommendation"),
        "promotable_candidates": promotable,
        "baseline_mean_ndcg_at_10": payload.get("baseline_mean_ndcg_at_10"),
        "candidate_summaries": summaries,
    }


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
        elif path == "/api/beir_results":
            self.handle_api_beir_results()
        elif path == "/api/campaign_history":
            self.handle_api_campaign_history(query_params)
        elif path == "/api/validation_history":
            self.handle_api_validation_history(query_params)
        elif path == "/api/preflight_history":
            self.handle_api_preflight_history(query_params)
        elif path == "/api/evidence_index":
            self.handle_api_evidence_index()
        elif path == "/api/evidence_chain_history":
            self.handle_api_evidence_chain_history(query_params)
        elif path == "/api/evidence_cleanup_plan":
            self.handle_api_evidence_cleanup_plan(query_params)
        elif path == "/api/phase_c_results":
            self.handle_api_phase_c_results()
        elif path == "/api/phase_c_analysis":
            self.handle_api_phase_c_analysis()
        elif path == "/api/model_scope/events":
            self.handle_api_model_scope_events(query_params)
        elif path == "/api/model_scope/features":
            self.handle_api_model_scope_features(query_params)
        elif path == "/api/model_scope/interventions":
            self.handle_api_model_scope_interventions(query_params)
        elif path == "/api/tts/status":
            self.handle_api_tts_status()
        elif path == "/api/tts/last_result":
            self.handle_api_tts_last_result()
        elif path == "/api/disk_llm_estimate":
            self.handle_api_disk_llm_estimate(query_params)
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
                self.send_json_response({
                    "data_status": "not_generated",
                    "reason": "large_sweep_results.json not found — run run_large_sweep.py to populate",
                    "results": [],
                })
                return
            
            with open(sweep_file, 'r') as f:
                results = json.load(f)
            if isinstance(results, dict):
                results.setdefault("data_status", "ok")
                self.send_json_response(results)
            else:
                self.send_json_response({"data_status": "ok", "results": results})
        except Exception as e:
            self.send_error_response(500, f"Error reading sweep results: {str(e)}")

    def handle_api_test_results(self):
        """Handle /api/test_results endpoint."""
        test_file = ".report.json"
        try:
            if not os.path.exists(test_file):
                self.send_json_response({
                    "data_status": "not_generated",
                    "reason": ".report.json not found — generate with: "
                              "python -m unittest discover -v 2>&1 | python generate_report_json.py",
                    "summary": None,
                    "tests": [],
                })
                return
            with open(test_file, 'r') as f:
                report = json.load(f)
            if isinstance(report, dict) and "data_status" not in report:
                report["data_status"] = "ok"
            self.send_json_response(report)
        except Exception as e:
            self.send_error_response(500, f"Error reading test results: {str(e)}")

    def handle_api_beir_results(self):
        """Handle /api/beir_results endpoint.

        Returns BEIR multi-dataset benchmark results from
        benchmark_beir_results.json if available.
        """
        beir_file = "benchmark_beir_results.json"
        try:
            if not os.path.exists(beir_file):
                self.send_json_response({
                    "data_status": "not_generated",
                    "reason": "benchmark_beir_results.json not found — run benchmark_beir.py to populate",
                    "results": [],
                    "aggregated_by_config": {},
                    "aggregated_by_dataset": {},
                    "heatmap": {"configs": [], "datasets": [], "ndcg_matrix": []},
                    "summary": {"num_datasets": 0, "num_configs": 0, "total_evaluations": 0},
                })
                return
            with open(beir_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict) and "data_status" not in data:
                data["data_status"] = "ok"
            self.send_json_response(data)
        except Exception as e:
            self.send_error_response(500, f"Error reading BEIR results: {str(e)}")

    def handle_api_campaign_history(self, query_params: Dict[str, List[str]]):
        """Handle /api/campaign_history endpoint."""
        limit = 25
        if "limit" in query_params:
            try:
                limit = int(query_params["limit"][0])
            except (ValueError, IndexError):
                limit = 25
        try:
            self.send_json_response(load_campaign_history(CAMPAIGN_HISTORY_ROOT, limit=limit))
        except Exception as e:
            self.send_error_response(500, f"Error reading campaign history: {str(e)}")

    def handle_api_validation_history(self, query_params: Dict[str, List[str]]):
        """Handle /api/validation_history endpoint."""
        limit = 10
        if "limit" in query_params:
            try:
                limit = int(query_params["limit"][0])
            except (ValueError, IndexError):
                limit = 10
        try:
            self.send_json_response(load_validation_history(VALIDATION_HISTORY_ROOT, limit=limit))
        except Exception as e:
            self.send_error_response(500, f"Error reading validation history: {str(e)}")

    def handle_api_preflight_history(self, query_params: Dict[str, List[str]]):
        """Handle /api/preflight_history endpoint."""
        limit = 10
        if "limit" in query_params:
            try:
                limit = int(query_params["limit"][0])
            except (ValueError, IndexError):
                limit = 10
        try:
            self.send_json_response(load_preflight_history(PREFLIGHT_HISTORY_ROOT, limit=limit))
        except Exception as e:
            self.send_error_response(500, f"Error reading preflight history: {str(e)}")

    def handle_api_evidence_index(self):
        """Handle /api/evidence_index endpoint."""
        try:
            self.send_json_response(load_evidence_index(EVIDENCE_INDEX_PATH))
        except Exception as e:
            self.send_error_response(500, f"Error reading evidence index: {str(e)}")

    def handle_api_evidence_chain_history(self, query_params: Dict[str, List[str]]):
        """Handle /api/evidence_chain_history endpoint."""
        limit = 10
        if "limit" in query_params:
            try:
                limit = int(query_params["limit"][0])
            except (ValueError, IndexError):
                limit = 10
        try:
            self.send_json_response(load_evidence_chain_history(EVIDENCE_CHAIN_HISTORY_ROOT, limit=limit))
        except Exception as e:
            self.send_error_response(500, f"Error reading evidence-chain history: {str(e)}")

    def handle_api_evidence_cleanup_plan(self, query_params: Dict[str, List[str]]):
        """Handle /api/evidence_cleanup_plan endpoint."""
        try:
            keep_latest = 1
            limit = 25
            if "keep_latest" in query_params:
                keep_latest = max(0, int(query_params["keep_latest"][0]))
            if "limit" in query_params:
                limit = max(0, int(query_params["limit"][0]))
            self.send_json_response(load_evidence_cleanup_plan(EVIDENCE_CLEANUP_ROOT, keep_latest=keep_latest, candidate_limit=limit))
        except Exception as e:
            self.send_error_response(500, f"Error reading evidence cleanup plan: {str(e)}")

    def handle_api_phase_c_results(self):
        """Handle /api/phase_c_results endpoint."""
        try:
            self.send_json_response(load_phase_c_results(PHASE_C_RESULTS_PATH))
        except Exception as e:
            self.send_error_response(500, f"Error reading Phase C results: {str(e)}")

    def handle_api_phase_c_analysis(self):
        """Handle /api/phase_c_analysis endpoint."""
        try:
            self.send_json_response(load_phase_c_analysis(PHASE_C_ANALYSIS_PATH))
        except Exception as e:
            self.send_error_response(500, f"Error reading Phase C analysis: {str(e)}")

    def handle_api_model_scope_events(self, query_params):
        """Handle /api/model_scope/events — lists recent activation event files."""
        from model_scope_artifacts import ArtifactStore, load_model_scope_artifact, summarize_model_scope_artifact
        try:
            limit = int(query_params.get("limit", ["20"])[0])
            store = ArtifactStore(base_dir=MODEL_SCOPE_ARTIFACT_ROOT)
            paths = store.list_artifacts(pattern="feature_event_*.json")[-limit:]
            items = []
            for p in reversed(paths):
                try:
                    artifact = load_model_scope_artifact(p)
                    items.append({"path": str(p), "summary": summarize_model_scope_artifact(artifact)})
                except Exception as e:
                    items.append({"path": str(p), "error": str(e)})
            self.send_json_response({
                "status": "ok" if items else "not_generated",
                "count": len(items),
                "reason": None if items else "no_artifacts_found",
                "items": items,
            })
        except Exception as e:
            self.send_error_response(500, f"Error reading model-scope events: {e}")

    def handle_api_model_scope_features(self, query_params):
        """Handle /api/model_scope/features — lists recent sparse feature events."""
        from model_scope_artifacts import ArtifactStore, load_model_scope_artifact
        try:
            limit = int(query_params.get("limit", ["20"])[0])
            store = ArtifactStore(base_dir=MODEL_SCOPE_ARTIFACT_ROOT)
            paths = store.list_artifacts(pattern="feature_event_*.json")[-limit:]
            items = []
            for p in reversed(paths):
                try:
                    raw = load_model_scope_artifact(p)
                    items.append(raw)
                except Exception as e:
                    items.append({"path": str(p), "error": str(e)})
            self.send_json_response({
                "status": "ok" if items else "not_generated",
                "count": len(items),
                "reason": None if items else "no_feature_events_found",
                "items": items,
            })
        except Exception as e:
            self.send_error_response(500, f"Error reading model-scope features: {e}")

    def handle_api_model_scope_interventions(self, query_params):
        """Handle /api/model_scope/interventions — lists recent intervention records."""
        from model_scope_artifacts import ArtifactStore, load_model_scope_artifact
        try:
            limit = int(query_params.get("limit", ["20"])[0])
            store = ArtifactStore(base_dir=MODEL_SCOPE_ARTIFACT_ROOT)
            paths = store.list_artifacts(pattern="intervention_*.json")[-limit:]
            items = []
            for p in reversed(paths):
                try:
                    items.append(load_model_scope_artifact(p))
                except Exception as e:
                    items.append({"path": str(p), "error": str(e)})
            self.send_json_response({
                "status": "ok" if items else "not_generated",
                "count": len(items),
                "reason": None if items else "no_interventions_found",
                "items": items,
            })
        except Exception as e:
            self.send_error_response(500, f"Error reading model-scope interventions: {e}")

    def handle_api_tts_status(self):
        """Handle /api/tts/status — returns TTS enable state, config, and last result summary."""
        try:
            state = _TTS_DASHBOARD_STATE
            last = state.get("last_result")
            last_summary: Optional[Dict[str, Any]] = None
            if last is not None:
                last_summary = {
                    "stages_applied": last.get("stages_applied", []),
                    "total_delta_norm": last.get("total_delta_norm", 0.0),
                }
            self.send_json_response({
                "enabled": state.get("enabled", False),
                "config": state.get("config", {}),
                "last_result_summary": last_summary,
            })
        except Exception as e:
            self.send_error_response(500, f"Error reading TTS status: {e}")

    def handle_api_tts_last_result(self):
        """Handle /api/tts/last_result — returns full TTSResult summary as JSON."""
        try:
            state = _TTS_DASHBOARD_STATE
            last = state.get("last_result")
            if last is None:
                self.send_json_response({
                    "last_result_available": False,
                    "translation_offset_norm": None,
                    "transport_weight_used": None,
                    "steering_delta_norm": None,
                    "norm_before": None,
                    "norm_after": None,
                    "total_delta_norm": None,
                    "stages_applied": None,
                })
            else:
                self.send_json_response(dict(last))
        except Exception as e:
            self.send_error_response(500, f"Error reading TTS last result: {e}")

    def handle_api_disk_llm_estimate(self, query_params: Dict[str, List[str]]):
        """Handle /api/disk_llm_estimate — feasibility estimate for a disk-resident LLM.

        Computed live by computational_storage_poc.disk_llm_estimator.estimate_disk_llm.
        Query parameters (all optional, with safe defaults):
          - hardware: one of DEFAULT_HARDWARE_PROFILES keys (default workstation_gen5)
          - params_billion: model size in billions of params (default 70)
          - bits: bits per weight after quantization (default 4)
        """
        try:
            hardware_name = query_params.get("hardware", ["workstation_gen5"])[0]
            if hardware_name not in DEFAULT_HARDWARE_PROFILES:
                self.send_error_response(
                    400,
                    f"unknown hardware profile '{hardware_name}'; "
                    f"available: {sorted(DEFAULT_HARDWARE_PROFILES)}",
                )
                return
            try:
                params_billion = float(query_params.get("params_billion", ["70"])[0])
                bits = float(query_params.get("bits", ["4"])[0])
            except (TypeError, ValueError):
                self.send_error_response(400, "params_billion and bits must be numeric")
                return

            hardware = DEFAULT_HARDWARE_PROFILES[hardware_name]
            estimate = estimate_disk_llm(
                params_billion=params_billion,
                bits_per_weight=bits,
                ssd_bandwidth_gbps=hardware.ssd_bandwidth_gbps,
                dram_gb=hardware.dram_gb,
            )
            self.send_json_response({
                "hardware": {
                    "name": hardware.name,
                    "ssd_bandwidth_gbps": hardware.ssd_bandwidth_gbps,
                    "dram_gb": hardware.dram_gb,
                    "cpu_cores": hardware.cpu_cores,
                },
                "request": {
                    "params_billion": params_billion,
                    "bits_per_weight": bits,
                },
                "estimate": {
                    "model_size_gb": estimate.model_size_gb,
                    "resident_set_gb": estimate.resident_set_gb,
                    "required_ram_gb": estimate.required_ram_gb,
                    "streamed_bytes_per_token_dense_gb": estimate.streamed_bytes_per_token_dense_gb,
                    "streamed_bytes_per_token_flash_gb": estimate.streamed_bytes_per_token_flash_gb,
                    "dense_tokens_per_second_upper": estimate.dense_tokens_per_second_upper,
                    "flash_tokens_per_second_upper": estimate.flash_tokens_per_second_upper,
                    "fits_resident_ram": estimate.fits_resident_ram,
                    "fits_paper_ratio": estimate.fits_paper_ratio,
                },
                "source": "computational_storage_poc.disk_llm_estimator.estimate_disk_llm",
            })
        except ValueError as exc:
            self.send_error_response(400, f"invalid estimator input: {exc}")
        except Exception as exc:
            self.send_error_response(500, f"Error computing disk-LLM estimate: {exc}")

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
