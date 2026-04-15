from __future__ import annotations


class HudHtmlPresenter:
    def render_shell(self, *, title: str, poll_interval_ms: int) -> str:
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #141b24;
      --panel-alt: #192330;
      --line: #2b3a4d;
      --text: #dbe6f3;
      --muted: #8ca0b8;
      --good: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
      --accent: #58a6ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Consolas, "SFMono-Regular", monospace; background: radial-gradient(circle at top, #132031 0%, var(--bg) 50%); color: var(--text); }}
    header {{ padding: 20px 24px; border-bottom: 1px solid var(--line); display:flex; justify-content:space-between; align-items:center; position:sticky; top:0; backdrop-filter: blur(8px); background: rgba(11,15,20,.92); z-index:10; }}
    h1,h2,h3 {{ margin: 0 0 12px; font-weight: 700; letter-spacing: .04em; }}
    main {{ padding: 20px; display:grid; gap:20px; }}
    .grid {{ display:grid; gap:16px; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); }}
    .panel {{ background: linear-gradient(180deg, rgba(25,35,48,.95), rgba(20,27,36,.98)); border: 1px solid var(--line); border-radius: 14px; padding: 16px; box-shadow: 0 14px 40px rgba(0,0,0,.28); }}
    .badge {{ display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid var(--line); font-size:12px; }}
    .status-ready {{ color: var(--good); border-color: rgba(63,185,80,.4); }}
    .status-degraded, .status-warning, .status-paused {{ color: var(--warn); border-color: rgba(210,153,34,.4); }}
    .status-failed, .status-error, .status-cancelled, .status-stopped {{ color: var(--bad); border-color: rgba(248,81,73,.4); }}
    .status-running, .status-pending, .status-info {{ color: var(--accent); border-color: rgba(88,166,255,.4); }}
    .toolbar {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
    button {{ background: var(--panel-alt); color: var(--text); border:1px solid var(--line); border-radius:10px; padding:8px 12px; cursor:pointer; }}
    button:hover {{ border-color: var(--accent); }}
    .list {{ display:grid; gap:10px; }}
    .row {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; border-top:1px solid rgba(43,58,77,.55); padding-top:10px; }}
    .row:first-child {{ border-top:none; padding-top:0; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .actions {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
    input, select {{ background:#0f1620; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px; min-width:180px; }}
    .timeline-entry {{ border-left: 2px solid var(--line); padding-left: 12px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>JARVIS Control Center</h1>
      <div class="muted">Operational HUD for runtime, autonomy and ops</div>
    </div>
    <div class="toolbar">
      <button onclick="refreshAll()">Refresh</button>
      <span class="muted" id="last-refresh">never</span>
    </div>
  </header>
  <main>
    <section class="panel"><h2>Dashboard</h2><div id="dashboard"></div></section>
    <section class="panel"><h2>Mission Control</h2><div id="missions"></div></section>
    <section class="panel"><h2>Ops / Diagnostics</h2><div id="ops"></div></section>
    <section class="panel"><h2>Timeline / Receipts</h2><div id="timeline"></div></section>
    <section class="panel"><h2>Runtime Panels</h2><div id="runtime-panels" class="grid"></div></section>
  </main>
  <script>
    const pollInterval = {poll_interval_ms};
    async function fetchJson(url, options) {{
      const response = await fetch(url, Object.assign({{headers: {{'Content-Type':'application/json'}}}}, options || {{}}));
      if (!response.ok) {{
        const payload = await response.text();
        throw new Error(payload || response.statusText);
      }}
      return response.json();
    }}
    function badge(status) {{
      const normalized = String(status || 'unknown').toLowerCase();
      return `<span class="badge status-${{normalized}}">${{normalized}}</span>`;
    }}
    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
    }}
    async function refreshDashboard() {{
      const data = await fetchJson('/hud/dashboard');
      const services = data.services.map(s => `<div class="row"><div><strong>${{escapeHtml(s.name)}}</strong><div class="muted">${{escapeHtml(JSON.stringify(s.details))}}</div></div><div>${{badge(s.status)}}</div></div>`).join('');
      const alerts = data.alerts.map(a => `<div class="row"><div><strong>${{escapeHtml(a.title)}}</strong><div class="muted">${{escapeHtml(a.message)}}</div></div><div>${{badge(a.level)}}</div></div>`).join('') || '<div class="muted">No alerts</div>';
      document.getElementById('dashboard').innerHTML = `
        <div class="grid">
          <div class="panel"><h3>Mode</h3><div>${{escapeHtml(data.mode.active_mode)}}</div><div class="muted">${{escapeHtml(data.environment)}}</div></div>
          <div class="panel"><h3>Health</h3><div>${{badge(data.health_summary.aggregate_status)}}</div><div class="muted">services: ${{data.health_summary.service_count}}</div><div class="muted">recent failures: ${{data.health_summary.recent_failures}}</div></div>
          <div class="panel"><h3>Alerts</h3><div class="list">${{alerts}}</div></div>
          <div class="panel"><h3>Services</h3><div class="list">${{services}}</div></div>
        </div>`;
      renderRuntimePanels(data.runtimes || []);
    }}
    async function refreshMissions() {{
      const data = await fetchJson('/hud/missions');
      const html = (data.missions || []).map(m => `
        <div class="row">
          <div>
            <strong>${{escapeHtml(m.goal)}}</strong>
            <div class="muted">${{escapeHtml(m.mission_id)}} | step: ${{escapeHtml(m.active_step_id || 'n/a')}}</div>
            <div class="muted">approval: ${{escapeHtml(m.pending_approval_step_id || 'none')}}</div>
            <div class="actions">
              <button onclick="hudAction('/hud/actions/approve', {{mission_id: m.mission_id, step_id: m.pending_approval_step_id}})">Approve</button>
              <button onclick="hudAction('/hud/actions/reject', {{mission_id: m.mission_id, step_id: m.pending_approval_step_id}})">Reject</button>
              <button onclick="hudAction('/hud/actions/pause', {{mission_id: m.mission_id}})">Pause</button>
              <button onclick="hudAction('/hud/actions/resume', {{mission_id: m.mission_id}})">Resume</button>
              <button onclick="hudAction('/hud/actions/stop', {{mission_id: m.mission_id}})">Stop</button>
            </div>
          </div>
          <div>${{badge(m.status)}}</div>
        </div>`).join('');
      document.getElementById('missions').innerHTML = html || '<div class="muted">No missions</div>';
    }}
    async function refreshOps() {{
      const data = await fetchJson('/hud/health');
      const diagnostics = (data.diagnostics || []).map(d => `<div class="row"><div><strong>${{escapeHtml(d.service_name)}}</strong><div class="muted">${{escapeHtml((d.warnings || []).join(', '))}}</div></div><div>${{badge(d.status)}}</div></div>`).join('');
      const actions = `
        <div class="actions">
          <button onclick="hudAction('/hud/actions/retention-sweep', {{}})">Retention Sweep</button>
          <button onclick="recoverPrompt()">Recover Service</button>
          <button onclick="resetBreakerPrompt()">Reset Breaker</button>
        </div>`;
      document.getElementById('ops').innerHTML = `
        <div>${{badge(data.status.aggregate_status)}}</div>
        <div class="muted">degraded dependencies: ${{escapeHtml((data.status.degraded_dependencies || []).join(', ') || 'none')}}</div>
        ${{actions}}
        <div class="list" style="margin-top:12px">${{diagnostics}}</div>`;
    }}
    async function refreshTimeline() {{
      const data = await fetchJson('/hud/timeline');
      const html = (data.entries || []).map(entry => `
        <div class="timeline-entry">
          <div class="row">
            <div><strong>${{escapeHtml(entry.title)}}</strong><div class="muted">${{escapeHtml(entry.service_name)}} · ${{new Date(entry.timestamp).toLocaleString()}}</div></div>
            <div>${{badge(entry.status)}}</div>
          </div>
        </div>`).join('');
      document.getElementById('timeline').innerHTML = html || '<div class="muted">No timeline entries</div>';
    }}
    function renderRuntimePanels(panels) {{
      const html = panels.map(panel => `
        <div class="panel">
          <div class="row">
            <div><h3>${{escapeHtml(panel.name)}}</h3><div class="muted">${{escapeHtml(JSON.stringify(panel.summary))}}</div></div>
            <div>${{badge(panel.status)}}</div>
          </div>
          <div class="actions">
            ${{
              (panel.quick_actions || []).map(action => `<button onclick="quickAction('${{action.action}}')">${{escapeHtml(action.action)}}</button>`).join('')
            }}
          </div>
        </div>`).join('');
      document.getElementById('runtime-panels').innerHTML = html;
    }}
    async function hudAction(url, payload) {{
      try {{
        await fetchJson(url, {{method:'POST', body: JSON.stringify(payload || {{}})}});
        await refreshAll();
      }} catch (error) {{
        alert(error.message);
      }}
    }}
    function quickAction(name) {{
      if (name === 'retention_sweep') return hudAction('/hud/actions/retention-sweep', {{}});
      if (name === 'indexing_run') return hudAction('/hud/actions/indexing', {{}});
      if (name === 'system_status') return hudAction('/hud/actions/system-status', {{}});
      if (name === 'unity_bridge_status') return hudAction('/hud/actions/unity-bridge-status', {{}});
      if (name === 'research_run') {{
        const query = prompt('Research query');
        if (query) hudAction('/hud/actions/research', {{query}});
      }}
      if (name === 'writing_run') {{
        const promptValue = prompt('Writing prompt');
        if (promptValue) hudAction('/hud/actions/writing', {{prompt: promptValue}});
      }}
    }}
    function recoverPrompt() {{
      const service_name = prompt('Service name to recover');
      if (service_name) hudAction('/hud/actions/recover', {{service_name}});
    }}
    function resetBreakerPrompt() {{
      const service_name = prompt('Service name');
      if (!service_name) return;
      const dependency_name = prompt('Dependency name (optional)') || null;
      hudAction('/hud/actions/reset-breaker', {{service_name, dependency_name}});
    }}
    async function refreshAll() {{
      await Promise.all([refreshDashboard(), refreshMissions(), refreshOps(), refreshTimeline()]);
      document.getElementById('last-refresh').textContent = `refreshed ${{new Date().toLocaleTimeString()}}`;
    }}
    refreshAll();
    window.setInterval(refreshAll, pollInterval);
  </script>
</body>
</html>"""
