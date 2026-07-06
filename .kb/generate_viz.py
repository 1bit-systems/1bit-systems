#!/usr/bin/env python3
"""Generate a self-contained HTML graph visualization of an OKF bundle."""

import json, re, yaml, sys
from pathlib import Path

INDEX_NAME = "index.md"
LINK_RE = re.compile(r"\]\(([^)\s]+\.md)(?:#[A-Za-z0-9_\-]*)?\)")
TYPE_PALETTE = {
    "Engine": "#8b5cf6",
    "Structure": "#3b82f6",
    "Reference": "#10b981",
}
DEFAULT_COLOR = "#94a3b8"

bundle_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else bundle_root / "viz.html"

concepts = []
for md in sorted(bundle_root.rglob("*.md")):
    if md.name == INDEX_NAME or md.name == "log.md":
        continue
    rel = md.relative_to(bundle_root).with_suffix("")
    cid = "/".join(rel.parts)
    text = md.read_text(encoding="utf-8")
    fm = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
            except: pass
            body = parts[2].strip()
    tags = fm.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    typ = str(fm.get("type", "Unknown"))
    concepts.append({
        "id": cid,
        "type": typ,
        "title": str(fm.get("title", cid)),
        "description": str(fm.get("description", "")),
        "resource": str(fm.get("resource", "")),
        "tags": [str(t) for t in tags],
        "body": body,
    })

ids = {c["id"] for c in concepts}
bodies = {c["id"]: c["body"] for c in concepts}
types = sorted({c["type"] for c in concepts})

nodes = []
for c in concepts:
    color = TYPE_PALETTE.get(c["type"], DEFAULT_COLOR)
    nodes.append({
        "data": {
            "id": c["id"],
            "label": c["title"],
            "type": c["type"],
            "description": c["description"],
            "resource": c["resource"],
            "tags": c["tags"],
            "color": color,
            "size": 30 + min(60, len(c["body"]) // 200),
        }
    })

edges = []
seen = set()
for c in concepts:
    doc_dir = bundle_root / "/".join(c["id"].split("/")[:-1]) if "/" in c["id"] else bundle_root
    for m in LINK_RE.finditer(c["body"]):
        t = m.group(1)
        if "://" in t:
            continue
        try:
            if t.startswith("/"):
                # Bundle-relative path: resolve from bundle root
                resolved = (bundle_root / t.lstrip("/")).resolve().relative_to(bundle_root.resolve())
            else:
                resolved = (doc_dir / t).resolve().relative_to(bundle_root.resolve())
        except:
            continue
        target = resolved.as_posix().replace(".md", "")
        if target == c["id"] or target not in ids:
            continue
        key = (c["id"], target)
        if key in seen: continue
        seen.add(key)
        edges.append({
            "data": {"id": f"{c['id']}__{target}", "source": c["id"], "target": target}
        })

template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>1bit.systems Knowledge Base</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;display:flex;flex-direction:column;height:100vh;background:#0f172a;color:#e2e8f0}
header{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:#1e293b;border-bottom:1px solid #334155;flex-wrap:wrap;gap:8px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
.controls input,.controls select{padding:6px 10px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;font-size:13px}
.controls button{padding:6px 14px;border-radius:6px;border:none;background:#3b82f6;color:white;cursor:pointer;font-size:13px}
main{display:flex;flex:1;overflow:hidden}
#graph{flex:2;min-width:0}
#detail{flex:1;min-width:320px;max-width:480px;overflow-y:auto;padding:20px;background:#1e293b;border-left:1px solid #334155}
#detail-empty{text-align:center;margin-top:40px;color:#64748b}
.detail-header{margin-bottom:16px}
.type-chip{display:inline-block;padding:3px 10px;border-radius:99px;font-size:11px;font-weight:600;color:white;margin-bottom:8px}
.frontmatter{font-size:13px;margin-bottom:12px}
.frontmatter dt{color:#94a3b8;font-weight:600;margin-top:8px}
.frontmatter dd{color:#cbd5e1;margin-left:0}
#detail-body{font-size:14px;line-height:1.6;color:#cbd5e1}
#detail-body h1{font-size:18px;margin:16px 0 8px;color:#f1f5f9}
#detail-body h2{font-size:15px;margin:12px 0 6px;color:#e2e8f0}
#detail-body code{background:#0f172a;padding:2px 5px;border-radius:3px;font-size:13px}
#detail-body pre{background:#0f172a;padding:12px;border-radius:6px;overflow-x:auto;margin:8px 0;font-size:13px}
#detail-body table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
#detail-body th,#detail-body td{border:1px solid #334155;padding:6px 10px;text-align:left}
#detail-body th{background:#1e293b;color:#94a3b8;font-weight:600}
#detail-body a{color:#60a5fa;text-decoration:none}
#detail-body a:hover{text-decoration:underline}
#detail-body blockquote{border-left:3px solid #475569;padding-left:12px;color:#94a3b8;margin:8px 0}
#detail-body ul,#detail-body ol{padding-left:20px;margin:8px 0}
#detail-backlinks{margin-top:20px;padding-top:16px;border-top:1px solid #334155}
#detail-backlinks h2{font-size:14px;color:#94a3b8;margin-bottom:8px}
#backlinks-list{list-style:none;padding:0}
#backlinks-list li{padding:4px 0}
#backlinks-list a{color:#60a5fa;text-decoration:none;font-size:13px}
.muted{color:#64748b}
.title strong{font-size:16px}.title .muted{font-size:13px;margin-left:6px}
</style>
</head>
<body>
<header>
  <div class="title"><strong>__BUNDLE_NAME__</strong><span class="muted">OKF knowledge base</span></div>
  <div class="controls">
    <input id="search" type="search" placeholder="Search title / id / tag">
    <select id="filter-type"><option value="">All types</option>__TYPE_OPTIONS__</select>
    <select id="layout">
      <option value="cose">cose (force)</option>
      <option value="concentric">concentric</option>
      <option value="breadthfirst">breadth-first</option>
      <option value="circle">circle</option>
      <option value="grid">grid</option>
    </select>
    <button id="reset">Reset view</button>
  </div>
</header>
<main>
  <section id="graph"></section>
  <section id="detail">
    <div id="detail-empty" class="muted">Click a node to see its details.</div>
    <article id="detail-content" hidden>
      <header class="detail-header">
        <span class="type-chip" id="detail-type"></span>
        <h1 id="detail-title"></h1>
        <div class="muted" id="detail-id"></div>
      </header>
      <dl class="frontmatter">
        <dt>Description</dt><dd id="detail-description"></dd>
        <dt>Resource</dt><dd id="detail-resource"></dd>
        <dt>Tags</dt><dd id="detail-tags"></dd>
      </dl>
      <hr>
      <div id="detail-body"></div>
      <section id="detail-backlinks" hidden>
        <h2>Cited by</h2>
        <ul id="backlinks-list"></ul>
      </section>
    </article>
  </section>
</main>
<script>
window.BUNDLE_NAME = '__BUNDLE_NAME__';
window.BUNDLE = __BUNDLE_DATA__;
/*VIZ_JS*/
</script>
</body>
</html>"""

viz_js = """
const cy = cytoscape({
  container: document.getElementById('graph'),
  elements: [...BUNDLE.nodes, ...BUNDLE.edges],
  style: [
    { selector: 'node', style: { 'background-color': 'data(color)', 'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center', 'color': '#e2e8f0', 'font-size': '12px', 'width': 'data(size)', 'height': 'data(size)', 'text-wrap': 'wrap', 'text-max-width': '100' } },
    { selector: 'edge', style: { 'width': 1.5, 'line-color': '#475569', 'target-arrow-color': '#475569', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'arrow-scale': 0.8 } },
    { selector: ':selected', style: { 'border-width': 3, 'border-color': '#fbbf24' } }
  ],
  layout: { name: 'cose', nodeRepulsion: () => 8000, idealEdgeLength: () => 120 },
  minZoom: 0.3, maxZoom: 3, wheelSensitivity: 0.3
});

document.getElementById('bundle-name').textContent = BUNDLE_NAME;

const typeSelect = document.getElementById('filter-type');
for (const t of BUNDLE.types || []) {
  const opt = document.createElement('option');
  opt.value = t; opt.textContent = t;
  typeSelect.appendChild(opt);
}

document.getElementById('layout').addEventListener('change', e => {
  cy.layout({ name: e.target.value, nodeRepulsion: () => 8000, idealEdgeLength: () => 120 }).run();
});

document.getElementById('search').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  cy.nodes().forEach(n => {
    const d = n.data();
    const match = !q || d.label.toLowerCase().includes(q) || d.id.toLowerCase().includes(q) || (d.tags || []).some(t => t.toLowerCase().includes(q));
    n.style({ opacity: match ? 1 : 0.2 });
  });
});

document.getElementById('reset').addEventListener('click', () => {
  cy.fit(50);
  cy.nodes().style({ opacity: 1 });
  document.getElementById('search').value = '';
  document.getElementById('filter-type').value = '';
});

let selectedNode = null;
cy.on('tap', 'node', e => {
  const n = e.target;
  const d = n.data();
  selectedNode = d.id;
  document.getElementById('detail-empty').hidden = true;
  document.getElementById('detail-content').hidden = false;
  document.getElementById('detail-type').textContent = d.type || 'Unknown';
  document.getElementById('detail-type').style.backgroundColor = d.color || '#94a3b8';
  document.getElementById('detail-title').textContent = d.label || d.id;
  document.getElementById('detail-id').textContent = d.id;
  document.getElementById('detail-description').textContent = d.description || '(none)';
  const res = document.getElementById('detail-resource');
  if (d.resource) {
    const a = document.createElement('a');
    a.href = d.resource; a.textContent = d.resource; a.target = '_blank';
    res.innerHTML = ''; res.appendChild(a);
  } else { res.textContent = '(none)'; }
  document.getElementById('detail-tags').textContent = (d.tags || []).join(', ') || '(none)';
  const body = BUNDLE.bodies[d.id] || '';
  const html = marked.parse(body);
  document.getElementById('detail-body').innerHTML = html;
  // Rewire internal links
  document.getElementById('detail-body').querySelectorAll('a[href]').forEach(a => {
    const h = a.getAttribute('href');
    if (h.endsWith('.md') && !h.startsWith('http')) {
      a.addEventListener('click', e => { e.preventDefault();
        const targetId = h.replace(/\\//g, '/').replace(/\\.md$/,'').replace(/^\\.\\//,'');
        const node = cy.getElementById(targetId);
        if (node.length) { cy.animate({ center: { eles: node }, zoom: 2 }); /* tap it */ }
      });
    }
  });
  // Backlinks
  const bl = BUNDLE.edges.filter(e => e.data.target === d.id).map(e => e.data.source);
  const blSection = document.getElementById('detail-backlinks');
  if (bl.length) {
    blSection.hidden = false;
    const list = document.getElementById('backlinks-list');
    list.innerHTML = '';
    for (const src of bl) {
      const srcNode = BUNDLE.nodes.find(n => n.data.id === src);
      const label = srcNode ? srcNode.data.label : src;
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#'; a.textContent = label;
      a.addEventListener('click', e => { e.preventDefault();
        const node = cy.getElementById(src);
        if (node.length) { cy.animate({ center: { eles: node }, zoom: 2 }); }
      });
      li.appendChild(a); list.appendChild(li);
    }
  } else { blSection.hidden = true; }
});

cy.on('tap', e => { if (e.target === cy) { selectedNode = null; } });
""".strip()

graph_data = json.dumps({"nodes": nodes, "edges": edges, "bodies": bodies, "types": types})
type_options = "\n".join(f'<option value="{t}">{t}</option>' for t in types)

html = template.replace("__BUNDLE_NAME__", "1bit.systems").replace("__BUNDLE_DATA__", graph_data).replace("/*VIZ_JS*/", viz_js).replace("__TYPE_OPTIONS__", type_options)
out_path.write_text(html, encoding="utf-8")
print(f"Wrote {len(nodes)} nodes, {len(edges)} edges → {out_path} ({len(html.encode('utf-8'))} bytes)")
