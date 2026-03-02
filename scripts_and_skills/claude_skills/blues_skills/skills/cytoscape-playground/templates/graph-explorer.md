# Graph Explorer Template

Use this template when the playground is a Cytoscape.js-powered graph/network visualization:
dependency maps, call graphs, knowledge graphs, module relationship explorers, flow graphs.

## Layout

```
+----------------------+-----------------------------------+
|  Controls:           |                                   |
|  • Layout selector   |  Cytoscape canvas                 |
|  • Direction (TB/LR) |  (drag, pan, zoom, click nodes)   |
|  • Group filter      |                                   |
|  • Edge labels on/off|  [Fit] [Reset] [+] [-]            |
|  • Neighbor highlight|                                   |
|  • [3-5 Presets]     +-----------------------------------+
|                      |  Prompt output        [Copy]      |
+----------------------+-----------------------------------+
```

## Required CDN scripts (include in `<head>`)

```html
<!-- Cytoscape core -->
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/dist/cytoscape.min.js"></script>

<!-- Dagre layout (for DAGs/trees — include when graph has direction) -->
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
```

Register extension before init:
```javascript
cytoscape.use(cytoscapeDagre);
```

## Full HTML skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Graph Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif;
         height: 100vh; display: flex; flex-direction: column; }

  .top { display: flex; flex: 1; overflow: hidden; }

  /* Left sidebar */
  .sidebar { width: 220px; min-width: 220px; padding: 16px; background: #1e293b;
             display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
  .sidebar h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
                color: #64748b; margin-bottom: 4px; }
  .sidebar label { font-size: 13px; color: #94a3b8; display: flex;
                   align-items: center; gap: 6px; cursor: pointer; }
  .sidebar select { width: 100%; background: #0f172a; border: 1px solid #334155;
                    color: #e2e8f0; padding: 5px 8px; border-radius: 4px; font-size: 13px; }

  /* Preset buttons */
  .presets { display: flex; flex-direction: column; gap: 6px; }
  .preset-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
                padding: 6px 10px; border-radius: 4px; cursor: pointer; font-size: 12px;
                text-align: left; }
  .preset-btn:hover { border-color: #63b3ed; color: #e2e8f0; }
  .preset-btn.active { border-color: #3b82f6; color: #93c5fd; background: #1e3a5f; }

  /* Graph canvas area */
  .canvas-wrap { flex: 1; position: relative; }
  #cy { width: 100%; height: 100%; background: #0f172a; }

  /* Graph toolbar */
  .graph-toolbar { position: absolute; top: 12px; right: 12px; display: flex; gap: 6px; }
  .graph-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
               width: 30px; height: 30px; border-radius: 4px; cursor: pointer;
               font-size: 14px; display: flex; align-items: center; justify-content: center; }
  .graph-btn:hover { border-color: #63b3ed; color: #e2e8f0; }

  /* Bottom prompt area */
  .prompt-area { border-top: 1px solid #1e293b; padding: 12px 16px;
                 display: flex; gap: 12px; align-items: flex-start; background: #0f172a; }
  #prompt-output { flex: 1; font-family: ui-monospace, monospace; font-size: 12px;
                   color: #94a3b8; white-space: pre-wrap; line-height: 1.5;
                   min-height: 48px; max-height: 100px; overflow-y: auto; }
  #copy-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8;
              padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px;
              white-space: nowrap; }
  #copy-btn:hover { border-color: #63b3ed; color: #e2e8f0; }
  #copy-btn.copied { border-color: #10b981; color: #6ee7b7; }
</style>
</head>
<body>

<div class="top">
  <!-- Sidebar controls -->
  <div class="sidebar">
    <div>
      <h3>Layout</h3>
      <select id="ctrl-layout">
        <option value="dagre">Dagre (hierarchical)</option>
        <option value="cose">CoSE (force-directed)</option>
        <option value="breadthfirst">Breadth-first (tree)</option>
        <option value="circle">Circle</option>
        <option value="grid">Grid</option>
      </select>
    </div>
    <div>
      <h3>Direction</h3>
      <select id="ctrl-dir">
        <option value="TB">Top → Bottom</option>
        <option value="LR">Left → Right</option>
        <option value="BT">Bottom → Top</option>
        <option value="RL">Right → Left</option>
      </select>
    </div>
    <div>
      <h3>Filter</h3>
      <select id="ctrl-filter">
        <option value="all">All nodes</option>
        <!-- Add group options per domain -->
      </select>
    </div>
    <div>
      <h3>Options</h3>
      <label><input type="checkbox" id="ctrl-edge-labels" checked> Edge labels</label>
      <label><input type="checkbox" id="ctrl-neighbors"> Highlight neighbors</label>
    </div>
    <div>
      <h3>Presets</h3>
      <div class="presets" id="presets">
        <!-- Preset buttons injected by JS -->
      </div>
    </div>
  </div>

  <!-- Cytoscape canvas -->
  <div class="canvas-wrap">
    <div id="cy"></div>
    <div class="graph-toolbar">
      <button class="graph-btn" id="btn-fit" title="Fit">⊡</button>
      <button class="graph-btn" id="btn-reset" title="Reset zoom">↺</button>
      <button class="graph-btn" id="btn-zoom-in" title="Zoom in">+</button>
      <button class="graph-btn" id="btn-zoom-out" title="Zoom out">−</button>
    </div>
  </div>
</div>

<!-- Prompt output -->
<div class="prompt-area">
  <div id="prompt-output"></div>
  <button id="copy-btn">Copy</button>
</div>

<script>
cytoscape.use(cytoscapeDagre);

// ── DATA ──────────────────────────────────────────────────────────────────────
// Replace with real domain data
const ELEMENTS = {
  nodes: [
    { data: { id: 'a', label: 'Module A', group: 'core' } },
    { data: { id: 'b', label: 'Module B', group: 'core' } },
    { data: { id: 'c', label: 'Service C', group: 'external' } },
    { data: { id: 'd', label: 'Module D', group: 'core' } },
  ],
  edges: [
    { data: { id: 'ab', source: 'a', target: 'b', label: 'imports' } },
    { data: { id: 'ac', source: 'a', target: 'c', label: 'calls' } },
    { data: { id: 'bd', source: 'b', target: 'd', label: 'uses' } },
  ]
};

const GROUPS = ['all', 'core', 'external']; // populate filter dropdown

const PRESETS = [
  {
    name: 'Overview',
    state: { layout: 'dagre', rankDir: 'TB', filterGroup: 'all', showEdgeLabels: true }
  },
  {
    name: 'Left→Right',
    state: { layout: 'dagre', rankDir: 'LR', filterGroup: 'all', showEdgeLabels: true }
  },
  {
    name: 'Force',
    state: { layout: 'cose', rankDir: 'TB', filterGroup: 'all', showEdgeLabels: false }
  },
  {
    name: 'Core only',
    state: { layout: 'dagre', rankDir: 'TB', filterGroup: 'core', showEdgeLabels: true }
  },
];

// ── STATE ─────────────────────────────────────────────────────────────────────
const DEFAULTS = { layout: 'dagre', rankDir: 'TB', filterGroup: 'all',
                   showEdgeLabels: true, highlightNeighbors: false };
const state = { ...DEFAULTS };

// ── CYTOSCAPE STYLE ───────────────────────────────────────────────────────────
const CY_STYLE = [
  {
    selector: 'node',
    style: {
      'background-color': '#1e293b',
      'border-color': '#475569',
      'border-width': 1,
      'color': '#e2e8f0',
      'label': 'data(label)',
      'font-size': '11px',
      'font-family': 'ui-monospace, monospace',
      'text-valign': 'center',
      'text-halign': 'center',
      'width': 'label',
      'height': 'label',
      'padding': '10px',
      'shape': 'round-rectangle',
    }
  },
  {
    selector: 'node[group = "external"]',
    style: { 'background-color': '#1e3a5f', 'border-color': '#2563eb' }
  },
  {
    selector: 'edge',
    style: {
      'width': 1.5,
      'line-color': '#475569',
      'target-arrow-color': '#475569',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'label': 'data(label)',
      'font-size': '9px',
      'color': '#64748b',
      'text-background-color': '#0f172a',
      'text-background-opacity': 1,
      'text-background-padding': '2px',
    }
  },
  {
    selector: 'edge[?hideLabel]',
    style: { 'label': '' }
  },
  {
    selector: ':selected',
    style: {
      'border-color': '#60a5fa',
      'border-width': 2,
      'line-color': '#60a5fa',
      'target-arrow-color': '#60a5fa',
    }
  },
  {
    selector: '.highlighted',
    style: { 'background-color': '#1d4ed8', 'border-color': '#60a5fa', 'border-width': 2 }
  },
  {
    selector: '.faded',
    style: { 'opacity': 0.12 }
  },
  {
    selector: '.hidden',
    style: { 'display': 'none' }
  },
];

// ── INIT CYTOSCAPE ────────────────────────────────────────────────────────────
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: ELEMENTS,
  style: CY_STYLE,
  layout: { name: 'dagre', rankDir: 'TB' },
  minZoom: 0.1,
  maxZoom: 5,
  wheelSensitivity: 0.3,
});

// ── POPULATE CONTROLS ─────────────────────────────────────────────────────────
const filterSel = document.getElementById('ctrl-filter');
GROUPS.forEach(g => {
  const opt = document.createElement('option');
  opt.value = g;
  opt.textContent = g === 'all' ? 'All nodes' : g;
  filterSel.appendChild(opt);
});

const presetsEl = document.getElementById('presets');
PRESETS.forEach((p, i) => {
  const btn = document.createElement('button');
  btn.className = 'preset-btn' + (i === 0 ? ' active' : '');
  btn.textContent = p.name;
  btn.addEventListener('click', () => {
    Object.assign(state, p.state);
    syncControls();
    updateAll();
    presetsEl.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
  presetsEl.appendChild(btn);
});

function syncControls() {
  document.getElementById('ctrl-layout').value = state.layout;
  document.getElementById('ctrl-dir').value = state.rankDir;
  document.getElementById('ctrl-filter').value = state.filterGroup;
  document.getElementById('ctrl-edge-labels').checked = state.showEdgeLabels;
  document.getElementById('ctrl-neighbors').checked = state.highlightNeighbors;
}

// ── CONTROL LISTENERS ─────────────────────────────────────────────────────────
document.getElementById('ctrl-layout').addEventListener('change', e => {
  state.layout = e.target.value;
  updateAll();
});
document.getElementById('ctrl-dir').addEventListener('change', e => {
  state.rankDir = e.target.value;
  updateAll();
});
document.getElementById('ctrl-filter').addEventListener('change', e => {
  state.filterGroup = e.target.value;
  updateAll();
});
document.getElementById('ctrl-edge-labels').addEventListener('change', e => {
  state.showEdgeLabels = e.target.checked;
  applyEdgeLabels();
  updatePrompt();
});
document.getElementById('ctrl-neighbors').addEventListener('change', e => {
  state.highlightNeighbors = e.target.checked;
  cy.elements().removeClass('faded highlighted');
  updatePrompt();
});

// ── TOOLBAR BUTTONS ───────────────────────────────────────────────────────────
document.getElementById('btn-fit').addEventListener('click', () => cy.fit(undefined, 40));
document.getElementById('btn-reset').addEventListener('click', () => cy.reset());
document.getElementById('btn-zoom-in').addEventListener('click', () => cy.zoom(cy.zoom() * 1.3));
document.getElementById('btn-zoom-out').addEventListener('click', () => cy.zoom(cy.zoom() / 1.3));

// ── GRAPH INTERACTIONS ────────────────────────────────────────────────────────
cy.on('tap', 'node', function(e) {
  if (!state.highlightNeighbors) return;
  const node = e.target;
  cy.elements().addClass('faded');
  node.neighborhood().add(node).removeClass('faded').addClass('highlighted');
  updatePrompt();
});

cy.on('tap', function(e) {
  if (e.target !== cy) return;
  cy.elements().removeClass('faded highlighted');
  updatePrompt();
});

// ── UPDATE FUNCTIONS ──────────────────────────────────────────────────────────
function applyFilters() {
  cy.elements().removeClass('hidden');
  if (state.filterGroup === 'all') return;
  cy.nodes().forEach(n => {
    if (n.data('group') !== state.filterGroup) {
      n.addClass('hidden');
      n.connectedEdges().addClass('hidden');
    }
  });
}

function applyEdgeLabels() {
  if (state.showEdgeLabels) {
    cy.edges().forEach(e => e.removeData('hideLabel'));
  } else {
    cy.edges().forEach(e => e.data('hideLabel', true));
  }
}

function runLayout() {
  const opts = { name: state.layout, animate: true, animationDuration: 250 };
  if (state.layout === 'dagre') {
    opts.rankDir = state.rankDir;
    opts.nodeSep = 60;
    opts.rankSep = 80;
  }
  const layout = cy.layout(opts);
  layout.on('layoutstop', () => cy.fit(undefined, 40));
  layout.run();
}

function updatePrompt() {
  const visible = cy.nodes(':visible').length;
  const total = cy.nodes().length;
  const selected = cy.nodes('.highlighted');
  const parts = [];

  if (state.filterGroup !== 'all') parts.push(`showing ${state.filterGroup} nodes only`);
  if (selected.length > 0) {
    const names = selected.map(n => n.data('label')).join(', ');
    parts.push(`focused on: ${names} and neighbors`);
  }
  if (state.layout !== 'dagre') parts.push(`using ${state.layout} layout`);

  const context = parts.length ? ' ' + parts.join(', ') + '.' : '';
  document.getElementById('prompt-output').textContent =
    `Graph shows ${visible} of ${total} nodes.${context} Please analyze the relationships and suggest improvements.`;
}

function updateAll() {
  applyFilters();
  applyEdgeLabels();
  runLayout();
  updatePrompt();
}

// ── COPY BUTTON ───────────────────────────────────────────────────────────────
document.getElementById('copy-btn').addEventListener('click', () => {
  const btn = document.getElementById('copy-btn');
  navigator.clipboard.writeText(document.getElementById('prompt-output').textContent).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
  });
});

// ── INITIAL RENDER ────────────────────────────────────────────────────────────
updateAll();
</script>
</body>
</html>
```

## Controls to offer

| Decision | Control type | Example values |
|---|---|---|
| Layout algorithm | Select | Dagre / CoSE / Breadth-first / Circle / Grid |
| Flow direction | Select | Top→Bottom / Left→Right / Bottom→Top / Right→Left |
| Node group filter | Select | All / core / external / [domain groups] |
| Edge labels | Checkbox | Show / hide |
| Neighbor highlight | Checkbox | Click node to highlight neighborhood |
| Presets | Buttons | Overview / Left→Right / Force / Focus on [group] |

## Presets to include

Offer 3-5 presets that each make sense for the domain:
- **Overview** — dagre TB, all nodes, edge labels on
- **Left→Right** — dagre LR, all nodes
- **Force-directed** — cose layout, all nodes (shows clustering naturally)
- **[Group] only** — filter to one meaningful group
- **Compact** — circle or grid, useful for counting/comparing

## Prompt output pattern

State the visible scope (filtered or full), any focused neighborhood, and produce a natural
instruction Claude can act on:

> "Graph shows 12 of 20 nodes (core modules only). Focused on: Router and its neighbors
> (Auth, Logger, Config). Please explain these dependencies and suggest if any can be
> decoupled."

## Example topics

- npm/pip dependency graph (packages as nodes, `depends on` as edges)
- Codebase module graph (files/modules, `imports` edges, grouped by layer)
- Knowledge graph (concepts, `is-a`, `has-a`, `relates-to` edges)
- Workflow / state machine (states as nodes, transitions as edges with labels)
- Microservices map (services, queues, databases — different shapes per type)
- Call graph (functions as nodes, call edges, highlight hot paths)
- Git branch / commit DAG
- Entity-relationship diagram (tables, FK relationships)
