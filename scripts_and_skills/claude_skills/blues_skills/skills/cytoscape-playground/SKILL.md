---
name: cytoscape-playground
triggers:
  - "graph playground"
  - "cytoscape"
  - "network visualization"
  - "dependency graph"
  - "knowledge graph"
  - "node-link diagram"
description: >
  Extends the playground skill with Cytoscape.js graph/network visualization. Use when
  the user asks for a graph playground, network explorer, dependency map, knowledge graph,
  node-link diagram, or any interactive visualization involving nodes and edges where
  layout algorithms, filtering, and graph traversal matter. Works alongside the core
  playground skill — follow its conventions (single HTML file, live preview, prompt output,
  copy button, dark theme) and add Cytoscape as the rendering engine.
---

# Cytoscape Playground Extension

Cytoscape.js (https://www.npmjs.com/package/cytoscape) is a graph theory library for
interactive node-link diagrams. This skill teaches you how to wire it into a playground
HTML file and optionally set it up as an npm dependency for real projects.

## When to use this skill (vs plain playground templates)

Use Cytoscape when the visualization is fundamentally a **graph** — nodes connected by
edges where layout, traversal, or graph-specific interactions (shortest path, expand/collapse,
neighborhood highlight) are needed. Use the base playground templates for:

- Architecture diagrams with fixed positions → `code-map.md` (SVG)
- Concept maps with manual drag → `concept-map.md` (canvas)

Use Cytoscape when:
- The graph is **data-driven** (unknown shape at authoring time)
- You need **layout algorithms** (hierarchical, force-directed, radial, dagre)
- You need **graph traversal** (select neighbors, shortest path, ancestors)
- The user asks for: dependency graph, knowledge graph, network map, flow graph, call graph

## Setup: two modes

### Mode A — Playground HTML (CDN, zero install)

Load Cytoscape from CDN inside the single HTML file. This is the default for playground
builds. Use jsDelivr for reliability:

```html
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.31.0/dist/cytoscape.min.js"></script>
```

Optional layout extensions (add only if needed):

```html
<!-- Dagre layout (hierarchical, best for DAGs/dependency trees) -->
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>

<!-- Cola layout (force-directed with constraints) -->
<script src="https://cdn.jsdelivr.net/npm/webcola@3.4.0/WebCola/cola.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-cola@2.5.1/cytoscape-cola.js"></script>
```

Register extensions before `cytoscape({...})`:
```javascript
cytoscape.use(cytoscapeDagre); // if using dagre
```

### Mode B — npm project setup (when user wants to integrate into a real project)

When the user says "set up Cytoscape" for a project (not just a playground), give them
the full setup:

```bash
npm install cytoscape
# Optional layout extensions:
npm install cytoscape-dagre dagre
npm install cytoscape-cola webcola
npm install cytoscape-elk
```

**ESM import:**
```javascript
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
cytoscape.use(dagre);
```

**TypeScript types:**
```bash
npm install --save-dev @types/cytoscape
```

**Bundler note:** Cytoscape works with Vite, webpack, Rollup out of the box.
No special config needed — it's a UMD/ESM package.

## Cytoscape initialization pattern

```javascript
const cy = cytoscape({
  container: document.getElementById('cy'),

  elements: {
    nodes: [
      { data: { id: 'a', label: 'Node A', group: 'core' } },
      { data: { id: 'b', label: 'Node B', group: 'external' } },
    ],
    edges: [
      { data: { id: 'ab', source: 'a', target: 'b', label: 'calls', weight: 1 } },
    ]
  },

  style: cytoscapeStyle,   // see Styling section below
  layout: { name: 'dagre', rankDir: 'TB', nodeSep: 60, rankSep: 80 },

  // Good defaults
  minZoom: 0.2,
  maxZoom: 4,
  wheelSensitivity: 0.3,
});
```

## State management pattern (follows playground convention)

```javascript
const state = {
  layout: 'dagre',
  rankDir: 'TB',
  showEdgeLabels: true,
  filterGroup: 'all',
  highlightNeighbors: false,
  // ...
};

function updateAll() {
  applyFilters();
  runLayout();
  updatePrompt();
}
```

## Styling pattern (dark theme)

```javascript
const cytoscapeStyle = [
  {
    selector: 'node',
    style: {
      'background-color': '#2d3748',
      'border-color': '#4a5568',
      'border-width': 1,
      'color': '#e2e8f0',
      'label': 'data(label)',
      'font-size': '11px',
      'font-family': 'ui-monospace, monospace',
      'text-valign': 'center',
      'text-halign': 'center',
      'width': 'label',
      'height': 'label',
      'padding': '8px',
      'shape': 'round-rectangle',
    }
  },
  {
    selector: 'node[group = "external"]',
    style: { 'background-color': '#1a365d', 'border-color': '#2b6cb0' }
  },
  {
    selector: 'edge',
    style: {
      'width': 1.5,
      'line-color': '#4a5568',
      'target-arrow-color': '#4a5568',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'label': 'data(label)',
      'font-size': '9px',
      'color': '#718096',
      'text-background-color': '#1a202c',
      'text-background-opacity': 1,
      'text-background-padding': '2px',
    }
  },
  {
    selector: ':selected',
    style: {
      'border-color': '#63b3ed',
      'border-width': 2,
      'line-color': '#63b3ed',
      'target-arrow-color': '#63b3ed',
    }
  },
  {
    selector: '.highlighted',
    style: { 'background-color': '#2c5282', 'border-color': '#63b3ed', 'border-width': 2 }
  },
  {
    selector: '.faded',
    style: { 'opacity': 0.15 }
  },
];
```

## Layout controls (offer as sidebar dropdowns/buttons)

| Layout name | Best for | Key options |
|---|---|---|
| `dagre` | DAGs, dependency trees, call graphs | `rankDir: TB/LR/BT/RL`, `nodeSep`, `rankSep` |
| `cose` | General graphs, no extension needed | `idealEdgeLength`, `nodeRepulsion` |
| `breadthfirst` | Trees, hierarchies | `directed: true/false`, `circle: false` |
| `circle` | Small graphs, overview | `radius` |
| `grid` | Comparison layouts | `rows`, `cols` |
| `cola` | Force-directed with overlap avoidance | requires extension |

Run a layout:
```javascript
function runLayout() {
  cy.layout({
    name: state.layout,
    rankDir: state.rankDir,  // dagre only
    animate: true,
    animationDuration: 300,
  }).run();
}
```

## Graph interactions to include

```javascript
// Neighbor highlight on click
cy.on('tap', 'node', function(e) {
  const node = e.target;
  cy.elements().addClass('faded');
  node.neighborhood().add(node).removeClass('faded').addClass('highlighted');
});

// Deselect all on background tap
cy.on('tap', function(e) {
  if (e.target === cy) {
    cy.elements().removeClass('faded highlighted');
  }
});

// Fit to viewport
document.getElementById('btn-fit').addEventListener('click', () => cy.fit(undefined, 40));

// Reset zoom
document.getElementById('btn-reset').addEventListener('click', () => cy.reset());
```

## Filtering pattern

```javascript
function applyFilters() {
  if (state.filterGroup === 'all') {
    cy.elements().removeClass('hidden');
  } else {
    cy.nodes().forEach(n => {
      if (n.data('group') !== state.filterGroup) {
        n.addClass('hidden');
        n.connectedEdges().addClass('hidden');
      } else {
        n.removeClass('hidden');
        n.connectedEdges().removeClass('hidden');
      }
    });
  }
}

// Add to style: { selector: '.hidden', style: { display: 'none' } }
```

## Prompt output for graph playgrounds

The prompt should describe the visible graph state and user's selections:

```javascript
function updatePrompt() {
  const visibleNodes = cy.nodes(':visible');
  const selectedNodes = cy.nodes(':selected');
  const parts = [];

  if (state.filterGroup !== 'all') parts.push(`filtered to ${state.filterGroup} nodes`);
  if (selectedNodes.length > 0) {
    const names = selectedNodes.map(n => n.data('label')).join(', ');
    parts.push(`focusing on: ${names}`);
  }
  if (state.layout !== 'dagre') parts.push(`layout: ${state.layout}`);

  prompt.textContent = `Graph has ${visibleNodes.length} visible nodes. ` +
    (parts.length ? parts.join('. ') + '. ' : '') +
    'Please explain the relationships and suggest improvements.';
}
```

## Layout

Follow the playground's layout conventions (dark theme, left controls, right preview, bottom
prompt with copy button). For graph playgrounds, the Cytoscape container IS the preview:

```
+----------------------+---------------------------+
|  Controls:           |                           |
|  • Layout selector   |  Cytoscape canvas         |
|  • Direction (TB/LR) |  (nodes + edges)          |
|  • Group filter      |                           |
|  • Edge labels on/off|  [Fit] [Reset] [+] [-]    |
|  • Neighbor highlight|                           |
|  • Presets           +---------------------------+
|                      |  Prompt output  [Copy]    |
+----------------------+---------------------------+
```

## Common mistakes to avoid

- Calling `cy.layout().run()` before elements are added → always init elements first
- Not destroying the old Cytoscape instance when re-rendering → `cy.destroy()` first, or update elements in place with `cy.json({ elements })`
- Forgetting to call `cy.fit()` after layout completes → use `layout.on('layoutstop', () => cy.fit())`
- Loading dagre extension but forgetting `cytoscape.use(cytoscapeDagre)` → register before `cytoscape({...})`
- Container div has no explicit height → Cytoscape needs a pixel height on its container
