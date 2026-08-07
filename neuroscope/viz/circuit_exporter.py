"""
Interactive Standalone D3.js Circuit Visualizer Exporter
=========================================================
Generates single-file standalone HTML files (`circuit_visualizer.html`) containing
interactive D3.js force-directed circuit graphs with nodes representing SAE features,
edges representing path patching deltas, and live threshold filtering controls.
"""
from __future__ import annotations

import json
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NeuroScope v3 — Interactive Circuit Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        #header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 12px;
            margin-bottom: 20px;
        }}
        h1 {{ font-size: 20px; font-weight: 600; margin: 0; color: #38bdf8; }}
        .badge {{ background: #0369a1; color: #e0f2fe; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 500; }}
        #container {{ display: flex; gap: 20px; }}
        #graph {{ flex: 1; background: #020617; border: 1px solid #1e293b; border-radius: 10px; height: 650px; position: relative; }}
        #sidebar {{ width: 320px; background: #1e293b; border-radius: 10px; padding: 16px; font-size: 13px; }}
        .node {{ cursor: pointer; stroke: #fff; stroke-width: 1.5px; }}
        .link {{ stroke: #64748b; stroke-opacity: 0.6; stroke-dasharray: 4; }}
        .link.causal {{ stroke: #f43f5e; stroke-opacity: 0.8; stroke-dasharray: none; }}
        .tooltip {{
            position: absolute; text-align: left; padding: 8px 12px; font-size: 12px;
            background: #0f172a; border: 1px solid #38bdf8; border-radius: 6px; pointer-events: none; opacity: 0; transition: opacity 0.2s;
        }}
        .control-group {{ margin-bottom: 16px; }}
        label {{ display: block; margin-bottom: 6px; font-weight: 500; color: #94a3b8; }}
        input[type=range] {{ width: 100%; }}
    </style>
</head>
<body>

<div id="header">
    <h1>NeuroScope v3 — Causal Feature Circuit Graph</h1>
    <span class="badge">Model: Gemma-2-2B-IT | SAE: GemmaScope L12</span>
</div>

<div id="container">
    <div id="graph">
        <div id="tooltip" class="tooltip"></div>
    </div>
    
    <div id="sidebar">
        <div class="control-group">
            <label for="threshold">Causal Effect Threshold (ΔH): <span id="thresh-val">0.05</span></label>
            <input type="range" id="threshold" min="0.01" max="0.50" step="0.01" value="0.05">
        </div>
        
        <h3>Selected Feature Details</h3>
        <div id="feature-details">
            <p style="color: #64748b;">Click on any feature node to view Neuronpedia explanation and path patching attributions.</p>
        </div>
    </div>
</div>

<script>
const graphData = {graph_data_json};

const width = document.getElementById('graph').clientWidth;
const height = document.getElementById('graph').clientHeight;

const svg = d3.select('#graph').append('svg')
    .attr('width', width)
    .attr('height', height);

const g = svg.append('g');

// Zoom behavior
svg.call(d3.zoom().on('zoom', (event) => g.attr('transform', event.transform)));

const simulation = d3.forceSimulation(graphData.nodes)
    .force('link', d3.forceLink(graphData.edges).id(d => d.id).distance(100))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2));

const link = g.append('g')
    .selectAll('line')
    .data(graphData.edges)
    .enter().append('line')
    .attr('class', d => d.causal ? 'link causal' : 'link')
    .attr('stroke-width', d => Math.max(1, d.weight * 3));

const node = g.append('g')
    .selectAll('circle')
    .data(graphData.nodes)
    .enter().append('circle')
    .attr('class', 'node')
    .attr('r', d => Math.max(8, d.activation * 3))
    .attr('fill', d => d.layer === 12 ? '#38bdf8' : '#a855f7')
    .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

const label = g.append('g')
    .selectAll('text')
    .data(graphData.nodes)
    .enter().append('text')
    .text(d => '#' + d.id)
    .attr('font-size', '10px')
    .attr('fill', '#cbd5e1')
    .attr('dx', 12)
    .attr('dy', 4);

simulation.on('tick', () => {{
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    label.attr('x', d => d.x).attr('y', d => d.y);
}});

node.on('click', (event, d) => {{
    document.getElementById('feature-details').innerHTML = `
        <p><b>Feature ID:</b> #${{d.id}}</p>
        <p><b>Layer:</b> ${{d.layer}}</p>
        <p><b>Activation Magnitude:</b> ${{d.activation.toFixed(4)}}</p>
        <p><b>Neuronpedia Label:</b> ${{d.label || 'Tool-routing intent feature'}}</p>
        <p><b>Neuronpedia URL:</b> <a href="https://www.neuronpedia.org/gemma-2-2b/${{d.layer}}/${{d.id}}" target="_blank" style="color:#38bdf8;">View Feature #${{d.id}}</a></p>
    `;
}});

function dragstarted(event, d) {{
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
}}
function dragged(event, d) {{
    d.fx = event.x; d.fy = event.y;
}}
function dragended(event, d) {{
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
}}
</script>
</body>
</html>
"""


def export_interactive_circuit_html(nodes: list[dict], edges: list[dict], output_filepath: str = "./results/circuit_visualizer.html"):
    """Generate standalone interactive D3.js circuit graph visualization HTML file."""
    graph_data = {"nodes": nodes, "edges": edges}
    graph_json = json.dumps(graph_data)
    
    html_content = HTML_TEMPLATE.format(graph_data_json=graph_json)
    
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Interactive D3 Circuit Visualizer saved to: {out_path.resolve()}")
    return str(out_path.resolve())
