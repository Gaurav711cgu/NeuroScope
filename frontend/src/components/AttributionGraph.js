import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { CATEGORICAL_8 } from "@/lib/colors";

export default function AttributionGraph({ graph, width = 520, height = 320 }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!graph || !ref.current) return;
        const svg = d3.select(ref.current);
        svg.selectAll("*").remove();

        const nodes = graph.nodes.map((n) => ({ ...n }));
        const edges = (graph.edges || []).map((e) => ({
            source: e.source,
            target: e.target,
            weight: e.weight,
        }));

        const sim = d3
            .forceSimulation(nodes)
            .force(
                "link",
                d3
                    .forceLink(edges)
                    .id((d) => d.id)
                    .distance((d) => 90 - Math.abs(d.weight) * 60)
                    .strength((d) => Math.abs(d.weight) * 0.6),
            )
            .force("charge", d3.forceManyBody().strength(-180))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(18));

        // Define markers for directed causal edges
        const defs = svg.append("defs");
        
        defs.append("marker")
            .attr("id", "arrow-pos")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 22)
            .attr("refY", 0)
            .attr("markerWidth", 5)
            .attr("markerHeight", 5)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-4L8,0L0,4")
            .attr("fill", "#6EE7B7");

        defs.append("marker")
            .attr("id", "arrow-neg")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 22)
            .attr("refY", 0)
            .attr("markerWidth", 5)
            .attr("markerHeight", 5)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-4L8,0L0,4")
            .attr("fill", "#F87171");

        const link = svg
            .append("g")
            .attr("stroke-opacity", 0.4)
            .selectAll("line")
            .data(edges)
            .enter()
            .append("line")
            .attr("stroke", (d) => (d.weight > 0 ? "#6EE7B7" : "#F87171"))
            .attr("stroke-width", (d) => Math.max(0.5, Math.abs(d.weight) * 2))
            .attr("marker-end", (d) => {
                // If it's a causal graph (path patched), render directed arrows
                if (graph.method === "causal_path_patching" || graph.edges?.[0]?.causal) {
                    return d.weight > 0 ? "url(#arrow-pos)" : "url(#arrow-neg)";
                }
                return null;
            });

        const maxActivation = Math.max(
            1e-6,
            ...nodes.map((n) => Math.abs(n.activation || 0)),
        );

        const node = svg
            .append("g")
            .selectAll("g")
            .data(nodes)
            .enter()
            .append("g");

        node.append("circle")
            .attr("r", (d) => 5 + 14 * (Math.abs(d.activation || 0) / maxActivation))
            .attr("fill", (_, i) => CATEGORICAL_8[i % CATEGORICAL_8.length])
            .attr("stroke", "var(--ns-bg-canvas)")
            .attr("stroke-width", 1.5);

        node.append("text")
            .text((d) => `#${d.id}`)
            .attr("dy", 4)
            .attr("dx", 14)
            .attr("font-family", "JetBrains Mono, monospace")
            .attr("font-size", 10)
            .attr("fill", "var(--ns-fg-secondary)");

        node.call(
            d3
                .drag()
                .on("start", (event, d) => {
                    if (!event.active) sim.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on("drag", (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on("end", (event, d) => {
                    if (!event.active) sim.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }),
        );

        sim.on("tick", () => {
            link.attr("x1", (d) => d.source.x)
                .attr("y1", (d) => d.source.y)
                .attr("x2", (d) => d.target.x)
                .attr("y2", (d) => d.target.y);
            node.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
        });

        return () => sim.stop();
    }, [graph, width, height]);

    return (
        <div
            className="rounded-md border"
            style={{ background: "var(--ns-bg-codeblock)", borderColor: "var(--ns-border-subtle)" }}
            data-testid="attribution-graph"
        >
            <svg ref={ref} width={width} height={height} />
        </div>
    );
}
