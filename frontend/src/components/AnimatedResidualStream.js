import React, { useEffect, useRef } from "react";

export default function AnimatedResidualStream() {
    const canvasRef = useRef(null);
    const mouseRef = useRef({ x: -1000, y: -1000, active: false });

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        let animationFrameId;
        let width = (canvas.width = canvas.offsetWidth);
        let height = (canvas.height = canvas.offsetHeight);

        const handleResize = () => {
            if (canvas) {
                width = canvas.width = canvas.offsetWidth;
                height = canvas.height = canvas.offsetHeight;
                initNodes();
            }
        };

        window.addEventListener("resize", handleResize);

        // Nodes definition: 5 layers (representing L0, L6, L12, L18, L24)
        const numLayers = 5;
        const nodesPerLayer = 4;
        let layers = [];

        const initNodes = () => {
            layers = [];
            const layerSpacing = width / (numLayers + 1);
            const paddingY = 40;
            const usableHeight = height - paddingY * 2;

            for (let i = 0; i < numLayers; i++) {
                const x = layerSpacing * (i + 1);
                const nodes = [];
                for (let j = 0; j < nodesPerLayer; j++) {
                    const y = paddingY + (usableHeight / (nodesPerLayer - 1)) * j;
                    nodes.push({
                        id: `${i}-${j}`,
                        x,
                        y,
                        baseX: x,
                        baseY: y,
                        radius: 4 + Math.random() * 3,
                        pulse: Math.random() * Math.PI * 2,
                        pulseSpeed: 0.02 + Math.random() * 0.03,
                        active: false,
                        activationValue: 0.1 + Math.random() * 0.9,
                    });
                }
                layers.push(nodes);
            }
        };

        initNodes();

        // Particles definition
        const particles = [];
        const maxParticles = 60;

        const spawnParticle = () => {
            if (layers.length < 2) return;
            const srcLayerIdx = Math.floor(Math.random() * (numLayers - 1));
            const tgtLayerIdx = srcLayerIdx + 1;
            const srcNodes = layers[srcLayerIdx];
            const tgtNodes = layers[tgtLayerIdx];
            const srcNode = srcNodes[Math.floor(Math.random() * srcNodes.length)];
            const tgtNode = tgtNodes[Math.floor(Math.random() * tgtNodes.length)];

            particles.push({
                src: srcNode,
                tgt: tgtNode,
                progress: 0,
                speed: 0.004 + Math.random() * 0.008,
                size: 1.5 + Math.random() * 1.5,
                color: Math.random() > 0.3 ? "#4fb3c8" : "#f2c14e", // Cyan or Amber
            });
        };

        // Pre-populate particles
        for (let i = 0; i < maxParticles; i++) {
            spawnParticle();
            if (particles[i]) {
                particles[i].progress = Math.random();
            }
        }

        const handleMouseMove = (e) => {
            const rect = canvas.getBoundingClientRect();
            mouseRef.current = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
                active: true,
            };
        };

        const handleMouseLeave = () => {
            mouseRef.current = { x: -1000, y: -1000, active: false };
        };

        canvas.addEventListener("mousemove", handleMouseMove);
        canvas.addEventListener("mouseleave", handleMouseLeave);

        const drawBezierPath = (x1, y1, x2, y2, controlScale = 0.5) => {
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            const cp1x = x1 + (x2 - x1) * controlScale;
            const cp2x = x2 - (x2 - x1) * controlScale;
            ctx.bezierCurveTo(cp1x, y1, cp2x, y2, x2, y2);
        };

        const render = () => {
            // Fade effect to create trailing paths
            ctx.fillStyle = "rgba(11, 15, 20, 0.12)";
            ctx.fillRect(0, 0, width, height);

            const mouse = mouseRef.current;

            // Update & draw connection lines (background paths)
            ctx.lineWidth = 0.8;
            for (let i = 0; i < numLayers - 1; i++) {
                const currentLayer = layers[i];
                const nextLayer = layers[i + 1];

                for (const nodeA of currentLayer) {
                    for (const nodeB of nextLayer) {
                        // Check if mouse is close to the connection
                        const midX = (nodeA.x + nodeB.x) / 2;
                        const midY = (nodeA.y + nodeB.y) / 2;
                        const distToMouse = Math.hypot(mouse.x - midX, mouse.y - midY);
                        const isClose = mouse.active && distToMouse < 80;

                        ctx.strokeStyle = isClose
                            ? "rgba(79, 179, 200, 0.25)"
                            : "rgba(38, 52, 74, 0.08)";
                        
                        drawBezierPath(nodeA.x, nodeA.y, nodeB.x, nodeB.y);
                        ctx.stroke();
                    }
                }
            }

            // Update & draw particles
            for (let i = particles.length - 1; i >= 0; i--) {
                const p = particles[i];
                p.progress += p.speed;

                // Causal steering warp effect near cursor
                let drawX = p.src.x + (p.tgt.x - p.src.x) * p.progress;
                let drawY = p.src.y + (p.tgt.y - p.src.y) * p.progress;

                // Apply bezier curve to actual drawing position to match path
                const cp1x = p.src.x + (p.tgt.x - p.src.x) * 0.5;
                const cp2x = p.tgt.x - (p.tgt.x - p.src.x) * 0.5;
                const t = p.progress;
                const mt = 1 - t;
                
                drawX = mt * mt * mt * p.src.x + 3 * mt * mt * t * cp1x + 3 * mt * t * t * cp2x + t * t * t * p.tgt.x;
                drawY = mt * mt * mt * p.src.y + 3 * mt * mt * t * p.src.y + 3 * mt * t * t * p.tgt.y + t * t * t * p.tgt.y;

                if (mouse.active) {
                    const dx = mouse.x - drawX;
                    const dy = mouse.y - drawY;
                    const dist = Math.hypot(dx, dy);
                    if (dist < 100) {
                        const force = (100 - dist) / 100;
                        drawX += (dx / dist) * force * 15;
                        drawY += (dy / dist) * force * 15;
                    }
                }

                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(drawX, drawY, p.size, 0, Math.PI * 2);
                ctx.fill();

                // Reset particle on completion
                if (p.progress >= 1) {
                    particles.splice(i, 1);
                    spawnParticle();
                }
            }

            // Draw nodes
            for (const layer of layers) {
                for (const node of layer) {
                    node.pulse += node.pulseSpeed;
                    
                    const distToMouse = mouse.active ? Math.hypot(mouse.x - node.x, mouse.y - node.y) : 999;
                    const isHot = distToMouse < 60;
                    
                    // Ripple effect
                    const glowRadius = node.radius + Math.sin(node.pulse) * 1.5;
                    
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, glowRadius + (isHot ? 4 : 0), 0, Math.PI * 2);
                    
                    if (isHot) {
                        ctx.fillStyle = "rgba(79, 179, 200, 0.4)";
                        ctx.strokeStyle = "#6ee7b7"; // Mint border on activation focus
                        ctx.lineWidth = 1.2;
                        ctx.fill();
                        ctx.stroke();
                    } else {
                        // Gradient fill based on activation value
                        ctx.fillStyle = `rgba(79, 179, 200, ${0.1 + node.activationValue * 0.4})`;
                        ctx.strokeStyle = "rgba(38, 52, 74, 0.3)";
                        ctx.lineWidth = 0.8;
                        ctx.fill();
                        ctx.stroke();
                    }

                    // Activation node core center
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, 2, 0, Math.PI * 2);
                    ctx.fillStyle = isHot ? "#6ee7b7" : "#4fb3c8";
                    ctx.fill();
                }
            }

            // Mouse indicator
            if (mouse.active) {
                ctx.beginPath();
                ctx.arc(mouse.x, mouse.y, 80, 0, Math.PI * 2);
                const grad = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 80);
                grad.addColorStop(0, "rgba(79, 179, 200, 0.04)");
                grad.addColorStop(1, "rgba(79, 179, 200, 0)");
                ctx.fillStyle = grad;
                ctx.fill();
            }

            animationFrameId = requestAnimationFrame(render);
        };

        render();

        return () => {
            window.removeEventListener("resize", handleResize);
            cancelAnimationFrame(animationFrameId);
            if (canvas) {
                canvas.removeEventListener("mousemove", handleMouseMove);
                canvas.removeEventListener("mouseleave", handleMouseLeave);
            }
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 h-full w-full pointer-events-auto"
            style={{ zIndex: 1, mixBlendMode: "screen", opacity: 0.6 }}
        />
    );
}
