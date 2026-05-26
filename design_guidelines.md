{
  "brand": {
    "name": "NeuroScope",
    "positioning": "Agentic mechanistic interpretability research tool (not SaaS). Dense, terminal-adjacent, JupyterLab-like. Credible for Anthropic/OpenAI/DeepMind interp teams.",
    "design_personality": [
      "research-grade",
      "dense-but-legible",
      "instrumentation UI",
      "calm precision",
      "editorial annotations (Distill-like)"
    ]
  },
  "design_tokens": {
    "color_palette_hex": {
      "bg": {
        "canvas": "#0B0F14",
        "surface_1": "#0F1620",
        "surface_2": "#121C28",
        "surface_3": "#162233",
        "overlay": "#0A0E13",
        "codeblock": "#0A111A"
      },
      "fg": {
        "primary": "#E6EDF6",
        "secondary": "#B7C3D6",
        "muted": "#7F8CA3",
        "faint": "#5B667A",
        "inverse": "#0B0F14"
      },
      "border": {
        "subtle": "#1E2A3B",
        "default": "#26344A",
        "strong": "#334764"
      },
      "accent": {
        "primary_ocean": "#4FB3C8",
        "primary_ocean_2": "#2E8FA6",
        "mint": "#6EE7B7",
        "amber": "#F2C14E",
        "rose": "#F87171",
        "violet_disabled": "#6B7280"
      },
      "semantic": {
        "info": "#4FB3C8",
        "success": "#6EE7B7",
        "warning": "#F2C14E",
        "danger": "#F87171",
        "neutral": "#7F8CA3",
        "patch": "#A7F3D0",
        "flag": "#FBBF24",
        "active": "#4FB3C8"
      },
      "viz": {
        "gridline": "#1B2636",
        "axis": "#2A3A52",
        "heatmap_sequential": [
          "#0B0F14",
          "#102033",
          "#163A5A",
          "#1F5F7A",
          "#2E8FA6",
          "#4FB3C8",
          "#8BE3F0"
        ],
        "diverging": {
          "neg": "#F87171",
          "zero": "#1E2A3B",
          "pos": "#6EE7B7"
        },
        "categorical_8": [
          "#4FB3C8",
          "#6EE7B7",
          "#F2C14E",
          "#F87171",
          "#93C5FD",
          "#34D399",
          "#F59E0B",
          "#FB7185"
        ]
      },
      "focus_ring": "#7DD3FC",
      "selection": {
        "bg": "#0E2A3A",
        "border": "#2E8FA6"
      }
    },
    "css_custom_properties": {
      "instructions": "Implement these in /app/frontend/src/index.css under :root and .dark. Use HSL tokens for shadcn variables, but keep HEX tokens as additional custom vars for D3/Recharts. Avoid transition: all.",
      "add_vars": {
        "--ns-bg-canvas": "#0B0F14",
        "--ns-bg-surface-1": "#0F1620",
        "--ns-bg-surface-2": "#121C28",
        "--ns-bg-surface-3": "#162233",
        "--ns-fg-primary": "#E6EDF6",
        "--ns-fg-secondary": "#B7C3D6",
        "--ns-fg-muted": "#7F8CA3",
        "--ns-border": "#26344A",
        "--ns-border-subtle": "#1E2A3B",
        "--ns-accent": "#4FB3C8",
        "--ns-accent-2": "#2E8FA6",
        "--ns-success": "#6EE7B7",
        "--ns-warning": "#F2C14E",
        "--ns-danger": "#F87171",
        "--ns-focus": "#7DD3FC",
        "--ns-shadow-elev-1": "0 1px 0 rgba(255,255,255,0.04), 0 10px 30px rgba(0,0,0,0.35)",
        "--ns-shadow-elev-2": "0 1px 0 rgba(255,255,255,0.05), 0 18px 50px rgba(0,0,0,0.45)",
        "--ns-radius-sm": "8px",
        "--ns-radius-md": "12px",
        "--ns-radius-lg": "16px"
      },
      "shadcn_dark_hsl_overrides": {
        "--background": "215 45% 6%",
        "--foreground": "215 35% 94%",
        "--card": "215 40% 8%",
        "--card-foreground": "215 35% 94%",
        "--popover": "215 45% 7%",
        "--popover-foreground": "215 35% 94%",
        "--primary": "191 55% 55%",
        "--primary-foreground": "215 45% 8%",
        "--secondary": "215 30% 14%",
        "--secondary-foreground": "215 35% 94%",
        "--muted": "215 28% 14%",
        "--muted-foreground": "215 18% 68%",
        "--accent": "215 30% 14%",
        "--accent-foreground": "215 35% 94%",
        "--destructive": "0 78% 60%",
        "--destructive-foreground": "215 35% 94%",
        "--border": "215 28% 20%",
        "--input": "215 28% 20%",
        "--ring": "199 85% 70%",
        "--radius": "0.75rem"
      }
    },
    "typography": {
      "google_fonts": {
        "ui_sans": "IBM Plex Sans",
        "mono": "JetBrains Mono"
      },
      "font_usage": {
        "body": "IBM Plex Sans (400/500)",
        "headings": "IBM Plex Sans (600)",
        "numbers_ids_code": "JetBrains Mono (400/500)",
        "viz_axes": "JetBrains Mono (400)"
      },
      "tailwind_font_classes": {
        "ui": "font-sans",
        "mono": "font-mono tabular-nums"
      },
      "type_scale": {
        "h1": "text-4xl sm:text-5xl lg:text-6xl font-semibold tracking-tight",
        "h2": "text-base md:text-lg font-medium text-[color:var(--ns-fg-secondary)]",
        "section_title": "text-sm font-semibold uppercase tracking-[0.12em] text-[color:var(--ns-fg-muted)]",
        "body": "text-sm md:text-base leading-6 text-[color:var(--ns-fg-primary)]",
        "caption": "text-xs leading-5 text-[color:var(--ns-fg-muted)]",
        "mono_small": "text-xs font-mono tabular-nums text-[color:var(--ns-fg-secondary)]"
      }
    },
    "spacing_and_grid": {
      "spacing_scale_px": {
        "1": 4,
        "2": 8,
        "3": 12,
        "4": 16,
        "5": 20,
        "6": 24,
        "8": 32,
        "10": 40,
        "12": 48,
        "16": 64
      },
      "layout": {
        "desktop_grid": "12-col, max-w-[1440px], px-6, gap-6",
        "analysis_shell": "3-pane: left nav (260px), main (fluid), right inspector (360px) using shadcn Resizable",
        "mobile": "single column; right inspector becomes Sheet/Drawer"
      },
      "density_rules": [
        "Default card padding: p-4 (dense) or p-5 (hero viz)",
        "Table row height: 32–36px",
        "Chart panels: keep titles compact; use captions for methodology notes"
      ]
    },
    "shadows_and_surfaces": {
      "surface_recipe": "Use subtle 1px top highlight + deep shadow for elevation. Prefer borders over heavy shadows.",
      "card_class": "bg-[color:var(--ns-bg-surface-1)] border border-[color:var(--ns-border-subtle)] rounded-[var(--ns-radius-md)] shadow-[var(--ns-shadow-elev-1)]"
    },
    "radius": {
      "buttons": "10–12px",
      "cards": "12–16px",
      "chips_badges": "9999px (pill)"
    },
    "motion": {
      "principles": [
        "Technical, subtle, fast",
        "No marketing easing; prefer ease-out",
        "Animate opacity/position only; avoid layout thrash",
        "Respect prefers-reduced-motion"
      ],
      "durations_ms": {
        "hover": 120,
        "panel": 180,
        "viz_highlight": 80
      },
      "tailwind": {
        "hover": "transition-colors duration-150",
        "panel": "transition-opacity duration-200",
        "focus": "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ns-focus)] focus-visible:ring-offset-0"
      }
    }
  },
  "component_path": {
    "shadcn_primary": "/app/frontend/src/components/ui",
    "use_components": {
      "button": "button.jsx",
      "input": "input.jsx",
      "textarea": "textarea.jsx",
      "slider": "slider.jsx",
      "select": "select.jsx",
      "tabs": "tabs.jsx",
      "table": "table.jsx",
      "badge": "badge.jsx",
      "card": "card.jsx",
      "dialog": "dialog.jsx",
      "sheet": "sheet.jsx",
      "resizable": "resizable.jsx",
      "scroll_area": "scroll-area.jsx",
      "tooltip": "tooltip.jsx",
      "hover_card": "hover-card.jsx",
      "popover": "popover.jsx",
      "separator": "separator.jsx",
      "progress": "progress.jsx",
      "sonner_toast": "sonner.jsx",
      "command_palette": "command.jsx",
      "kbd_hint": "(use Badge + font-mono + border)"
    },
    "notes": "Do not use raw HTML dropdown/calendar/toast; use shadcn components above. Files are .jsx (not .tsx)."
  },
  "iconography": {
    "library": "lucide-react",
    "stroke": "1.75",
    "common_icons": {
      "run": "Play",
      "experiments": "FlaskConical",
      "docs": "BookOpen",
      "search": "Search",
      "patch": "Wand2",
      "risk": "TriangleAlert",
      "layer": "Layers",
      "feature": "Fingerprint",
      "copy": "Copy",
      "external": "ArrowUpRight",
      "status_ok": "CheckCircle2",
      "status_working": "LoaderCircle (only for <2s inline)",
      "status_error": "XCircle"
    }
  },
  "visual_recipes_key_functionalities": {
    "1_activation_timeline_heatmap": {
      "layout": "Hero panel at top of /run/:id and /experiments/:slug. Left axis = layers (0–11), top axis = steps (1–N).",
      "style": {
        "cell_size": "desktop 22x18px; compact mode 18x14px",
        "cell_gap": "1px",
        "cell_radius": "2px",
        "gridline": "stroke var(--ns-border-subtle)",
        "color_scale": "sequential using viz.heatmap_sequential; map low values to bg.canvas and high to #8BE3F0",
        "hover": "show tooltip with layer, step, L2 norm; highlight row/col with subtle overlay",
        "selection": "selected cell gets 2px outline in --ns-focus"
      },
      "testids": {
        "container": "activation-timeline-heatmap",
        "tooltip": "activation-heatmap-tooltip"
      }
    },
    "2_sae_feature_drift_chart": {
      "chart": "Recharts LineChart (25 lines) with legend collapsed by default; click feature to pin.",
      "style": {
        "line_width": "1.25",
        "inactive_opacity": "0.18",
        "active_opacity": "1",
        "drift_encoding": "color intensity: low drift -> muted blue-gray; high drift -> accent.primary_ocean",
        "axes": "mono ticks; subtle gridlines",
        "interaction": "hover shows feature_id + label + per-step value; click toggles pinned state"
      },
      "testids": {
        "container": "sae-feature-drift-chart",
        "legend": "sae-feature-drift-legend"
      }
    },
    "3_hallucination_risk_timeline": {
      "chart": "Recharts AreaChart with threshold line at 0.65 and per-step markers.",
      "integrity_note": "Inline caption under chart: 'Early warning diagnostic — intervention success rate ~20–30% in literature.'",
      "style": {
        "area_fill": "solid (no gradient) using accent.primary_ocean with 18% opacity",
        "threshold": "stroke warning (#F2C14E) dashed",
        "risk_above": "points above threshold get warning ring",
        "breakdown": "on hover, show entropy/attention diffusion/uncertainty in tooltip"
      },
      "testids": {
        "container": "hallucination-risk-timeline",
        "integrity-note": "hallucination-risk-integrity-note"
      }
    },
    "4_cross_step_patch_matrix": {
      "layout": "Matrix panel with right-side inspector. Click cell -> run patch -> show token shifts.",
      "style": {
        "cell": "square 20px; 1px gap; radius 2px",
        "color_scale": "sequential teal->cyan; extreme KL values clamp to top color",
        "selected": "outline 2px focus color; show source/target labels",
        "inspector": "Right panel shows top-5 token probability shifts as table with delta bars"
      },
      "testids": {
        "matrix": "cross-step-patch-matrix",
        "cell": "cross-step-patch-matrix-cell",
        "inspector": "cross-step-patch-inspector"
      }
    },
    "5_nl_circuit_query_box": {
      "layout": "Chat-like box anchored bottom-right of main analysis view; collapsible.",
      "style": {
        "input": "Textarea with mono placeholders for step/layer references",
        "messages": "Assistant responses include citations chips: Step 3 · Layer 7 · Feature 1204",
        "citations": "Clickable badges that deep-link to step detail panel"
      },
      "testids": {
        "container": "nl-circuit-query-box",
        "input": "nl-circuit-query-input",
        "send": "nl-circuit-query-send-button",
        "message": "nl-circuit-query-message"
      }
    },
    "6_attribution_graph": {
      "viz": "D3 force-directed graph; nodes=features, edges=causal links.",
      "style": {
        "node": "radius 4–8 based on importance; fill uses categorical_8",
        "edge": "stroke axis color; opacity 0.35; highlight path on hover",
        "labels": "only on hover or pinned selection to avoid clutter",
        "panel": "Graph sits in Card with a compact toolbar (zoom, reset, pin)"
      },
      "testids": {
        "container": "attribution-graph",
        "toolbar": "attribution-graph-toolbar"
      }
    },
    "7_step_detail_panel": {
      "layout": "Split: main content (output + charts) and right inspector (top features + signals).",
      "style": {
        "output": "Code-like block with line numbers; tokens highlighted on hover",
        "top_features": "Table with feature_id (mono), label, activation, drift; row hover reveals HoverCard with examples",
        "signals": "Stacked mini-bars for entropy/attention diffusion/uncertainty"
      },
      "testids": {
        "container": "step-detail-panel",
        "top-features": "step-top-features-table",
        "signals": "step-hallucination-signal-breakdown"
      }
    },
    "8_prebuilt_experiments_library": {
      "layout": "5 cards in 2-column grid desktop; each card shows question, key finding, and a tiny sparkline preview.",
      "style": {
        "card": "border strong on hover; subtle lift",
        "tagging": "Badges: 'Hallucination', 'Tool-use', 'IOI', etc",
        "preview": "Mini heatmap strip (steps x layers) as thumbnail"
      },
      "testids": {
        "grid": "experiments-library-grid",
        "card": "experiments-library-card"
      }
    },
    "9_run_create_panel": {
      "layout": "Form-first page: task input, n_steps slider (3–6), suggested tasks list, start button.",
      "style": {
        "suggested_tasks": "Command palette style list; click to fill input",
        "slider": "Show tick marks 3–6; value in mono badge",
        "cta": "Primary button in ocean accent; secondary ghost for 'Load experiment'"
      },
      "testids": {
        "task-input": "run-create-task-input",
        "n-steps-slider": "run-create-n-steps-slider",
        "start": "run-create-start-analysis-button"
      }
    },
    "10_reproducibility_snippet": {
      "component": "Code block with Copy button; show language=python; include run_id.",
      "style": {
        "code": "bg codeblock, border subtle, mono, small",
        "copy": "Icon button top-right; toast via sonner"
      },
      "testids": {
        "container": "reproducibility-snippet",
        "copy": "reproducibility-copy-button"
      }
    }
  },
  "component_specs": {
    "buttons": {
      "variants": {
        "primary": "Button (shadcn) with bg --ns-accent, text --ns-fg-inverse, hover darken to --ns-accent-2",
        "secondary": "outline: border --ns-border, bg transparent, hover bg surface_2",
        "ghost": "no border, hover bg surface_2",
        "danger": "bg --ns-danger, hover slightly darker"
      },
      "sizes": {
        "sm": "h-8 px-3 text-xs",
        "md": "h-9 px-4 text-sm",
        "lg": "h-10 px-5 text-sm"
      },
      "micro_interactions": "hover: transition-colors; active: scale-[0.98] (only on button)"
    },
    "inputs": {
      "text": "Input/Textarea with bg surface_2, border subtle, focus ring --ns-focus, placeholder fg-muted",
      "search": "Use Command component for global search (feature/layer/step)"
    },
    "slider": {
      "n_steps": "Use shadcn Slider; add tick labels 3–6 below; show current value in mono Badge"
    },
    "dropdown_select": {
      "select": "Use shadcn Select for model/run selection; keep menu dense with ScrollArea"
    },
    "modal_drawer": {
      "dialog": "Use Dialog for methodology notes and patch confirmation",
      "sheet": "Use Sheet for right inspector on smaller screens"
    },
    "tabs": {
      "usage": "Use Tabs for switching between 'Overview / Patching / Features / Logs' within /run/:id"
    },
    "side_panel": {
      "right_inspector": "Use Resizable + ScrollArea; sticky header with run id + copy buttons"
    },
    "status_indicator": {
      "states": {
        "queued": "muted dot",
        "running": "info dot + subtle pulse",
        "done": "success dot",
        "error": "danger dot"
      },
      "implementation": "Badge + small inline span with rounded-full w-2 h-2"
    },
    "data_table": {
      "component": "shadcn Table",
      "style": "dense rows, mono numeric columns, sticky header, row hover bg surface_2",
      "columns": [
        "feature_id (mono)",
        "label",
        "activation",
        "drift",
        "layer",
        "links"
      ]
    },
    "code_block": {
      "style": "pre with font-mono text-xs, bg codeblock, border subtle, rounded-md, overflow-auto",
      "extras": "Optional line numbers; Copy button"
    },
    "kbd_shortcut": {
      "style": "Badge variant outline + font-mono text-[10px] tracking-wide",
      "examples": [
        "⌘K for Command",
        "G then S for Step",
        "P to toggle Patch mode"
      ]
    }
  },
  "page_layouts": {
    "/": {
      "wireframe": [
        "Top bar: NeuroScope wordmark (mono accent dot) | Experiments | Docs | Run",
        "Hero: left = thesis + 3 bullets; right = embedded preview card (mini heatmap + risk chart)",
        "Section: 'How it works' as 4-step pipeline (Hook → Capture → SAE → Patch)",
        "Section: credibility strip (citations + limitations)",
        "Footer: repo link + methodology"
      ],
      "testids": {
        "cta-run": "landing-cta-run",
        "cta-experiments": "landing-cta-experiments"
      }
    },
    "/run": {
      "wireframe": [
        "Header: Create Run",
        "Card: Task input (Textarea)",
        "Row: n_steps slider + estimated runtime",
        "Suggested tasks list (Command-like)",
        "Primary CTA: Start Analysis",
        "Secondary: Load pre-built experiment"
      ]
    },
    "/run/:id": {
      "wireframe": [
        "3-pane shell (Resizable):",
        "Left: Step list with status + quick metrics",
        "Main: Activation Timeline Heatmap (hero) → SAE Drift → Risk Timeline → Patch Matrix",
        "Right: Inspector (selected step/cell/feature) + token shift table + reproducibility snippet",
        "Bottom-right: NL Circuit Query Box (collapsible)"
      ]
    },
    "/run/:id/step/:n": {
      "wireframe": [
        "Top: breadcrumb Run / Step n",
        "Main: Output text block + attribution graph",
        "Right: Top features table + hallucination breakdown + links to patch targets",
        "Below: mini heatmap strip for context"
      ]
    },
    "/experiments": {
      "wireframe": [
        "Header: Experiments Library",
        "Filters row (Badges/toggles)",
        "Grid: 5 experiment cards with preview + finding",
        "Right rail (optional): 'What to look for' notes"
      ]
    },
    "/experiments/:slug": {
      "wireframe": [
        "Same as /run/:id but with 'Experiment' header and read-only badges",
        "Add: 'Key finding' callout card near top"
      ]
    },
    "/docs": {
      "wireframe": [
        "Left: docs nav (ScrollArea)",
        "Main: methodology article with code blocks and figures",
        "Inline callouts: limitations + uncertainty",
        "Bottom: reproducibility + citations"
      ]
    }
  },
  "loading_empty_states": {
    "loading": {
      "rule": "Never show generic spinner for >5s. Use step-by-step progress log.",
      "pattern": "Card with Progress + log lines: 'Hooking layer 7… capturing residual stream… SAE decomposing… computing drift score…'",
      "components": [
        "progress.jsx",
        "skeleton.jsx",
        "scroll-area.jsx"
      ],
      "testids": {
        "container": "analysis-loading-state",
        "log": "analysis-loading-log"
      }
    },
    "empty": {
      "pattern": "Explain what data is missing + provide next action (Load experiment / Start run).",
      "examples": {
        "no_run": "No run loaded. Choose an experiment or start a new trajectory.",
        "no_patch": "Select a source/target cell to run causal patching.",
        "no_feature": "Click a feature line to pin details."
      }
    },
    "error": {
      "pattern": "Alert component with error code (mono) + retry button + link to docs.",
      "component": "alert.jsx"
    }
  },
  "libraries_and_scaffolds": {
    "framer_motion": {
      "need": "Optional; only for panel entrance/exit and subtle list transitions.",
      "install": "npm i framer-motion",
      "usage": "Use motion.div for opacity/y transitions on inspector and chat box; keep durations <=200ms. Respect prefers-reduced-motion."
    },
    "d3": {
      "need": "Heatmaps + force graph.",
      "notes": "Use canvas for heatmaps if performance issues; otherwise SVG with rects is fine for 12x6 grids."
    },
    "recharts": {
      "need": "Risk timeline + drift chart.",
      "notes": "Use custom tooltip with mono numbers; disable animations for dense mode or large datasets."
    }
  },
  "image_urls": {
    "note": "This product is primarily data-viz; avoid stock photography. Use subtle procedural textures only.",
    "textures": [
      {
        "category": "noise_overlay",
        "description": "CSS-only noise overlay for hero/landing background (no external image).",
        "url": "(generate via CSS)"
      }
    ]
  },
  "honest_research_tool_patterns": {
    "limitations_callouts": {
      "style": "Inline callout card with border-l-2 warning color, bg surface_2, caption text.",
      "examples": [
        "Hallucination risk is an early warning diagnostic; intervention success rate ~20–30% in literature.",
        "SAE features are approximate; labels may be misleading; validate with causal tests.",
        "Causal patching can introduce distribution shift; interpret KL deltas cautiously."
      ]
    },
    "uncertainty_display": {
      "pattern": "Show confidence bands or 'low/med/high' uncertainty badges next to key claims; never present as certainty.",
      "testids": {
        "uncertainty-badge": "uncertainty-badge"
      }
    },
    "methodology_links": {
      "pattern": "Every viz panel has a small 'Method' link opening Dialog with definitions and equations.",
      "testids": {
        "method-link": "panel-methodology-link"
      }
    }
  },
  "instructions_to_main_agent": {
    "global": [
      "Replace default CRA App.css centering; do not use .App { text-align:center }.",
      "Set <html className=\"dark\"> or apply 'dark' class at root to enable dark tokens.",
      "Add Google Fonts (IBM Plex Sans + JetBrains Mono) in public/index.html or via CSS import.",
      "Use font-mono + tabular-nums for all numeric readouts, IDs, KL, layer/step indices.",
      "All interactive + key informational elements must include data-testid in kebab-case.",
      "Avoid gradients except mild background accents under 20% viewport; no saturated purple/pink gradients.",
      "Prefer Resizable 3-pane shell for /run/:id; right inspector collapses to Sheet on small screens.",
      "Tooltips and hover cards are essential for density without clutter."
    ],
    "suggested_global_components": [
      "TopBar with Command palette (⌘K) using command.jsx",
      "Left StepList with status dots",
      "Right InspectorPanel with tabs",
      "VizPanel wrapper (Card + header + method link + actions)"
    ],
    "css_notes": [
      "Do not use transition: all.",
      "Use subtle noise overlay via pseudo-element on landing hero only.",
      "Keep borders visible; dark UIs need separators."
    ]
  },
  "general_ui_ux_design_guidelines": "    - You must **not** apply universal transition. Eg: `transition: all`. This results in breaking transforms. Always add transitions for specific interactive elements like button, input excluding transforms\n    - You must **not** center align the app container, ie do not add `.App { text-align: center; }` in the css file. This disrupts the human natural reading flow of text\n   - NEVER: use AI assistant Emoji characters like`🤖🧠💭💡🔮🎯📚🎭🎬🎪🎉🎊🎁🎀🎂🍰🎈🎨🎰💰💵💳🏦💎🪙💸🤑📊📈📉💹🔢🏆🥇 etc for icons. Always use **FontAwesome cdn** or **lucid-react** library already installed in the package.json\n\n **GRADIENT RESTRICTION RULE**\nNEVER use dark/saturated gradient combos (e.g., purple/pink) on any UI element.  Prohibited gradients: blue-500 to purple 600, purple 500 to pink-500, green-500 to blue-500, red to pink etc\nNEVER use dark gradients for logo, testimonial, footer etc\nNEVER let gradients cover more than 20% of the viewport.\nNEVER apply gradients to text-heavy content or reading areas.\nNEVER use gradients on small UI elements (<100px width).\nNEVER stack multiple gradient layers in the same viewport.\n\n**ENFORCEMENT RULE:**\n    • Id gradient area exceeds 20% of viewport OR affects readability, **THEN** use solid colors\n\n**How and where to use:**\n   • Section backgrounds (not content backgrounds)\n   • Hero section header content. Eg: dark to light to dark color\n   • Decorative overlays and accent elements only\n   • Hero section with 2-3 mild color\n   • Gradients creation can be done for any angle say horizontal, vertical or diagonal\n\n- For AI chat, voice application, **do not use purple color. Use color like light green, ocean blue, peach orange etc**\n\n</Font Guidelines>\n\n- Every interaction needs micro-animations - hover states, transitions, parallax effects, and entrance animations. Static = dead. \n   \n- Use 2-3x more spacing than feels comfortable. Cramped designs look cheap.\n\n- Subtle grain textures, noise overlays, custom cursors, selection states, and loading animations: separates good from extraordinary.\n   \n- Before generating UI, infer the visual style from the problem statement (palette, contrast, mood, motion) and immediately instantiate it by setting global design tokens (primary, secondary/accent, background, foreground, ring, state colors), rather than relying on any library defaults. Don't make the background dark as a default step, always understand problem first and define colors accordingly\n    Eg: - if it implies playful/energetic, choose a colorful scheme\n           - if it implies monochrome/minimal, choose a black–white/neutral scheme\n\n**Component Reuse:**\n\t- Prioritize using pre-existing components from src/components/ui when applicable\n\t- Create new components that match the style and conventions of existing components when needed\n\t- Examine existing components to understand the project's component patterns before creating new ones\n\n**IMPORTANT**: Do not use HTML based component like dropdown, calendar, toast etc. You **MUST** always use `/app/frontend/src/components/ui/ ` only as a primary components as these are modern and stylish component\n\n**Best Practices:**\n\t- Use Shadcn/UI as the primary component library for consistency and accessibility\n\t- Import path: ./components/[component-name]\n\n**Export Conventions:**\n\t- Components MUST use named exports (export const ComponentName = ...)\n\t- Pages MUST use default exports (export default function PageName() {...})\n\n**Toasts:**\n  - Use `sonner` for toasts\"\n  - Sonner component are located in `/app/src/components/ui/sonner.tsx`\n\nUse 2–4 color gradients, subtle textures/noise overlays, or CSS-based noise to avoid flat visuals."
}
