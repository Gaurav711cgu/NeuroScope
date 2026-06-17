import { useEffect } from "react";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import "@/App.css";

import TopBar from "@/components/TopBar";
import Landing from "@/pages/Landing";
import RunCreate from "@/pages/RunCreate";
import RunAnalysis from "@/pages/RunAnalysis";
import StepDetail from "@/pages/StepDetail";
import ExperimentsLibrary from "@/pages/ExperimentsLibrary";
import ExperimentDetail from "@/pages/ExperimentDetail";
import Docs from "@/pages/Docs";
import Findings from "@/pages/Findings";
import NeuroShield from "@/pages/NeuroShield";

function ScrollToTop() {
    const { pathname } = useLocation();
    useEffect(() => {
        window.scrollTo(0, 0);
    }, [pathname]);
    return null;
}

function Shell() {
    return (
        <div className="App">
            <TopBar />
            <ScrollToTop />
            <Routes>
                <Route path="/" element={<Landing />} />
                <Route path="/run" element={<RunCreate />} />
                <Route path="/run/:id" element={<RunAnalysis />} />
                <Route path="/run/:id/step/:n" element={<StepDetail />} />
                <Route path="/experiments" element={<ExperimentsLibrary />} />
                <Route
                    path="/experiments/:slug"
                    element={<ExperimentDetail />}
                />
                <Route path="/docs" element={<Docs />} />
                <Route path="/findings" element={<Findings />} />
                <Route path="/shield" element={<NeuroShield />} />
            </Routes>
            <Toaster
                theme="dark"
                position="bottom-right"
                toastOptions={{
                    style: {
                        background: "var(--ns-bg-surface-2)",
                        border: "1px solid var(--ns-border)",
                        color: "var(--ns-fg-primary)",
                        fontFamily: "JetBrains Mono, monospace",
                        fontSize: "12px",
                    },
                }}
            />
        </div>
    );
}

export default function App() {
    return (
        <BrowserRouter>
            <Shell />
        </BrowserRouter>
    );
}
