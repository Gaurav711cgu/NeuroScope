import axios from "axios";

export const API = process.env.REACT_APP_API_URL || 
                   (process.env.REACT_APP_BACKEND_URL ? `${process.env.REACT_APP_BACKEND_URL}/api/v1` : null) || 
                   "http://localhost:8000/api/v1";

export const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 
                           API.replace("/api/v1", "") || 
                           "http://localhost:8000";

const client = axios.create({
    baseURL: API,
    timeout: 120000,
});

export const api = {
    health: () => client.get("/health").then((r) => r.data),
    suggestedTasks: () => client.get("/suggested-tasks").then((r) => r.data),
    listRuns: () => client.get("/runs").then((r) => r.data),
    createRun: (body) => client.post("/runs", body).then((r) => r.data),
    getRun: (id) => client.get(`/runs/${id}`).then((r) => r.data),
    getStep: (id, n) => client.get(`/runs/${id}/steps/${n}`).then((r) => r.data),
    patch: (id, body) =>
        client.post(`/runs/${id}/patch`, body).then((r) => r.data),
    patchMatrix: (id, body) =>
        client.post(`/runs/${id}/patch-matrix`, body).then((r) => r.data),
    listPatches: (id) => client.get(`/runs/${id}/patches`).then((r) => r.data),
    attribution: (id, body) =>
        client.post(`/runs/${id}/attribution`, body).then((r) => r.data),
    query: (id, body) =>
        client.post(`/runs/${id}/query`, body).then((r) => r.data),
    listQueries: (id) => client.get(`/runs/${id}/queries`).then((r) => r.data),
    listExperiments: () => client.get("/experiments").then((r) => r.data),
    getExperiment: (slug) =>
        client.get(`/experiments/${slug}`).then((r) => r.data),
    getFeature: (layer, fid) =>
        client.get(`/feature/${layer}/${fid}`).then((r) => r.data),
    steer: (body) => client.post("/steer", body).then((r) => r.data),
    trainProbe: (body) => client.post("/probe/train", body).then((r) => r.data),
};

export default api;
