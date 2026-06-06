import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
  timeout: 6000,
});

export async function predictFrame(frame = "") {
  const { data } = await api.post("/predict", { frame });
  return data;
}

/** Canli kamera: son birkaç kare (base64 JPEG dizisi). */
export async function predictLive(frames) {
  const { data } = await api.post("/predict-live", { frames }, { timeout: 90_000 });
  return data;
}

export async function predictVideo(file) {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post("/predict-video", formData, {
    timeout: 120_000,
  });
  return data;
}

export async function llmCorrect(text) {
  const { data } = await api.post("/llm-correct", { text });
  return data;
}

export async function checkHealth() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
