import { create } from "zustand";

const useAppStore = create((set) => ({
  isStreaming: false,
  rawText: "Sistem başlatılmadı.",
  correctedText: "Başlattığınızda düzeltilmiş metin burada görünecek.",
  predictionMeta: "",
  isLoading: false,
  error: "",
  setStreaming: (isStreaming) => set({ isStreaming }),
  setPrediction: ({ rawText, correctedText }) => set({ rawText, correctedText }),
  setPredictionMeta: (predictionMeta) => set({ predictionMeta }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  resetTexts: () =>
    set({
      rawText: "Sistem başlatılmadı.",
      correctedText: "Başlattığınızda düzeltilmiş metin burada görünecek.",
      predictionMeta: "",
    }),
}));

export default useAppStore;
