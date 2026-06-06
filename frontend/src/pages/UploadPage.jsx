import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import ResultCard from "../components/ResultCard";
import { predictVideo } from "../api/client";

function UploadPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [rawText, setRawText] = useState("—");
  const [correctedText, setCorrectedText] = useState("—");
  const [meta, setMeta] = useState("");
  const [alternatives, setAlternatives] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFileChange = (event) => {
    const next = event.target.files?.[0];
    setError("");
    setRawText("—");
    setCorrectedText("—");
    setMeta("");
    setAlternatives([]);
    if (!next) {
      setFile(null);
      return;
    }
    if (!next.type.startsWith("video/")) {
      setError("Lütfen geçerli bir video dosyası seçin.");
      setFile(null);
      return;
    }
    setFile(next);
  };

  const clearSelection = useCallback(() => {
    setFile(null);
    setPreviewUrl("");
    setRawText("—");
    setCorrectedText("—");
    setMeta("");
    setAlternatives([]);
    setError("");
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }, []);

  const handleAnalyze = async () => {
    if (!file) {
      setError("Önce cihazınızdan bir video seçin.");
      return;
    }
    setError("");
    setIsProcessing(true);
    setRawText("İşleniyor…");
    setCorrectedText("İşleniyor…");
    try {
      const data = await predictVideo(file);
      setRawText(data.raw_text ?? "—");
      setCorrectedText(data.corrected_text ?? "—");
      const conf = data.confidence != null ? `${Math.round(data.confidence * 100)}%` : "—";
      setMeta(`Güven: ${conf} · ${data.mode ?? "?"}`);
      setAlternatives(Array.isArray(data.alternatives) ? data.alternatives : []);
    } catch {
      setError("Sunucuya bağlanılamadı. API’nin çalıştığından emin olun.");
      setRawText("—");
      setCorrectedText("—");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen px-6 py-8 md:px-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <h1 className="text-2xl font-bold text-slate-100 md:text-3xl">Video ile analiz</h1>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/app"
              className="rounded-xl border border-green-600/50 bg-slate-900/60 px-4 py-2 text-sm font-medium text-green-400 transition hover:border-green-500 hover:text-green-300"
            >
              Canlı yayına geç
            </Link>
            <Link to="/" className="text-sm font-medium text-green-400 hover:text-green-300">
              Ana sayfa
            </Link>
          </div>
        </div>

        {error && (
          <div className="rounded-xl border border-red-700 bg-red-950/70 px-4 py-3 text-sm text-red-200">{error}</div>
        )}

        <section className="grid gap-5 lg:grid-cols-2">
          <div className="rounded-2xl bg-slate-900/75 p-5 shadow-soft ring-1 ring-slate-700 backdrop-blur">
            <label className="block text-sm font-medium text-slate-300">Cihazdan video seç</label>
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              onChange={onFileChange}
              className="mt-3 block w-full cursor-pointer text-sm text-slate-400 file:mr-4 file:cursor-pointer file:rounded-lg file:border-0 file:bg-green-700 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-green-600"
              disabled={isProcessing}
            />
            <p className="mt-2 text-xs text-slate-500">Desteklenen formatlar: tarayıcının tanıdığı video türleri (ör. MP4, WebM).</p>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!file || isProcessing}
                className="rounded-xl bg-green-700 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-green-600 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isProcessing ? "İşleniyor…" : "Videoyu analiz et"}
              </button>
              <button
                type="button"
                onClick={clearSelection}
                disabled={isProcessing || (!file && !previewUrl)}
                className="rounded-xl border border-slate-600 px-5 py-2.5 text-sm font-medium text-slate-300 transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Temizle
              </button>
            </div>

            <div className="relative mt-5 overflow-hidden rounded-xl bg-black ring-1 ring-slate-700">
              {isProcessing && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-slate-950/80 backdrop-blur-sm">
                  <div className="h-10 w-10 animate-spin rounded-full border-2 border-green-500 border-t-transparent" />
                  <span className="text-sm text-slate-200">Video yükleniyor ve işleniyor…</span>
                </div>
              )}
              {previewUrl ? (
                <video src={previewUrl} className="aspect-video w-full object-contain" controls playsInline />
              ) : (
                <div className="flex aspect-video items-center justify-center text-slate-500">
                  Ön izleme için video seçin
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-5">
            <ResultCard title="Algılanan işaret (ham)" content={rawText} subtitle={meta} />
            <ResultCard title="Düzeltilmiş cümle" content={correctedText} />
            {alternatives.length > 0 && (
              <div className="rounded-2xl bg-slate-900/75 p-5 ring-1 ring-slate-700">
                <h3 className="text-sm font-medium uppercase tracking-wide text-slate-400">Diğer olasılar</h3>
                <ul className="mt-2 space-y-1 text-sm text-slate-300">
                  {alternatives.map((a) => (
                    <li key={`${a.word}-${a.score}`}>
                      {a.word} <span className="text-slate-500">({Math.round(a.score * 100)}%)</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

export default UploadPage;
