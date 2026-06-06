import { useCallback, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import WebcamView from "../components/WebcamView";
import ResultCard from "../components/ResultCard";
import ControlPanel from "../components/ControlPanel";
import useAppStore from "../store/useAppStore";
import { predictLive } from "../api/client";

/** ~5 sn pencere: 250 ms'de bir kare, en fazla 20 kare */
const SAMPLE_MS = 250;
const ANALYZE_MS = 4500;
const BUFFER_MAX = 20;
const MIN_FRAMES = 14;

function AppPage() {
  const webcamRef = useRef(null);
  const bufferRef = useRef([]);
  const busyRef = useRef(false);
  const lastWordRef = useRef("");
  const stableHitsRef = useRef(0);

  const {
    isStreaming,
    rawText,
    correctedText,
    predictionMeta,
    isLoading,
    error,
    setStreaming,
    setPrediction,
    setPredictionMeta,
    setLoading,
    setError,
    resetTexts,
  } = useAppStore();

  const applyPrediction = useCallback(
    (data) => {
      const word = data.raw_text ?? "—";
      const conf = data.confidence ?? 0;

      if (word === lastWordRef.current) {
        stableHitsRef.current += 1;
      } else {
        lastWordRef.current = word;
        stableHitsRef.current = 1;
      }

      // Yuksek guven tek karede yanlis kelime verebiliyor; iki analiz ust uste ayni olmali
      if (stableHitsRef.current < 2) {
        return;
      }

      setPrediction({
        rawText: word,
        correctedText: data.corrected_text ?? "—",
      });
      const confPct = data.confidence != null ? `${Math.round(data.confidence * 100)}%` : "—";
      const modeLabel =
        data.mode === "live_gallery"
          ? "canlı · galeri"
          : data.mode === "live_classifier"
            ? "canlı · model"
            : data.mode ?? "?";
      setPredictionMeta(`Güven: ${confPct} · ${modeLabel}`);
    },
    [setPrediction, setPredictionMeta]
  );

  const toggleStream = () => {
    if (isStreaming) {
      setStreaming(false);
      setLoading(false);
      bufferRef.current = [];
      lastWordRef.current = "";
      stableHitsRef.current = 0;
      resetTexts();
      return;
    }
    setError("");
    setStreaming(true);
  };

  const handlePermissionError = useCallback(
    (message) => {
      setStreaming(false);
      setLoading(false);
      setError(message);
    },
    [setError, setLoading, setStreaming]
  );

  useEffect(() => {
    if (!isStreaming) {
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    bufferRef.current = [];

    const sampleId = setInterval(() => {
      const frame = webcamRef.current?.captureFrame?.();
      if (!frame) {
        return;
      }
      const buf = bufferRef.current;
      buf.push(frame);
      if (buf.length > BUFFER_MAX) {
        buf.shift();
      }
    }, SAMPLE_MS);

    const runAnalyze = async () => {
      if (busyRef.current || cancelled) {
        return;
      }
      const frames = bufferRef.current.slice(-BUFFER_MAX);
      if (frames.length < MIN_FRAMES) {
        return;
      }

      busyRef.current = true;
      try {
        const data = await predictLive(frames);
        if (!cancelled) {
          applyPrediction(data);
          setError("");
        }
      } catch {
        if (!cancelled) {
          setError("Sunucuya bağlanılamadı. API’nin çalıştığından emin olun.");
        }
      } finally {
        busyRef.current = false;
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    runAnalyze();
    const analyzeId = setInterval(runAnalyze, ANALYZE_MS);

    return () => {
      cancelled = true;
      clearInterval(sampleId);
      clearInterval(analyzeId);
    };
  }, [isStreaming, setError, setLoading, applyPrediction]);

  return (
    <main className="min-h-screen px-6 py-8 md:px-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <h1 className="text-2xl font-bold text-slate-100 md:text-3xl">Canlı işaret dili</h1>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/upload"
              className="rounded-xl border border-green-600/50 bg-slate-900/60 px-4 py-2 text-sm font-medium text-green-400 transition hover:border-green-500 hover:text-green-300"
            >
              Video yükle
            </Link>
            <Link to="/" className="text-sm font-medium text-green-400 hover:text-green-300">
              Ana sayfa
            </Link>
          </div>
        </div>

        <p className="text-sm text-slate-400">
          İşareti yaklaşık <strong className="text-slate-300">4–5 saniye</strong> sabit tutun; sistem bu sürede
          toplanan kareleri birlikte değerlendirir. Sonuç ~{ANALYZE_MS / 1000} sn’de bir güncellenir.
        </p>

        {error && (
          <div className="rounded-xl border border-red-700 bg-red-950/70 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        )}

        <section className="grid gap-5 lg:grid-cols-2">
          <WebcamView ref={webcamRef} isStreaming={isStreaming} onPermissionError={handlePermissionError} />
          <div className="flex flex-col gap-5">
            <ResultCard title="Algılanan işaret" content={rawText} subtitle={predictionMeta} />
            <ResultCard title="Düzeltilmiş cümle" content={correctedText} />
          </div>
        </section>

        <ControlPanel isStreaming={isStreaming} isLoading={isLoading} onToggle={toggleStream} />
      </div>
    </main>
  );
}

export default AppPage;
