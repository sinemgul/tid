function ControlPanel({ isStreaming, isLoading, onToggle }) {
  return (
    <div className="rounded-2xl bg-slate-900/75 p-5 shadow-soft ring-1 ring-slate-700 backdrop-blur">
      <button
        type="button"
        onClick={onToggle}
        className={`w-full rounded-xl px-5 py-3 text-sm font-semibold text-white transition ${
          isStreaming
            ? "bg-red-600 hover:bg-red-500"
            : "bg-green-700 hover:bg-green-600"
        }`}
      >
        {isStreaming ? "Durdur" : "Baslat"}
      </button>

        {isLoading && (
        <div className="mt-4 flex items-center gap-3 text-sm text-slate-300">
          <span className="h-3 w-3 animate-pulseSoft rounded-full bg-green-500" />
          Veriler işleniyor…
        </div>
      )}
    </div>
  );
}

export default ControlPanel;
