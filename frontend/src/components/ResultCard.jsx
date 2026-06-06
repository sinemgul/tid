function ResultCard({ title, content, subtitle }) {
  return (
    <div className="rounded-2xl bg-slate-900/75 p-5 shadow-soft ring-1 ring-slate-700 backdrop-blur transition hover:ring-green-700/50">
      <h3 className="text-sm font-medium uppercase tracking-wide text-slate-400">{title}</h3>
      {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
      <p className="mt-3 min-h-16 text-lg leading-relaxed text-slate-100">{content}</p>
    </div>
  );
}

export default ResultCard;
