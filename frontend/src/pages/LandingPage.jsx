import { Link } from "react-router-dom";

const features = [
  {
    title: "Gerçek zamanlı analiz",
    description: "Kamera akışını anlık işleyip hızlı çıktı sunar.",
  },
  {
    title: "Yapay zekâ destekli akış",
    description: "Ham metni daha anlaşılır Türkçeye dönüştüren süreç.",
  },
  {
    title: "Kolay kullanım",
    description: "Tek tıkla başlat, sonuçları anında görüntüle.",
  },
];

function LandingPage() {
  return (
    <main className="min-h-screen px-6 py-12 md:px-10">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="rounded-2xl bg-slate-900/70 p-8 shadow-soft ring-1 ring-slate-700 backdrop-blur">
          <p className="text-sm uppercase tracking-wider text-green-400">İşaret dili çözüm platformu</p>
          <h1 className="mt-3 text-3xl font-bold text-slate-100 md:text-5xl">
            İşaret dilini anında Türkçeye çevir
          </h1>
          <p className="mt-4 max-w-2xl text-slate-300">
            Kamera veya yüklediğiniz video ile işaret hareketlerinden metin oluşturun; düzeltilmiş ve
            anlaşılır Türkçe çıktıyı hemen görün.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              to="/app"
              className="inline-flex rounded-xl bg-green-700 px-6 py-3 font-semibold text-white transition hover:bg-green-600"
            >
              Canlı yayın
            </Link>
            <Link
              to="/upload"
              className="inline-flex rounded-xl border border-green-500/60 bg-slate-900/40 px-6 py-3 font-semibold text-green-300 transition hover:border-green-400 hover:bg-slate-900/60"
            >
              Video yükle
            </Link>
          </div>
        </header>

        <section className="grid gap-4 md:grid-cols-3">
          {features.map((feature) => (
            <article
              key={feature.title}
              className="rounded-2xl bg-slate-900/70 p-6 shadow-soft ring-1 ring-slate-700 transition hover:-translate-y-0.5 hover:ring-green-700/50"
            >
              <h2 className="text-xl font-semibold text-slate-100">{feature.title}</h2>
              <p className="mt-3 text-slate-300">{feature.description}</p>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}

export default LandingPage;
