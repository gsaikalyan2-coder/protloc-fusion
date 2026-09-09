import "@/styles/global.css";

/**
 * The application shell.
 *
 * Phase 1 builds structure, not behaviour. What renders here is the shell and
 * the design tokens proving they load — the routes, pages and result views
 * arrive in Phase 12, against the contract frozen in Phase 2. There is
 * deliberately no mock prediction: a fake response added "so the shape is
 * visible" is load-bearing by Phase 11 and nobody remembers it was fake.
 */
export default function App(): React.JSX.Element {
  return (
    <main className="shell">
      <h1 className="shell__title">ProtLoc Fusion</h1>
      <p className="shell__lede">
        One sequence and one organism group in; subcellular location, membrane type, sorting signal
        and cleavage site, GO function, homology evidence and a per-residue explanation out — from a
        single shared representation.
      </p>
      <p className="shell__status">
        Scaffold. The interface is built in Phase 12 against the contract frozen in Phase 2.
      </p>
    </main>
  );
}
