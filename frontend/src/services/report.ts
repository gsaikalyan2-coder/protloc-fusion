/* Pure helpers for the printable report (feature 10). */
import type { Result } from "./demoApi";

/** The sequence in blocks of ten, sixty residues a line, each line numbered. */
export function sequenceLines(seq: string): string[] {
  const lines: string[] = [];
  for (let i = 0; i < seq.length; i += 60) {
    const chunk = seq.slice(i, i + 60);
    const blocks = chunk.match(/.{1,10}/g) ?? [];
    lines.push(`${String(i + 1).padStart(5, " ")}  ${blocks.join(" ")}`);
  }
  return lines;
}

/** The residues with the largest positive attribution (they support the explained label). */
export function topResidues(
  r: Result,
  n = 8,
): { position: number; residue: string; score: number }[] {
  return r.attribution
    .map((score, i) => ({ position: i + 1, residue: r.sequence.charAt(i), score }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || a.position - b.position)
    .slice(0, n);
}
