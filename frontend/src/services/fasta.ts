/* Reading what a person pasted: raw residues or FASTA, one record or several. */

/** One record: its residues and, if it had a header, the header's first token. */
export interface ParsedInput {
  records: number;
  sequence: string;
  id: string | undefined;
}

/** Count FASTA records and, for one record, strip the header and whitespace. */
export function parseInput(text: string): ParsedInput {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.filter((l) => l.startsWith(">"));
  const records = headers.length > 0 ? headers.length : text.trim() ? 1 : 0;
  const sequence = lines
    .filter((l) => !l.startsWith(">"))
    .join("")
    .replace(/\s+/g, "")
    .toUpperCase();
  const token = headers[0]?.slice(1).trim().split(/\s+/)[0];
  const id = token === "" ? undefined : token;
  return { records, sequence, id };
}
