import { describe, expect, it } from "vitest";
import { parseInput } from "./fasta";

describe("parseInput", () => {
  it("reads raw residues as one record without an id", () => {
    expect(parseInput("mktay iakq\nrqis")).toEqual({
      records: 1,
      sequence: "MKTAYIAKQRQIS",
      id: undefined,
    });
  });

  it("strips a single FASTA header and keeps its first token as the id", () => {
    expect(parseInput(">sp|P1|X desc\nMKT\nAYI\n")).toEqual({
      records: 1,
      sequence: "MKTAYI",
      id: "sp|P1|X",
    });
  });

  it("counts several records", () => {
    expect(parseInput(">a\nMK\n>b\nTA").records).toBe(2);
  });

  it("is empty for blank input", () => {
    expect(parseInput("  \n ").records).toBe(0);
  });
});
