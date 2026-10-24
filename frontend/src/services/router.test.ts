import { describe, expect, it } from "vitest";
import { formatHash, parseHash } from "./router";

describe("router", () => {
  it("reads a screen and its parameters", () => {
    expect(parseHash("#/predict?acc=P12345")).toEqual({
      screen: "predict",
      params: { acc: "P12345" },
    });
  });

  it("treats an empty or unknown hash as Home", () => {
    expect(parseHash("").screen).toBe("home");
    expect(parseHash("#/").screen).toBe("home");
    expect(parseHash("#/inspector").screen).toBe("home");
  });

  it("writes Home as #/ and sorts parameters", () => {
    expect(formatHash("home")).toBe("#/");
    expect(formatHash("predict", { z: "1", acc: "Q9" })).toBe("#/predict?acc=Q9&z=1");
  });

  it("round-trips an encoded value", () => {
    const hash = formatHash("predict", { acc: "A B/C" });
    expect(parseHash(hash).params.acc).toBe("A B/C");
  });
});
