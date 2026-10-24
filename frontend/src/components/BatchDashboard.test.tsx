import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { makeResult } from "../test/fixtures";
import { BatchDashboard } from "./BatchDashboard";

describe("BatchDashboard", () => {
  it("draws each distribution with its counts in text", () => {
    render(
      <BatchDashboard
        results={[makeResult({ id: "a" }), makeResult({ id: "b", result_cache: "hit" })]}
        refused={1}
      />,
    );
    expect(screen.getByText(/2 scored · 1 refused · 1 from cache/)).toBeTruthy();
    const locations = screen.getByRole("figure", { name: /Predicted locations/ });
    expect(within(locations).getByText("Cytoplasm")).toBeTruthy();
    expect(within(locations).getByText("(100%)")).toBeTruthy();
    const go = screen.getByRole("figure", { name: /GO terms/ });
    expect(within(go).getByText("catalytic activity")).toBeTruthy();
    expect(screen.getByRole("figure", { name: /Sorting signal/ })).toBeTruthy();
    expect(screen.getByRole("figure", { name: /Homology evidence/ })).toBeTruthy();
  });

  it("says so when nothing cleared a threshold", () => {
    const quiet = makeResult({
      no_confident_call: true,
      localization: makeResult().localization.map((l) => ({ ...l, predicted: false })),
      go_function: [],
    });
    render(<BatchDashboard results={[quiet]} />);
    expect(screen.getByText("No location cleared its threshold.")).toBeTruthy();
    expect(screen.getByText("No GO term cleared its threshold.")).toBeTruthy();
    expect(screen.getByText(/1 no confident call/)).toBeTruthy();
  });
});
