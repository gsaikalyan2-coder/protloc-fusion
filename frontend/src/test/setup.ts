/* Vitest setup: unmount rendered components between tests (globals are off, so do it explicitly). */
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
