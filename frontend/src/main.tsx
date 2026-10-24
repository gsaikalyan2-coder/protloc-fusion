import React from "react";
import { createRoot } from "react-dom/client";

// IBM Plex, self-hosted (docs/DESIGN.md §1.2, ADR-0020): Latin subset, the two weights the
// design system allows. Nothing is requested from a font CDN.
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";

import App from "@/App";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing the #root element the application mounts into.");
}

createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
