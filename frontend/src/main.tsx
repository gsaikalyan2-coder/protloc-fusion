import React from "react";
import { createRoot } from "react-dom/client";

import App from "@/App";
import "@/App.css";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing the #root element the application mounts into.");
}

createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
