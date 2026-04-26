import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "react-mosaic-component/react-mosaic-component.css";
import "./theme.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("helm-tauri: #root not found in index.html");
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
