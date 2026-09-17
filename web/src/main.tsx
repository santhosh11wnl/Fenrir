/**
 * Mount point.
 *
 * Creates its own container rather than requiring one in the host page, so
 * embedding is a single script tag and the widget cannot collide with an
 * existing element id.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

const CONTAINER_ID = "mcp-chat-widget";

let container = document.getElementById(CONTAINER_ID);
if (!container) {
  container = document.createElement("div");
  container.id = CONTAINER_ID;
  document.body.appendChild(container);
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
