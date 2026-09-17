import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Dashboard } from "./Dashboard";
import "./admin.css";

const root = document.getElementById("root");
if (!root) throw new Error("no #root element");

createRoot(root).render(
  <StrictMode>
    <Dashboard />
  </StrictMode>,
);
