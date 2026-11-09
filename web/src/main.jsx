// main.jsx - the very first file React runs. It finds <div id="root"> in index.html
// and draws our <App /> inside it.
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(<App />);
