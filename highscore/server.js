#!/usr/bin/env node
// Highscore server (Express version)
// - Local only (127.0.0.1)
// - Endpoints: /submit, /state, /register, /dismiss
// - Persists to highscores.json
// - Keeps highscores sorted (fastest first) and trimmed to top 10
// - Stores { name, phone, time, date } on register

const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");

const HOST = "127.0.0.1";
const PORT = 3000;

const FILE = path.join(__dirname, "highscores.json");
const PUBLIC_DIR = path.join(__dirname, "public");

// -------------------- Load state --------------------
let state = { highscores: [], pending: [] };

try {
  if (fs.existsSync(FILE)) {
    const raw = fs.readFileSync(FILE, "utf8");
    const parsed = JSON.parse(raw || "{}");
    state.highscores = Array.isArray(parsed.highscores) ? parsed.highscores : [];
    state.pending = Array.isArray(parsed.pending) ? parsed.pending : [];
  } else {
    persist(); // create empty file on first run
  }
} catch (e) {
  console.warn("Kunne ikke lese highscores.json, starter med tom state:", e.message);
  state = { highscores: [], pending: [] };
  persist();
}

keepTop10();

// -------------------- App setup --------------------
const app = express();

app.use(bodyParser.json());

// No-store everywhere (avoid caching during dev and polling)
app.use((req, res, next) => {
  res.set("Cache-Control", "no-store");
  next();
});

// CSP: allow same-origin scripts, styles, fetch/XHR, images
app.use((req, res, next) => {
  res.set(
    "Content-Security-Policy",
    [
      "default-src 'self'",
      "connect-src 'self'",        // allow fetch/XHR to /state, /register, etc.
      "script-src 'self'",
      "style-src 'self' 'unsafe-inline'", // inline styles in your <style> are OK
      "img-src 'self' data:"
      // add more if you ever load fonts/images from elsewhere
    ].join("; ")
  );
  next();
});


// Serve static frontend
app.use(express.static(PUBLIC_DIR));

// -------------------- Helpers --------------------
function persist() {
  try {
    fs.writeFileSync(FILE, JSON.stringify(state, null, 2), "utf8");
  } catch (e) {
    console.error("Kunne ikke skrive highscores.json:", e);
  }
}

function keepTop10() {
  state.highscores.sort((a, b) => a.time - b.time);
  state.highscores = state.highscores.slice(0, 10);
}

function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function isNonEmptyString(v) {
  return typeof v === "string" && v.trim().length > 0;
}

// -------------------- API --------------------

// POST /submit  { time: <number seconds> }
app.post("/submit", (req, res) => {
  const { time } = req.body || {};
  if (typeof time !== "number" || !isFinite(time) || time < 0) {
    return res.status(400).json({ error: "Invalid time (must be non-negative number)" });
  }

  const id = newId();
  state.pending.push({ id, time: Math.round(time) }); // store as integer seconds
  persist();
  res.json({ ok: true, id });
});

// GET /state
app.get("/state", (req, res) => {
  keepTop10();
  res.json(state);
});

// POST /register  { id, name, phone }
app.post("/register", (req, res) => {
  const { id, name, phone } = req.body || {};
  const idx = state.pending.findIndex((p) => p.id === String(id));

  if (idx === -1) {
    return res.status(404).json({ error: "Pending score not found" });
  }

  // Both required (as requested)
  if (!isNonEmptyString(name) || !isNonEmptyString(phone)) {
    return res.status(400).json({ error: "Name and phone are required" });
  }

  state.highscores.push({
    name: String(name).trim(),
    phone: String(phone).trim(), // stored but not shown in table
    time: state.pending[idx].time, // integer seconds
    date: new Date().toISOString(),
  });

  state.pending.splice(idx, 1);
  keepTop10();
  persist();
  res.json({ ok: true });
});

// POST /dismiss  { id }
app.post("/dismiss", (req, res) => {
  const { id } = req.body || {};
  const before = state.pending.length;
  state.pending = state.pending.filter((p) => p.id !== String(id));
  if (state.pending.length !== before) {
    persist();
  }
  res.json({ ok: true });
});

// (Optional) simple health endpoint
app.get("/health", (_req, res) => res.json({ ok: true }));

// -------------------- Start --------------------
app.listen(PORT, HOST, () => {
  console.log(`Highscore server running at http://${HOST}:${PORT}`);
});
