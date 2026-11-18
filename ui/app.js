// ================== CONFIG ==================
// Default: Laptop 1's FastAPI server. On demo day either edit this line or
// open the page with ?api=http://<laptop-1-ip>:8000 (no code change).
// ?mock=1 uses ui/mock/*.json instead (offline fallback, no backend needed).
const params   = new URLSearchParams(window.location.search);
const API_BASE = (params.get("api") || "http://127.0.0.1:8000").replace(/\/+$/, "");
const USE_MOCK = params.get("mock") === "1";
// ============================================

const tripSelect = document.getElementById("tripSelect");
const predictBtn = document.getElementById("predictBtn");
const statusBox  = document.getElementById("status");

// ---------- Map setup ----------
const map = L.map("map").setView([20, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// One layer group per thing we draw, so we can clear them between trips
// and toggle them with the layer control.
const gpsLayer   = L.layerGroup().addTo(map);
const routeLayer = L.layerGroup().addTo(map);
L.control.layers(null, {
  "Raw GPS": gpsLayer,
  "Predicted route": routeLayer,
}, { collapsed: false }).addTo(map);

// ---------- Small helpers ----------
function setStatus(kind, text) {
  statusBox.className = kind;          // "loading" | "error" | "ok" | ""
  statusBox.textContent = text;
}

function fmt(x, digits = 3) {
  return (typeof x === "number" && isFinite(x)) ? x.toFixed(digits) : "–";
}

// Wraps fetch so network errors and non-200 replies both become readable messages.
async function getJSON(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (e) {
    throw new Error(`Cannot reach ${url}. Is the API running and on the same network?`);
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  return res.json();
}

// ---------- API calls (mock or real) ----------
async function fetchHealth() {
  if (USE_MOCK) return { status: "ok", mock: true };
  return getJSON(`${API_BASE}/health`);
}

async function fetchTrips() {
  const data = USE_MOCK
    ? await getJSON("mock/trips.json")
    : await getJSON(`${API_BASE}/trips`);
  // Accept either ["id1", ...] or {"trips": ["id1", ...]}
  return Array.isArray(data) ? data : data.trips;
}

async function fetchPrediction(tripId) {
  if (USE_MOCK) return getJSON("mock/predict.json");
  return getJSON(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trip_id: tripId }),
  });
}

// ---------- Rendering ----------
function drawResult(r) {
  gpsLayer.clearLayers();
  routeLayer.clearLayers();

  // gps and predicted_path are [[lat, lon], ...] as agreed in the API contract.
  r.gps.forEach(([lat, lon], i) => {
    L.circleMarker([lat, lon], {
      radius: 3, color: "#1f6feb", weight: 1, fillOpacity: 0.7,
    }).bindTooltip(`GPS #${i}`).addTo(gpsLayer);
  });

  if (r.predicted_path && r.predicted_path.length > 1) {
    L.polyline(r.predicted_path, { color: "#e5383b", weight: 5, opacity: 0.85 })
      .addTo(routeLayer);
  }

  // Start / end markers
  if (r.gps.length) {
    L.circleMarker(r.gps[0], { radius: 7, color: "#2b9348", fillOpacity: 1 })
      .bindTooltip("Start").addTo(gpsLayer);
    L.circleMarker(r.gps[r.gps.length - 1], { radius: 7, color: "#111", fillOpacity: 1 })
      .bindTooltip("End").addTo(gpsLayer);
    map.fitBounds(L.latLngBounds(r.gps), { padding: [30, 30] });
  }

  const m = r.metrics || {};
  document.getElementById("m_trip").textContent = r.trip_id ?? "–";
  document.getElementById("m_nobs").textContent = r.n_obs ?? r.gps.length;
  document.getElementById("m_top1").textContent = fmt(m.top1) === "–" ? "N/A" : fmt(m.top1);
  document.getElementById("m_f1").textContent = fmt(m.macro_f1) === "–" ? "N/A" : fmt(m.macro_f1);
  document.getElementById("m_cont").textContent = fmt(m.route_continuity) === "–" ? "N/A" : fmt(m.route_continuity);
  ocument.getElementById("m_seq").textContent = fmt(m.sequence_f1) === "–" ? "N/A" : fmt(m.sequence_f1);
  document.getElementById("m_lat").textContent  =
    (typeof r.latency_ms === "number") ? `${Math.round(r.latency_ms)} ms` : "–";

  // The API marks stub/passthrough answers with mock=true and a warning:
  // say so on screen instead of presenting them as model output.
  setMockBadge(USE_MOCK || r.mock === true);
  const warn = document.getElementById("warning");
  warn.textContent = r.warning || "";
  warn.style.display = r.warning ? "block" : "none";
}

function setMockBadge(on) {
  document.getElementById("mockBadge").style.display = on ? "inline" : "none";
}

// ---------- Event handlers ----------
predictBtn.addEventListener("click", async () => {
  const tripId = tripSelect.value;
  predictBtn.disabled = true;
  setStatus("loading", `Running pipeline for trip ${tripId}…`);
  try {
    const result = await fetchPrediction(tripId);
    drawResult(result);
    setStatus("ok", `Done: trip ${result.trip_id}`);
  } catch (e) {
    setStatus("error", e.message);
  } finally {
    predictBtn.disabled = false;
  }
});

async function init() {
  setMockBadge(USE_MOCK);
  document.getElementById("apiInfo").textContent = USE_MOCK ? "mock files" : API_BASE;
  try {
    const health = await fetchHealth();
    setMockBadge(USE_MOCK || health.mock === true);
    const trips = await fetchTrips();
    if (!trips || !trips.length) throw new Error("API returned no trips. Check EVAL_DATA_DIR on Laptop 1.");
    tripSelect.innerHTML = trips.map(t => `<option value="${t}">${t}</option>`).join("");
    predictBtn.disabled = false;
    setStatus("", "");
    // ?trip=<id> opens that trip straight away (handy for UI-03 screenshots).
    const wanted = params.get("trip");
    if (wanted && trips.includes(wanted)) {
      tripSelect.value = wanted;
      predictBtn.click();
    }
  } catch (e) {
    tripSelect.innerHTML = "<option>Could not load trips</option>";
    setStatus("error", e.message);
  }
}

init();
