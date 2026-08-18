const TEAM = {
  ferrari: "#ed4655", mclaren: "#ff8a3d", mercedes: "#36bfb1", red_bull: "#4778d2",
  alpine: "#e99bd0", aston_martin: "#2c8c78", haas: "#9aa1aa", rb: "#5b7fe8",
  racing_bulls: "#5b7fe8", williams: "#68a8ef", sauber: "#71bd45", audi: "#ef3d45"
};

const ICON = {
  calendar: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h14v16H5zM8 2v4m8-4v4M5 9h14"/></svg>',
  pin: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 21s6-5.2 6-11a6 6 0 1 0-12 0c0 5.8 6 11 6 11z"/><circle cx="12" cy="10" r="2"/></svg>',
  pole: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 21V4m0 1h11l-3 3 3 3H6"/></svg>',
  trophy: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4h8v4c0 4-2 6-4 6s-4-2-4-6V4zm0 2H4v2c0 2 1.5 4 4 4m8-6h4v2c0 2-1.5 4-4 4m-4 2v4m-4 2h8"/></svg>',
  podium: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 20v-6h6v6m0 0V8h6v12m0 0v-9h6v9M3 20h18"/></svg>'
};

const state = { pre: null, post: null, record: null, bt: null, modelView: "pre", backtestView: "post", sort: { key: "top1_accuracy", direction: "desc" } };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = value => String(value ?? "").replace(/[&<>'"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[ch]));
const title = id => String(id ?? "").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
const pct = (value, digits = 1) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
const fixed = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const teamColour = team => TEAM[String(team ?? "").toLowerCase()] || "#ed4655";
const fetchJSON = async path => { try { const response = await fetch(path, { cache: "no-store" }); return response.ok ? response.json() : null; } catch { return null; } };

function setTab(name, focus = false) {
  const chosen = $(`[data-tab="${name}"]`) || $("[data-tab]");
  $$('[role="tab"]').forEach(tab => {
    const active = tab === chosen;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  $$('.tab-panel').forEach(panel => { const active = panel.id === chosen.dataset.tab; panel.hidden = !active; panel.classList.toggle("is-active", active); });
  history.replaceState(null, "", `#${chosen.dataset.tab}`);
  if (focus) $(`#${chosen.getAttribute("aria-controls")}`).focus({ preventScroll: true });
}

function setupTabs() {
  const tabs = $$('[role="tab"]');
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
    tab.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus(); setTab(tabs[next].dataset.tab);
    });
  });
  const hash = location.hash.slice(1); if (["weekend", "models", "numbers"].includes(hash)) setTab(hash);
}

function pickModel(payload, modelName, key = "predictions") {
  const models = payload?.[key] || [];
  return models.find(model => model.model === modelName) || models[0] || null;
}

function renderRace() {
  const payload = state.post || state.pre;
  if (!payload?.race) {
    $("#race-title").textContent = "No upcoming race found";
    $("#race-meta").innerHTML = '<span class="meta-chip">The calendar will check again on the next refresh.</span>';
    $("#header-status").textContent = "Waiting for the calendar";
    return;
  }
  const race = payload.race;
  $("#race-title").textContent = race.name;
  $("#race-round").textContent = `Round ${race.round} · ${race.season}`;
  $("#header-status").textContent = `${race.name} forecast is live`;
  const date = new Date(`${race.date}T14:00:00Z`);
  const formatted = new Intl.DateTimeFormat(undefined, { weekday: "short", month: "long", day: "numeric" }).format(date);
  $("#race-meta").innerHTML = `<span class="meta-chip">${ICON.calendar}${esc(formatted)}</span><span class="meta-chip">${ICON.pin}${esc(title(race.circuit))}</span><span class="meta-chip">Published before race day</span>`;
  const updateCountdown = () => {
    const remaining = date - new Date();
    if (remaining <= 0) { $("#countdown").textContent = "Race day"; return; }
    const days = Math.floor(remaining / 864e5), hours = Math.floor(remaining / 36e5) % 24, minutes = Math.floor(remaining / 6e4) % 60;
    $("#countdown").textContent = `${days}d ${hours}h ${minutes}m`;
  };
  updateCountdown(); setInterval(updateCountdown, 60000);
}

function pickCard({ label, driver, colour, icon, className = "", detail = "" }) {
  if (!driver) return `<article class="pick-card ${className}" style="--card-color:${colour}"><p class="empty-state">Prediction not available yet.</p></article>`;
  return `<article class="pick-card ${className}" style="--card-color:${colour}">
    <div class="pick-kicker"><span>${esc(label)}</span><span class="target-icon">${icon}</span></div>
    <div class="driver-name">${esc(title(driver.driver_id))}</div>
    <div class="driver-team"><span class="team-dot" style="--team:${teamColour(driver.constructor_id)}"></span>${esc(title(driver.constructor_id))}</div>
    ${detail}
  </article>`;
}

function predictionMetrics(items) {
  return `<div class="prediction-metrics">${items.map(([label, value, note = ""]) => `<div class="prediction-metric"><span>${esc(label)}</span><strong>${esc(value)}</strong>${note ? `<small>${esc(note)}</small>` : ""}</div>`).join("")}</div>`;
}

function renderHouse() {
  const source = state.post || state.pre, house = state.bt?.house || {};
  if (!source) { $("#house-grid").innerHTML = '<p class="empty-state">No forecast file is available yet.</p>'; return; }
  const postQuali = Boolean(state.post);
  const winnerSpec = postQuali ? house.winner : (house.winner_pre_quali || house.winner);
  const winnerModel = pickModel(source, winnerSpec?.model);
  const podiumModel = pickModel(source, house.podium?.model || winnerSpec?.model);
  const qualifyingModel = pickModel(state.pre, null, "qualifying_predictions");
  const winner = winnerModel?.ranking?.[0];
  const pole = qualifyingModel?.ranking?.[0];
  const podium = podiumModel?.ranking?.slice(0, 3) || [];
  const podiumRows = podium.map((driver, index) => `<div class="podium-row"><strong>P${index + 1}</strong><strong>${esc(title(driver.driver_id))}</strong><span><i class="team-dot" style="--team:${teamColour(driver.constructor_id)}"></i>${esc(title(driver.constructor_id))}</span></div>`).join("");
  const poleConfidence = pole?.pole_probability ?? pole?.win_probability;
  const winnerConfidence = winner?.win_probability;
  const sampleSize = winnerSpec?.n_races || state.bt?.n_races || 103;
  $("#house-grid").innerHTML = [
    pickCard({ label: "Pole position", driver: pole, colour: "var(--lavender)", icon: ICON.pole, className: "pole-card", detail: predictionMetrics([["Model confidence", pct(poleConfidence), "this pole call"], ["Forecast stage", "Pre-quali", "before the grid"]]) }),
    pickCard({ label: "Race winner", driver: winner, colour: "var(--peach)", icon: ICON.trophy, className: "winner-card", detail: predictionMetrics([["Model confidence", pct(winnerConfidence), "this race"], ["Backtest accuracy", pct(winnerSpec?.value), `${sampleSize} races`]]) }),
    `<article class="pick-card podium-card" style="--card-color:var(--mint)"><div class="pick-kicker"><span>Predicted podium</span><span class="target-icon">${ICON.podium}</span></div><div class="podium-stack">${podiumRows || '<p class="empty-state">Podium unavailable.</p>'}</div>${predictionMetrics([["Podium hit rate", pct(house.podium?.value), "walk-forward"], ["Evidence", `${house.podium?.n_races || sampleSize} races`, "held out"]])}</article>`
  ].join("");
  $("#house-grid").setAttribute("aria-busy", "false");

  const score = winnerSpec?.value;
  $("#score-dial").style.setProperty("--score", score || 0);
  $("#score-value").textContent = pct(score);
  $("#score-label").textContent = postQuali ? "Post-qualifying hit rate" : "Pre-qualifying hit rate";
  $("#score-note").textContent = `${winnerSpec?.n_races || state.bt?.n_races || 103} held-out races · ${(winnerSpec?.lo != null) ? `${pct(winnerSpec.lo,0)}–${pct(winnerSpec.hi,0)} interval` : "walk-forward tested"}`;
}

function renderWeather() {
  const weather = (state.post || state.pre)?.weather || {};
  const wet = Number(weather.weather_is_wet) > 0;
  const items = [
    ["Air", `${fixed(weather.weather_temp_max, 1)}°C`],
    ["Rain", `${fixed(weather.weather_precipitation, 1)} mm`],
    ["Wind", `${fixed(weather.weather_windspeed_max, 0)} km/h`],
    ["Read", wet ? "Wet expected" : "Dry expected"]
  ];
  $("#weather-stats").innerHTML = items.map(([label, value]) => `<div class="weather-stat"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
}

function renderLastResult() {
  const settled = state.record?.settled || [];
  if (!settled.length) return;
  const latest = settled.at(-1), correct = latest.top1_correct === 1;
  $("#result-stamp").textContent = correct ? "Nailed it" : "Missed it";
  $("#last-result").classList.remove("empty-state");
  $("#last-result").innerHTML = `<div class="result-line"><span class="result-mark">${correct ? "✓" : "×"}</span><div><strong>${esc(latest.race_name || `Round ${latest.round}`)}</strong><p>Picked ${esc(title(latest.predicted_winner))}; ${esc(title(latest.actual_winner))} won.</p></div></div>`;
}

function tower(model) {
  const ranking = model?.ranking?.slice(0, 6) || [];
  const max = Math.max(...ranking.map(item => Number(item.win_probability) || 0), .001);
  return ranking.map(item => `<div class="tower-row"><span class="tower-pos">P${item.position}</span><strong>${esc(title(item.driver_id))}</strong><span class="tower-bar"><span style="--w:${((Number(item.win_probability) || 0) / max * 100).toFixed(1)}%;--team:${teamColour(item.constructor_id)}"></span></span><strong>${pct(item.win_probability)}</strong></div>`).join("");
}

function modelCard(model, kind = "Race") {
  const first = model?.ranking?.[0];
  return `<article class="model-card"><header><div><p class="eyebrow">${esc(kind)} call</p><h2>${esc(model.model)}</h2></div><span class="model-chip">${esc(model.view || "forecast")}</span></header><div class="model-pick">${esc(title(first?.driver_id || "—"))}</div><div class="driver-team"><span class="team-dot" style="--team:${teamColour(first?.constructor_id)}"></span>${esc(title(first?.constructor_id || "No team"))}</div><div class="tower">${tower(model)}</div></article>`;
}

function renderModels() {
  const payload = state.modelView === "post" ? state.post : state.pre;
  const grid = $("#model-grid");
  if (!payload?.predictions?.length) grid.innerHTML = `<article class="model-card"><h2>${state.modelView === "post" ? "Waiting for qualifying" : "No forecast available"}</h2><p class="empty-state">${state.modelView === "post" ? "The Saturday refresh will add these calls once the real grid is known." : "The next scheduled refresh will try again."}</p></article>`;
  else grid.innerHTML = payload.predictions.map(model => modelCard(model)).join("");
  const qualifying = state.pre?.qualifying_predictions || [];
  $("#quali-model-grid").innerHTML = qualifying.length ? qualifying.map(model => modelCard(model, "Pole")).join("") : '<p class="empty-state">No qualifying predictions yet.</p>';
  $("#model-view-note").textContent = state.modelView === "post" ? "Uses the real starting grid." : "The grid is not known yet.";
}

function setupModelToggle() {
  $$('[data-model-view]').forEach(button => button.addEventListener("click", () => {
    state.modelView = button.dataset.modelView;
    $$('[data-model-view]').forEach(other => { const active = other === button; other.classList.toggle("is-active", active); other.setAttribute("aria-pressed", String(active)); });
    renderModels();
  }));
}

function renderTable() {
  const rows = [...(state.backtestView === "post" ? state.bt?.post_quali || [] : state.bt?.pre_quali || [])];
  const { key, direction } = state.sort;
  rows.sort((a, b) => {
    const av = a[key], bv = b[key], comparison = typeof av === "string" ? String(av).localeCompare(String(bv)) : (Number(av) || 0) - (Number(bv) || 0);
    return direction === "asc" ? comparison : -comparison;
  });
  const houseModel = (state.backtestView === "post" ? state.bt?.house?.winner : state.bt?.house?.winner_pre_quali)?.model;
  $("#results-body").innerHTML = rows.length ? rows.map(row => `<tr class="${row.model === houseModel ? "is-house" : row.model.startsWith("naive:") ? "is-baseline" : ""}"><td>${esc(row.model)}</td><td><strong>${pct(row.top1_accuracy)}</strong><span class="ci">${pct(row.top1_accuracy_lo,0)}–${pct(row.top1_accuracy_hi,0)}</span></td><td>${pct(row.podium_hit_rate)}</td><td>${fixed(row.spearman_rho,3)}</td><td>${fixed(row.winner_log_loss,3)}</td><td>${esc(row.n_races)}</td></tr>`).join("") : '<tr><td colspan="6">No results for this view yet.</td></tr>';
  const baseline = state.bt?.baselines?.winner?.value;
  if (baseline != null) $("#baseline-headline").textContent = `The pole-sitter rule leads at ${pct(baseline)}.`;
}

function setupNumbers() {
  $$('[data-backtest-view]').forEach(button => button.addEventListener("click", () => {
    state.backtestView = button.dataset.backtestView;
    $$('[data-backtest-view]').forEach(other => { const active = other === button; other.classList.toggle("is-active", active); other.setAttribute("aria-pressed", String(active)); });
    renderTable();
  }));
  $$('[data-sort]').forEach(button => button.addEventListener("click", () => {
    const key = button.dataset.sort;
    state.sort.direction = state.sort.key === key && state.sort.direction === "desc" ? "asc" : "desc";
    state.sort.key = key;
    $$("#results-table th").forEach(th => th.removeAttribute("aria-sort"));
    button.closest("th").setAttribute("aria-sort", state.sort.direction === "asc" ? "ascending" : "descending");
    renderTable();
  }));
}

async function setupGarage() {
  const layer = $("#drive-layer");
  try {
    const { mountSiteCar } = await import("./car.js?v=6");
    const toggle = $("[data-car-toggle]");
    const toggleLabel = $("[data-car-toggle-label]");
    const hint = $("#garage-hint");
    const car = mountSiteCar(layer, {
      onStateChange({ driving }) {
        toggle.classList.toggle("is-active", driving);
        toggle.setAttribute("aria-pressed", String(driving));
        toggleLabel.textContent = driving ? "Driving armed" : "Car parked";
        hint.innerHTML = driving ? '<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd><span>or arrows to drive · <kbd>Esc</kbd> parks</span>' : '<span>Parked safely · select “Car parked” to re-arm</span>';
      }
    });
    toggle.addEventListener("click", () => car.toggle());
    $("[data-car-reset]").addEventListener("click", () => car.reset());
  } catch (error) {
    console.warn("Interactive site car unavailable", error);
    $("#garage-hint").textContent = "The car stayed in the truck. Predictions still work.";
  }
}

async function load() {
  setupTabs(); setupModelToggle(); setupNumbers(); setupGarage();
  const confidenceButton = $("#explain-confidence");
  confidenceButton.addEventListener("click", () => { const open = confidenceButton.getAttribute("aria-expanded") !== "true"; confidenceButton.setAttribute("aria-expanded", String(open)); $("#confidence-note").hidden = !open; });
  [state.pre, state.post, state.record, state.bt] = await Promise.all([
    fetchJSON("./data/latest_pre_quali.json"), fetchJSON("./data/latest_post_quali.json"), fetchJSON("./data/track_record.json"), fetchJSON("./data/backtest.json")
  ]);
  renderRace(); renderHouse(); renderWeather(); renderLastResult(); renderModels(); renderTable();
  const stamp = (state.post || state.pre)?.generated_at || state.record?.updated_at;
  $("#last-updated").textContent = stamp ? `Updated ${new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(stamp))}` : "Waiting for the latest forecast…";
}

load();
