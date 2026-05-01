/* ══════════════════════════════════════════
   AquaCheck AI — Client-Side Logic
   ══════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initScrollAnimations();
  initSliders();
  loadHistory();
});

/* ── NAVBAR ── */
function initNavbar() {
  const nav = document.querySelector(".navbar");
  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");

  if (toggle && links) {
    toggle.addEventListener("click", () => links.classList.toggle("open"));
    document.addEventListener("click", (e) => {
      if (!toggle.contains(e.target) && !links.contains(e.target)) links.classList.remove("open");
    });
  }
  window.addEventListener("scroll", () => {
    if (nav) nav.classList.toggle("scrolled", window.scrollY > 20);
  });
}

/* ── SCROLL ANIMATIONS ── */
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add("visible"); });
  }, { threshold: 0.1 });
  document.querySelectorAll(".fade-in").forEach((el) => observer.observe(el));
}

/* ── SLIDER SYNC ── */
function initSliders() {
  document.querySelectorAll(".param-card").forEach((card) => {
    const slider = card.querySelector(".param-slider");
    const number = card.querySelector(".param-number");
    if (!slider || !number) return;
    slider.addEventListener("input", () => { number.value = slider.value; });
    number.addEventListener("input", () => {
      let v = parseFloat(number.value);
      const min = parseFloat(slider.min), max = parseFloat(slider.max);
      if (v < min) v = min; if (v > max) v = max;
      slider.value = v;
    });
  });
}

/* ── PRESETS ── */
function loadPreset(type) {
  const presets = {
    safe: {"pH":7.8,"E.C":800,"TDS":500,"CO3":0,"HCO3":200,"Cl":80,"F":0.5,"NO3 ":20,"SO4":50,"Na":60,"K":3,"Ca":50,"Mg":30,"T.H":250,"SAR":1.5},
    unsafe: {"pH":9.1,"E.C":3790,"TDS":2426,"CO3":50,"HCO3":1000,"Cl":420,"F":2.3,"NO3 ":1.5,"SO4":109,"Na":800,"K":32,"Ca":16,"Mg":19,"T.H":120,"SAR":31.8}
  };
  const vals = presets[type];
  if (!vals) return;
  document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
  const btn = document.querySelector(`[data-preset="${type}"]`);
  if (btn) btn.classList.add("active");

  Object.entries(vals).forEach(([key, val]) => {
    const slider = document.querySelector(`input[type="range"][name="${key}"]`);
    const number = document.querySelector(`input[type="number"][name="${key}"]`);
    if (slider) slider.value = val;
    if (number) number.value = val;
  });
}

/* ── ANALYZE ── */
function analyzeWater() {
  const data = {};
  document.querySelectorAll(".param-number").forEach((input) => {
    data[input.name] = parseFloat(input.value) || 0;
  });

  showLoading();

  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
    .then((r) => r.json())
    .then((result) => {
      hideLoading();
      if (result.error) { alert("Error: " + result.error); return; }
      displayResults(result);
      saveHistory(result);
    })
    .catch((err) => {
      hideLoading();
      alert("Analysis failed. Make sure the model is trained (run python run_all.py).");
      console.error(err);
    });
}

/* ── LOADING ── */
function showLoading() {
  const el = document.getElementById("loadingOverlay");
  if (el) el.classList.add("active");
}
function hideLoading() {
  const el = document.getElementById("loadingOverlay");
  if (el) el.classList.remove("active");
}

/* ── DISPLAY RESULTS ── */
function displayResults(r) {
  const section = document.getElementById("resultsSection");
  if (!section) return;

  const isSafe = r.prediction === 1;
  const pct = (r.confidence * 100).toFixed(1);
  const suitPct = (r.probabilities.suitable * 100).toFixed(1);
  const unsuitPct = (r.probabilities.not_suitable * 100).toFixed(1);

  // Status card
  const statusCard = document.getElementById("statusCard");
  statusCard.className = `result-status-card ${isSafe ? "safe" : "unsafe"}`;
  statusCard.innerHTML = `
    <div class="result-icon">${isSafe ? "✅" : "❌"}</div>
    <div class="result-label">${r.label}</div>
    <div class="result-conf-text">Confidence: ${pct}%</div>
  `;

  // Confidence ring
  const circumference = 2 * Math.PI * 54; // r=54
  const ring = document.getElementById("ringFill");
  ring.style.strokeDasharray = `${(circumference * r.confidence)} ${circumference}`;
  ring.className = `ring-fill ${r.confidence > 0.8 ? (isSafe ? "safe" : "unsafe") : "moderate"}`;
  document.getElementById("confPct").textContent = pct + "%";

  // Probability bars
  document.getElementById("safeFill").style.width = suitPct + "%";
  document.getElementById("unsafeFill").style.width = unsuitPct + "%";
  document.getElementById("safePct").textContent = suitPct + "%";
  document.getElementById("unsafePct").textContent = unsuitPct + "%";

  // Factors
  const factorsEl = document.getElementById("factorsList");
  if (r.top_factors && r.top_factors.length) {
    factorsEl.innerHTML = r.top_factors.map((f) => `
      <div class="factor-item">
        <span class="factor-dir">${f.shap_value > 0 ? "🔺" : "🔻"}</span>
        <span class="factor-name">${f.feature}</span>
        <span class="factor-val">${f.value.toFixed(2)}</span>
        <span class="factor-shap ${f.shap_value > 0 ? "positive" : "negative"}">
          SHAP: ${f.shap_value > 0 ? "+" : ""}${f.shap_value.toFixed(3)}
        </span>
      </div>
    `).join("");
  } else {
    factorsEl.innerHTML = '<p style="color:var(--text-muted)">SHAP analysis unavailable.</p>';
  }

  // Explanation
  document.getElementById("explanationText").textContent = r.explanation;

  // SHAP chart
  const shapCard = document.getElementById("shapChart");
  if (r.shap_plot) {
    shapCard.innerHTML = `<h3>📊 SHAP Waterfall Chart</h3><img src="data:image/png;base64,${r.shap_plot}" alt="SHAP Waterfall">`;
    shapCard.style.display = "block";
  } else {
    shapCard.style.display = "none";
  }

  section.classList.add("active");
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── HISTORY (LocalStorage) ── */
function saveHistory(result) {
  const history = JSON.parse(localStorage.getItem("aquacheck_history") || "[]");
  history.unshift({
    label: result.label,
    prediction: result.prediction,
    confidence: result.confidence,
    timestamp: result.timestamp,
    inputs: result.input_values
  });
  if (history.length > 10) history.pop();
  localStorage.setItem("aquacheck_history", JSON.stringify(history));
  loadHistory();
}

function loadHistory() {
  const container = document.getElementById("historyGrid");
  if (!container) return;
  const history = JSON.parse(localStorage.getItem("aquacheck_history") || "[]");

  if (!history.length) {
    container.innerHTML = '<div class="history-empty"><p>💧 No analyses yet. Run your first water quality check!</p></div>';
    return;
  }

  container.innerHTML = history.map((h, i) => `
    <div class="glass-card history-card ${h.prediction === 1 ? "was-safe" : "was-unsafe"}" onclick="restoreHistory(${i})">
      <div class="history-meta">
        <span class="history-result" style="color:${h.prediction === 1 ? "#059669" : "#dc2626"}">
          ${h.prediction === 1 ? "✅" : "❌"} ${h.label}
        </span>
        <span class="history-time">${h.timestamp}</span>
      </div>
      <div class="history-conf">Confidence: ${(h.confidence * 100).toFixed(1)}%</div>
    </div>
  `).join("");
}

function restoreHistory(index) {
  const history = JSON.parse(localStorage.getItem("aquacheck_history") || "[]");
  const h = history[index];
  if (!h || !h.inputs) return;
  Object.entries(h.inputs).forEach(([key, val]) => {
    const slider = document.querySelector(`input[type="range"][name="${key}"]`);
    const number = document.querySelector(`input[type="number"][name="${key}"]`);
    if (slider) slider.value = val;
    if (number) number.value = val;
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function clearHistory() {
  localStorage.removeItem("aquacheck_history");
  loadHistory();
}
