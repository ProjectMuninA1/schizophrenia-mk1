const $ = (id) => document.getElementById(id);

const els = {
  ticker: $("ticker"),
  start: $("start"),
  train: $("btn-train"),
  predict: $("btn-predict"),
  retrain: $("btn-retrain"),
  modelStatus: $("model-status"),
  signalCard: $("signal-card"),
  badge: $("signal-badge"),
  title: $("signal-title"),
  sub: $("signal-sub"),
  prob: $("prob-value"),
  planGrid: $("plan-grid"),
  metricsPanel: $("metrics-panel"),
  metricsNote: $("metrics-note"),
  metricsBody: document.querySelector("#metrics-table tbody"),
  log: $("log"),
  spinner: $("spinner"),
};

let pollTimer = null;
let busy = false;

const ticker = () => els.ticker.value.trim().toUpperCase();
const num = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

function setBusy(state) {
  busy = state;
  [els.train, els.predict, els.retrain].forEach((b) => (b.disabled = state));
  els.spinner.classList.toggle("hidden", !state);
}

function setLog(lines) {
  els.log.textContent = lines.join("\n");
  els.log.scrollTop = els.log.scrollHeight;
}

async function checkModel() {
  const t = ticker();
  if (!t) {
    els.modelStatus.textContent = "Enter a ticker to begin.";
    els.modelStatus.className = "model-status";
    return;
  }
  try {
    const res = await fetch(`/api/model?ticker=${encodeURIComponent(t)}`);
    const data = await res.json();
    if (data.trained) {
      const when = new Date(data.trained_at).toLocaleString();
      els.modelStatus.textContent =
        `Model ready for ${t} — best: ${data.best_model} (trained ${when}).`;
      els.modelStatus.className = "model-status trained";
      renderMetrics(data.metrics, data.best_model, t);
    } else {
      els.modelStatus.textContent = `No model trained for ${t} yet — press Train first.`;
      els.modelStatus.className = "model-status untrained";
      els.metricsPanel.classList.add("hidden");
    }
  } catch (err) {
    els.modelStatus.textContent = "Could not reach the server.";
    els.modelStatus.className = "model-status untrained";
  }
}

function renderSignal(r) {
  const sig = (r.signal || "HOLD").toUpperCase();
  els.badge.textContent = sig;
  els.badge.className = `badge ${sig.toLowerCase()}`;
  els.title.textContent = `${r.ticker} — ${sig === "HOLD" ? "no trade" : sig.toLowerCase() + " setup"}`;
  els.sub.textContent = `as of ${r.as_of} · model: ${r.model}`;
  els.prob.textContent = `${(r.probability_up * 100).toFixed(1)}%`;

  const cells =
    sig === "HOLD"
      ? [
          ["Last close", num(r.entry_price)],
          ["ATR (14)", num(r.atr)],
          ["Reason", "probability in neutral band"],
        ]
      : [
          ["Entry", num(r.entry_price)],
          ["Stop loss", num(r.stop_loss)],
          ["Take profit", num(r.take_profit)],
          ["ATR (14)", num(r.atr)],
          ["Risk amount", num(r.risk_amount)],
          ["Position size", `${num(r.position_size, 4)} sh`],
        ];

  els.planGrid.innerHTML = cells
    .map(([k, v]) => `<div class="cell"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");
  els.signalCard.classList.remove("hidden");
}

function renderMetrics(metrics, best, tick) {
  if (!metrics) return;
  els.metricsNote.textContent = `— ${tick || ticker()}, held-out test set`;
  els.metricsBody.innerHTML = Object.entries(metrics)
    .map(([name, m]) => {
      const t = m.test || {};
      const cls = name === best ? ' class="best"' : "";
      return `<tr${cls}><td>${name}</td><td>${num(t.accuracy, 3)}</td><td>${num(t.precision, 3)}</td>` +
        `<td>${num(t.recall, 3)}</td><td>${num(t.f1, 3)}</td><td>${num(t.roc_auc, 3)}</td></tr>`;
    })
    .join("");
  els.metricsPanel.classList.remove("hidden");
}

function handleResult(action, result) {
  if (!result) return;
  if (action === "predict") {
    renderSignal(result);
  } else {
    renderMetrics(result.metrics, result.best_model, result.ticker);
    els.signalCard.classList.add("hidden");
    checkModel();
  }
}

async function poll() {
  const res = await fetch("/api/status");
  const data = await res.json();
  if (data.idle) return;
  setLog(data.log);
  if (!data.running) {
    clearInterval(pollTimer);
    pollTimer = null;
    setBusy(false);
    if (!data.error) handleResult(data.action, data.result);
  }
}

async function start(action) {
  const t = ticker();
  if (!t) {
    els.log.textContent = "Please enter a ticker symbol first.";
    return;
  }
  setBusy(true);
  els.log.textContent = `Starting ${action} for ${t}…`;
  const res = await fetch(`/api/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker: t, start: els.start.value || null }),
  });
  const data = await res.json();
  if (!data.ok) {
    els.log.textContent = data.message;
    setBusy(false);
    return;
  }
  pollTimer = setInterval(poll, 700);
}

els.train.addEventListener("click", () => start("train"));
els.predict.addEventListener("click", () => start("predict"));
els.retrain.addEventListener("click", () => start("retrain"));

let debounce = null;
els.ticker.addEventListener("input", () => {
  clearTimeout(debounce);
  debounce = setTimeout(checkModel, 400);
});
els.ticker.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !busy) start("predict");
});

checkModel();
