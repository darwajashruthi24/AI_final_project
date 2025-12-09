async function fetchJSON(url, options) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Request failed");
  }
  return res.json();
}

const checklistContainer = document.getElementById("checklist-container");
const btnTrain = document.getElementById("btn-train");
const btnLeaving = document.getElementById("btn-leaving");
const ctxSummary = document.getElementById("context-summary");

let predictions = [];
let packedState = {};

async function loadContextSummary() {
  try {
    const data = await fetchJSON("/api/insights");
    const ctx = data.today_context;
    const metrics = data.model_metrics || {};
    let parts = [];

    if (ctx) {
      const weekdayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
      const weekdayName = weekdayNames[ctx.weekday] || ctx.weekday;
      parts.push(`${ctx.day_type} (${weekdayName})`);
      if (ctx.has_work_event) parts.push("work day");
      if (ctx.has_gym_event) parts.push("gym day");
    }

    let txt = "";
    if (parts.length) {
      txt = "Today looks like: " + parts.join(" · ");
    } else {
      txt = "Today’s context is not labelled yet.";
    }

    if (metrics && metrics.n_samples) {
      txt += ` | Model trained on ${metrics.n_samples} samples (F1: ${(metrics.f1 * 100).toFixed(
        1
      )}%)`;
    } else {
      txt += " | Model not fully trained yet.";
    }

    ctxSummary.textContent = txt;
  } catch (e) {
    console.error(e);
    ctxSummary.textContent = "Could not load context summary.";
  }
}

async function loadPredictions() {
  checklistContainer.innerHTML = "<div class='loading'>Loading predictions...</div>";
  try {
    const data = await fetchJSON("/api/predict_today");
    const preds = data.predictions || [];
    predictions = preds;
    packedState = {};
    predictions.forEach((p) => {
      packedState[p.item_id] = false;
    });
    renderChecklist();
  } catch (e) {
    checklistContainer.innerHTML = "<div class='loading'>Failed to load predictions.</div>";
    console.error(e);
  }
}

function renderChecklist() {
  if (!predictions.length) {
    checklistContainer.innerHTML = "<p>No items configured yet. Add items first.</p>";
    return;
  }

  const wrapper = document.createElement("div");
  predictions.forEach((p) => {
    const row = document.createElement("div");
    row.className = "checklist-row";

    const left = document.createElement("div");
    const title = document.createElement("div");
    title.innerHTML = `<strong>${p.name}</strong>`;
    const meta = document.createElement("div");
    meta.className = "small-text";
    meta.textContent = `Need prob: ${(p.need_probability * 100).toFixed(
      0
    )}% · Forget risk: ${(p.forget_risk * 100).toFixed(0)}%`;
    left.appendChild(title);
    left.appendChild(meta);

    const right = document.createElement("label");
    right.className = "small-text";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!packedState[p.item_id];
    input.addEventListener("change", () => {
      packedState[p.item_id] = input.checked;
    });
    right.appendChild(input);
    right.append(" Packed");

    row.appendChild(left);
    row.appendChild(right);
    wrapper.appendChild(row);
  });

  checklistContainer.innerHTML = "";
  checklistContainer.appendChild(wrapper);
}

async function syncChecklist(neededLabel = true) {
  const statuses = predictions.map((p) => ({
    item_id: p.item_id,
    packed: !!packedState[p.item_id],
    needed_label: neededLabel, // assume predicted items are needed
  }));
  try {
    await fetchJSON("/api/checklist_update", {
      method: "POST",
      body: JSON.stringify({ statuses }),
    });
  } catch (e) {
    console.error("Failed to update checklist", e);
  }
}

btnTrain.addEventListener("click", async () => {
  try {
    await fetchJSON("/api/train_model", { method: "POST" });
    alert("Model trained successfully!");
    await loadContextSummary();
  } catch (e) {
    alert("Could not train model yet. Need more diverse data.");
  }
});

btnLeaving.addEventListener("click", async () => {
  await syncChecklist(true);

  const missingImportant = predictions.filter(
    (p) => !packedState[p.item_id] && p.need_probability > 0.6
  );

  if (!missingImportant.length) {
    alert("All good! You’ve packed everything important. ✅");
  } else {
    const names = missingImportant.map((x) => x.name).join(", ");
    alert(
      `Warning! You usually take: ${names}.\nThey are not marked as packed. Are you sure you want to leave?`
    );
  }
});

loadContextSummary();
loadPredictions();