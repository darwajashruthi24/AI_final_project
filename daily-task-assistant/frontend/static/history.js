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

const container = document.getElementById("insights-container");

function renderInsights(data) {
  container.innerHTML = "";

  const { per_item_stats, top_forgotten, model_metrics, today_context } = data;

  // Today context
  if (today_context) {
    const ctxDiv = document.createElement("div");
    ctxDiv.className = "list-item";
    ctxDiv.innerHTML = `
      <div>
        <strong>Today’s context</strong><br />
        <span class="small-text">
          ${today_context.day_type} · weekday index ${today_context.weekday}
          ${today_context.has_work_event ? " · Work event" : ""}
          ${today_context.has_gym_event ? " · Gym event" : ""}
        </span>
      </div>
    `;
    container.appendChild(ctxDiv);
  }

  // Model metrics
  if (model_metrics && model_metrics.n_samples) {
    const mDiv = document.createElement("div");
    mDiv.className = "list-item";
    mDiv.innerHTML = `
      <div>
        <strong>Model performance (on training data)</strong>
        <div class="small-text">
          Accuracy: ${(model_metrics.accuracy * 100).toFixed(1)}% ·
          Precision: ${(model_metrics.precision * 100).toFixed(1)}% ·
          Recall: ${(model_metrics.recall * 100).toFixed(1)}% ·
          F1: ${(model_metrics.f1 * 100).toFixed(1)}%<br />
          Samples used: ${model_metrics.n_samples}
        </div>
      </div>
    `;
    container.appendChild(mDiv);
  } else {
    const mDiv = document.createElement("div");
    mDiv.className = "list-item";
    mDiv.innerHTML = `
      <div>
        <strong>Model performance</strong>
        <div class="small-text">
          Not enough labelled data yet. Use the checklist and then click “Train Model”
          on the dashboard a few times to build history.
        </div>
      </div>
    `;
    container.appendChild(mDiv);
  }

  // Per-item stats table
  if (!per_item_stats || !per_item_stats.length) {
    const p = document.createElement("p");
    p.className = "small-text";
    p.textContent =
      "No item-level stats available yet. Use the checklist for a few days and mark whether items were needed.";
    container.appendChild(p);
  } else {
    const title = document.createElement("h4");
    title.textContent = "Per-item forget statistics";
    container.appendChild(title);

    const table = document.createElement("table");
    table.style.width = "100%";
    table.style.borderCollapse = "collapse";
    table.innerHTML = `
      <thead>
        <tr>
          <th style="text-align:left; padding:4px; border-bottom:1px solid #ddd;">Item</th>
          <th style="text-align:right; padding:4px; border-bottom:1px solid #ddd;">Needed days</th>
          <th style="text-align:right; padding:4px; border-bottom:1px solid #ddd;">Forgotten days</th>
          <th style="text-align:right; padding:4px; border-bottom:1px solid #ddd;">Forget rate</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;
    const tbody = table.querySelector("tbody");

    per_item_stats.forEach((st) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="padding:4px; border-bottom:1px solid #f3f4f6;">${st.name}</td>
        <td style="padding:4px; border-bottom:1px solid #f3f4f6; text-align:right;">${st.needed_days}</td>
        <td style="padding:4px; border-bottom:1px solid #f3f4f6; text-align:right;">${st.forgotten_days}</td>
        <td style="padding:4px; border-bottom:1px solid #f3f4f6; text-align:right;">
          ${(st.forget_rate * 100).toFixed(1)}%
        </td>
      `;
      tbody.appendChild(tr);
    });

    container.appendChild(table);
  }

  // Top forgotten summary (simple natural language explanation)
  if (top_forgotten && top_forgotten.length) {
    const maxItem = top_forgotten[0];
    if (maxItem.needed_days > 0 && maxItem.forget_rate > 0) {
      const summary = document.createElement("p");
      summary.className = "small-text";
      summary.textContent = `Insight: You most often forget "${maxItem.name}" — you needed it on ${maxItem.needed_days} day(s) and forgot it on ${maxItem.forgotten_days} of those (${(
        maxItem.forget_rate * 100
      ).toFixed(1)}% forget rate).`;
      container.appendChild(summary);
    }
  }
}

async function loadInsights() {
  try {
    const data = await fetchJSON("/api/insights");
    renderInsights(data);
  } catch (e) {
    console.error(e);
    container.innerHTML =
      "<div class='loading'>Failed to load insights. Check console for details.</div>";
  }
}

loadInsights();