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

const runBtn = document.getElementById("sim-run");
const weekdaySelect = document.getElementById("sim-weekday");
const isHolidayInput = document.getElementById("sim-is-holiday");
const workInput = document.getElementById("sim-work");
const gymInput = document.getElementById("sim-gym");
const resultsDiv = document.getElementById("sim-results");

runBtn.addEventListener("click", async () => {
  const payload = {
    weekday: parseInt(weekdaySelect.value, 10),
    is_holiday: isHolidayInput.checked,
    has_work_event: workInput.checked,
    has_gym_event: gymInput.checked,
  };

  resultsDiv.innerHTML = "<div class='loading'>Simulating...</div>";

  try {
    const data = await fetchJSON("/api/simulate_predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const preds = data.predictions || [];

    if (!preds.length) {
      resultsDiv.innerHTML = "<p>No active items found. Add items first.</p>";
      return;
    }

    const wrapper = document.createElement("div");
    preds.forEach((p) => {
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

      row.appendChild(left);
      wrapper.appendChild(row);
    });

    resultsDiv.innerHTML = "";
    resultsDiv.appendChild(wrapper);
  } catch (e) {
    console.error(e);
    resultsDiv.innerHTML =
      "<div class='loading'>Simulation failed. Check console for details.</div>";
  }
});