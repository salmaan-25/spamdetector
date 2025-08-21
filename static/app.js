const $ = (sel) => document.querySelector(sel);
const resultEl = $("#result");
const btn = $("#checkBtn");

btn.addEventListener("click", async () => {
  const message = $("#message").value.trim();
  resultEl.textContent = "";
  if (!message) {
    resultEl.textContent = "Please enter a message.";
    return;
  }
  btn.disabled = true;
  btn.textContent = "Checking...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    if (!data.ok) {
      resultEl.textContent = "Error: " + (data.error || "Unknown error");
      resultEl.style.color = "#fca5a5";
      return;
    }

    const prob = (data.spam_probability * 100).toFixed(1);
    resultEl.textContent = `Prediction: ${data.label} (${prob}% spam probability)`;
    resultEl.style.color = data.label === "Spam" ? "#fca5a5" : "#86efac";

    // === flash background + revert after a few seconds ===
    const body = document.body;

    // clear any previous timer and classes
    clearTimeout(window._flashTimer);
    body.classList.remove("is-spam", "is-ham");

    // apply new class
    if (data.label === "Spam") {
      body.classList.add("is-spam");
    } else {
      body.classList.add("is-ham");
    }

    // revert to original after 3 seconds
    window._flashTimer = setTimeout(() => {
      body.classList.remove("is-spam", "is-ham");
    }, 3000);
    // =====================================================

  } catch (e) {
    resultEl.textContent = "Network error. Check if the server is running.";
    resultEl.style.color = "#fca5a5";
  } finally {
    btn.disabled = false;
    btn.textContent = "Check";
  }
});
