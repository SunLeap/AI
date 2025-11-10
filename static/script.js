
// DOM refs
const inputText = document.getElementById("inputText");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");

const loading = document.getElementById("loading");
const resultSection = document.getElementById("resultSection");
const resultCard = document.getElementById("resultCard");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");

const sentimentIcon = document.getElementById("sentimentIcon");
const sentimentLabel = document.getElementById("sentimentLabel");
const sentimentEmoji = document.getElementById("sentimentEmoji");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceScore = document.getElementById("confidenceScore");

const starsDisplay = document.getElementById("starsDisplay");
const ratingText = document.getElementById("ratingText");
const suggestionsList = document.getElementById("suggestionsList");

function setExample(t) {
  inputText.value = t;
  updateCharCount();
  updateAnalyzeButton();
}

function updateCharCount() {
  if (!inputText || !charCount) return;
  charCount.textContent = inputText.value.length;
}

function updateAnalyzeButton() {
  analyzeBtn.disabled = inputText.value.trim().length === 0;
}

function setLoading(v) {
  loading.classList.toggle("show", !!v);
  analyzeBtn.disabled = !!v;
}

function showError(msg) {
  errorText.textContent = msg;
  errorMessage.classList.add("show");
  resultSection.classList.remove("show");
}

function clearError() {
  errorMessage.classList.remove("show");
}

function setStars(n) {
  const val = Math.max(1, Math.min(5, Math.round(n)));
  const icons = starsDisplay.querySelectorAll("i");
  icons.forEach((i, idx) => {
    if (idx < val) i.classList.add("filled");
    else i.classList.remove("filled");
  });
  ratingText.textContent = `${val} out of 5 stars`;
}

function setSentiment(label, conf) {
  const lc = (label || "neutral").toLowerCase();
  const map = {
    positive: {emoji: "😊", icon: "fa-smile"},
    neutral: {emoji: "😐", icon: "fa-meh"},
    negative: {emoji: "😞", icon: "fa-frown"},
  };
  const cfg = map[lc] || map.neutral;

  sentimentEmoji.textContent = cfg.emoji;
  sentimentLabel.innerHTML = `<span class="emoji">${cfg.emoji}</span> ${lc[0].toUpperCase() + lc.slice(1)}`;
  confidenceScore.textContent = `${(conf * 100).toFixed(1)}%`;
  confidenceFill.style.width = `${Math.max(4, conf * 100)}%`;

  sentimentIcon.classList.remove("positive", "neutral", "negative");
  resultCard.classList.remove("positive", "neutral", "negative");
  sentimentIcon.classList.add(lc);
  resultCard.classList.add(lc);
}

function suggestFromLabel(label) {
  const lc = (label || "neutral").toLowerCase();
  const items = {
    positive: [
      "Great! Consider highlighting what you liked most.",
      "You could share this as a testimonial or review."
    ],
    neutral: [
      "It sounds okay—try adding specifics about what worked and what didn't.",
      "Consider comparing with alternatives to form a clearer opinion."
    ],
    negative: [
      "Sorry it wasn't great—note the top 1–2 issues to improve next time.",
      "If this is a product/service, consider contacting support for a fix."
    ],
  }[lc] || [];
  suggestionsList.innerHTML = items.map(s => (
    `<div class="suggestion-item ${lc === 'negative' ? 'warning' : lc === 'positive' ? 'success' : 'info'}">
       <i class="fas fa-circle-info"></i><span>${s}</span>
     </div>`
  )).join("");
}

async function analyze() {
  clearError();
  setLoading(true);
  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText.value })
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || "Request failed");
    }
    const { label, confidence, stars } = data;
    setSentiment(label, confidence);
    setStars(stars || 3);
    suggestFromLabel(label);
    resultSection.classList.add("show");
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    setLoading(false);
  }
}

// events
inputText.addEventListener("input", () => { updateCharCount(); updateAnalyzeButton(); });
clearBtn.addEventListener("click", () => {
  inputText.value = "";
  updateCharCount();
  updateAnalyzeButton();
  resultSection.classList.remove("show");
  clearError();
});
analyzeBtn.addEventListener("click", analyze);

// init
updateCharCount();
updateAnalyzeButton();
