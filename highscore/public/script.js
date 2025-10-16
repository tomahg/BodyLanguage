const LOCALE = "nb-NO";
const REFRESH_MS = 2000;

let refreshTimer = null;
let forceRenderPendingOnce = false;

function startPolling() {
  // idempotent: don't reset an already-running timer
  if (refreshTimer) return;
  refreshTimer = setInterval(refresh, REFRESH_MS);
}

function pausePolling() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}

async function refresh() {
  const active = document.activeElement;
  const typing =
    !forceRenderPendingOnce &&
    active &&
    active.tagName === "INPUT" &&
    active.dataset &&
    active.dataset.pendingId;

  const res = await fetch("/state", { cache: "no-store" });
  const state = await res.json();

  renderHighscores(state.highscores);

  // Finn ut om det finnes pending-oppføringer på skjermen
  const hasPendingVisible = document.querySelector("#pending-section .pending-input");

  // Pause polling mens vi skriver / har pending-felter oppe,
  // ellers sørg for at polling kjører.
  if (typing || hasPendingVisible) {
    pausePolling();
  } else {
    startPolling();
  }

  // Ikke oppdater pending mens man skriver eller det vises pending-felter
  if (!typing && !hasPendingVisible) {
    renderPending(state.pending);
    forceRenderPendingOnce = false;
  }
}


function renderHighscores(list) {
  const tbody = document.querySelector("#scores tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  // Sorter og begrens til topp 10 (server gjør dette også, men vi dobbeltsikrer)
  const rows = [...list].sort((a, b) => a.time - b.time).slice(0, 10);

  rows.forEach((s, i) => {
    const tr = document.createElement("tr");

    // Norsk klokkeslett HH:MM
    const klokkeslett = new Date(s.date).toLocaleTimeString(LOCALE, {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });

    tr.innerHTML = `
      <td class="center">${i + 1}.</td>
      <td>${escapeHtml(s.name)}</td>
      <td class="right">${formatSecondsMMSS(s.time)}</td>
      <td>${klokkeslett}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderPending(pending) {
  const section = document.getElementById("pending-section");
  if (!section) return;
  section.innerHTML = "";

  if (!pending || pending.length === 0) {
    // Ikke vis overskrift når det ikke finnes pending
    return;
  }

  const container = document.createElement("div");
  section.appendChild(container);

  pending.forEach((p) => {
    // Ytre rad (sentrerer kolonnen)
    const row = document.createElement("div");
    row.className = "pending-row";

    // Indre kolonne (fast bredde, 4 linjer)
    const col = document.createElement("div");
    col.className = "pending-col";

    // Linje 1: Tid
    const line1 = document.createElement("div");
    line1.className = "pending-line center";
    const label = document.createElement("span");
    label.textContent = `Tid: ${formatSecondsMMSS(p.time)}`;
    line1.appendChild(label);

    // Linje 2: Navn
    const line2 = document.createElement("div");
    line2.className = "pending-line";
    const inputName = document.createElement("input");
    inputName.className = "pending-input";
    inputName.placeholder = "Navn";
    inputName.required = true;
    inputName.dataset.pendingId = p.id;

    // Linje 3: Telefon
    const line3 = document.createElement("div");
    line3.className = "pending-line";
    const inputPhone = document.createElement("input");
    inputPhone.className = "pending-input";
    inputPhone.placeholder = "Telefonnummer";
    inputPhone.required = true;
    inputPhone.type = "tel";
    inputPhone.dataset.pendingId = p.id;

    // Linje 4: Samtykke (checkbox, default unchecked)
    const line4 = document.createElement("div");
    line4.className = "pending-line";
    const inputConsent = document.createElement("input");
    inputConsent.type = "checkbox";
    inputConsent.name = "consent";
    inputConsent.id = `consent-${p.id}`;
    inputConsent.checked = false; // default: ikke krysset av

    const consentLabel = document.createElement("label");
    consentLabel.setAttribute("for", inputConsent.id);
    consentLabel.textContent =
      "Ved innsending samtykker jeg til at NOVA kan kontakte meg i forbindelse med konkurransen, relevante stillinger og nyheter i fremtiden.";

    // litt luft mellom boks og tekst
    inputConsent.style.marginRight = "0.5rem";

    line4.appendChild(inputConsent);
    line4.appendChild(consentLabel);

    // Linje 5: Knapper
    const line5 = document.createElement("div");
    line5.className = "pending-line buttons";
    const btnSave = document.createElement("button");
    btnSave.textContent = "✔";
    btnSave.title = "Lagre navn + telefon";
    const btnX = document.createElement("button");
    btnX.textContent = "✖";
    btnX.title = "Registrer ikke denne tiden";

    // Enter på navn
    inputName.addEventListener("keydown", async (e) => {
      if (e.key === "Enter") {
        const name = inputName.value.trim();
        const phone = inputPhone.value.trim();
        if (!name || !phone) return;
        await register(p.id, name, phone, inputConsent.checked);
        afterSubmitCleanup();
      }
    });

    // Enter på telefon
    inputPhone.addEventListener("keydown", async (e) => {
      if (e.key === "Enter") {
        const name = inputName.value.trim();
        const phone = inputPhone.value.trim();
        if (!name || !phone) return;
        await register(p.id, name, phone, inputConsent.checked);
        afterSubmitCleanup();
      }
    });

    // Klikk på ✔
    btnSave.onclick = async () => {
      const name = inputName.value.trim();
      const phone = inputPhone.value.trim();
      if (!name || !phone) return;
      await register(p.id, name, phone, inputConsent.checked);
      afterSubmitCleanup();
    };

    // Klikk på ✖
    btnX.onclick = async () => {
      await fetch("/dismiss", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: p.id }),
      });
      afterSubmitCleanup();
    };

    line2.appendChild(inputName);
    line3.appendChild(inputPhone);
    line5.appendChild(btnSave);
    line5.appendChild(btnX);

    col.appendChild(line1);
    col.appendChild(line2);
    col.appendChild(line3);
    col.appendChild(line4);
    col.appendChild(line5);

    row.appendChild(col);
    container.appendChild(row);

    inputName.focus();
  });
}

async function register(id, name, phone, consent) {
  await fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name, phone, consent }),
  });
}

function afterSubmitCleanup() {
  // Fjern fokus så typing-sperren ikke hindrer re-render
  if (document.activeElement && document.activeElement.blur) {
    document.activeElement.blur();
  }
  // Skjul pending-section umiddelbart
  const section = document.getElementById("pending-section");
  if (section) section.innerHTML = "";

  // Tving én pending-render og oppdater umiddelbart
  forceRenderPendingOnce = true;
  refresh();          // gjør en umiddelbar sjekk/oppdatering
  startPolling();     // i tilfelle inputs nå er borte – sørg for at polling er i gang
}

// --- Hjelpere ---

// Heltall sekunder -> mm:ss
function formatSecondsMMSS(seconds) {
  const s = Math.round(Number(seconds) || 0);
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${padMinutes(mins)}:${pad2(secs)}`;
}

function padMinutes(m) {
  const str = String(m);
  return str.length >= 2 ? str : "0" + str;
}

function pad2(n) {
  return n < 10 ? "0" + n : String(n);
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Start
startPolling();
refresh();
