// ================================================
// FRAUD DETECTION SYSTEM — Shared JS Utilities
// ================================================

const API_BASE = "http://127.0.0.1:8000";

// ---- API HELPERS ------------------------------------------------

async function apiFetch(path, options = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ---- AUTH -------------------------------------------------------

const AUTH = {
  // Simple session-based auth (demo only — use real auth in production)
  USERS: { admin: "fraud2024", analyst: "analyze123" },

  login(username, password) {
    if (this.USERS[username] && this.USERS[username] === password) {
      sessionStorage.setItem("fd_user", username);
      return true;
    }
    return false;
  },

  logout() {
    sessionStorage.removeItem("fd_user");
    window.location.href = "../index.html";
  },

  check() {
    const u = sessionStorage.getItem("fd_user");
    if (!u) window.location.href = "../index.html";
    return u;
  },

  user() {
    return sessionStorage.getItem("fd_user") || "guest";
  }
};

// ---- TOAST NOTIFICATIONS ----------------------------------------

function toast(msg, type = "info") {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  el.onclick = () => el.remove();
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity 0.3s"; setTimeout(() => el.remove(), 300); }, 3500);
}

// ---- RISK COLORS & BADGES ---------------------------------------

function riskBadge(level) {
  return `<span class="badge badge-${level}">${level}</span>`;
}

function riskColor(level) {
  return { LOW: "var(--green)", MEDIUM: "var(--yellow)", HIGH: "var(--orange)", CRITICAL: "var(--critical)" }[level] || "var(--text)";
}

function probColor(p) {
  if (p >= 0.7) return "var(--critical)";
  if (p >= 0.4) return "var(--orange)";
  if (p >= 0.2) return "var(--yellow)";
  return "var(--green)";
}

// ---- FORMATTING -------------------------------------------------

function fmt$( n)  { return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", minimumFractionDigits: 0 }).format(n); }
function fmtPct(n) { return (n * 100).toFixed(1) + "%"; }
function fmtDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" }) + " " +
         d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false });
}
function fmtShort(iso) {
  return new Date(iso).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false });
}

// ---- API STATUS CHECK -------------------------------------------

async function checkApiStatus() {
  const dot  = document.getElementById("api-status-dot");
  const text = document.getElementById("api-status-text");
  try {
    await apiFetch("/health");
    if (dot)  { dot.className = "status-dot online"; }
    if (text) text.textContent = "API online";
  } catch {
    if (dot)  { dot.className = "status-dot offline"; }
    if (text) text.textContent = "API offline";
  }
}

// ---- NAV HIGHLIGHT ----------------------------------------------

function setActiveNav() {
  const path = window.location.pathname;
  document.querySelectorAll(".nav-link").forEach(a => {
    a.classList.toggle("active", path.includes(a.getAttribute("href")));
  });
}

// ---- SIDEBAR SHARED HTML ----------------------------------------

function renderSidebar(activePage) {
  const links = [
    { href: "dashboard.html",  icon: "⬡", label: "Dashboard" },
    { href: "checker.html",    icon: "◈", label: "Fraud Checker" },
    { href: "history.html",    icon: "≡", label: "History" },
    { href: "analytics.html",  icon: "◉", label: "Analytics" },
    { href: "model.html",      icon: "⬡", label: "Model Info" },
  ];
  const el = document.getElementById("sidebar");
  if (!el) return;
  el.innerHTML = `
    <div class="sidebar-logo">
      <div class="logo-mark">FraudShield</div>
      <div class="logo-sub">Detection System v2</div>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section-label">Navigation</div>
      ${links.map(l => `
        <a class="nav-link ${activePage === l.href ? 'active' : ''}" href="${l.href}">
          <span class="icon">${l.icon}</span>${l.label}
        </a>`).join("")}
      <div class="nav-section-label" style="margin-top:16px">Account</div>
      <a class="nav-link" href="#" onclick="AUTH.logout()">
        <span class="icon">⏻</span>Logout
      </a>
    </nav>
    <div class="sidebar-footer">
      <div class="api-status">
        <span class="status-dot" id="api-status-dot"></span>
        <span id="api-status-text">Checking...</span>
      </div>
      <div style="margin-top:6px;color:var(--text-dim);font-size:10px;">User: ${AUTH.user()}</div>
    </div>`;
  checkApiStatus();
}

// Init on every page
document.addEventListener("DOMContentLoaded", () => {
  setActiveNav();
  // Poll API status every 30s
  setInterval(checkApiStatus, 30000);
});
