"use strict";

(() => {
  const library = document.getElementById("creation-library");
  const toggle = document.getElementById("library-toggle");
  const panel = document.getElementById("library-panel");
  const message = document.getElementById("library-message");
  const titles = document.getElementById("library-titles");
  if (!library || !toggle || !panel || !message || !titles) return;

  const cataloguePath = document.querySelector("main")?.dataset.staticCatalogue || "/api/creation-library";

  let open = false;
  let request = 0;
  let controller = null;

  function close(restoreFocus = false) {
    if (!open) return;
    open = false;
    request += 1;
    if (controller) controller.abort();
    panel.hidden = true;
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Open library");
    if (restoreFocus) toggle.focus({preventScroll: true});
  }

  function entry(item) {
    if (!item || typeof item.id !== "string" || typeof item.title !== "string" ||
        !item.title.trim()) return null;
    const details = {id: item.id, title: item.title, version: item.version,
      ready: typeof item.playback_ready === "boolean" ? item.playback_ready : item.semantic_ready !== false};
    if (!details.ready) return details;
    if (typeof item.playback_url !== "string") return null;
    try {
      const url = new URL(item.playback_url, window.location.href);
      if (url.origin !== window.location.origin || url.pathname !== "/") return null;
      const selectedId = url.searchParams.get("creation") || url.searchParams.get("recording") ||
        url.searchParams.get("preview");
      return {...details, href: url.href, selectedId};
    } catch {
      return null;
    }
  }

  function render(items) {
    const entries = items.map(entry).filter(Boolean);
    const titleCounts = new Map();
    entries.forEach(item => titleCounts.set(item.title, (titleCounts.get(item.title) || 0) + 1));
    const params = new URLSearchParams(window.location.search);
    const current = params.get("creation") || params.get("recording") || params.get("preview") ||
      document.querySelector("main").dataset.defaultRecording;
    const fragment = document.createDocumentFragment();
    entries.forEach(item => {
      const row = document.createElement("li");
      const label = document.createElement(item.ready ? "a" : "span");
      label.textContent = item.title;
      if (item.ready) {
        label.href = item.href;
        if (current === item.id || current === item.selectedId ||
            (titleCounts.get(item.title) === 1 &&
             document.getElementById("now-title")?.textContent === item.title)) {
          label.setAttribute("aria-current", "page");
        }
      } else {
        label.className = "library-pending";
        label.setAttribute("aria-disabled", "true");
      }
      if (titleCounts.get(item.title) > 1 && typeof item.version === "string" && item.version) {
        const version = document.createElement("span");
        version.className = "library-version";
        version.textContent = item.version;
        label.append(version);
      }
      if (!item.ready) {
        const status = document.createElement("span");
        status.className = "library-preparing";
        status.textContent = "Preparing visuals";
        label.append(status);
      }
      row.append(label);
      fragment.append(row);
    });
    titles.replaceChildren(fragment);
    titles.hidden = entries.length === 0;
    message.hidden = entries.length > 0;
    message.textContent = entries.length ? "" : "No titles yet";
    const firstLink = titles.querySelector("a");
    if (document.activeElement === panel && firstLink) {
      firstLink.focus({preventScroll: true});
    }
  }

  async function show() {
    open = true;
    const thisRequest = ++request;
    controller = new AbortController();
    const activeController = controller;
    const timeout = window.setTimeout(() => activeController.abort(), 7000);
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", "Close library");
    panel.hidden = false;
    titles.hidden = true;
    message.hidden = false;
    message.textContent = "Loading…";
    panel.focus({preventScroll: true});
    try {
      const response = await fetch(cataloguePath, {
        cache: "no-store",
        signal: activeController.signal
      });
      if (!response.ok) throw new Error("Library unavailable");
      const catalog = await response.json();
      if (!Array.isArray(catalog.items)) throw new Error("Library unavailable");
      if (open && request === thisRequest) render(catalog.items);
    } catch {
      if (open && request === thisRequest) {
        titles.hidden = true;
        message.hidden = false;
        message.textContent = "Library unavailable";
      }
    } finally {
      window.clearTimeout(timeout);
      if (controller === activeController) controller = null;
    }
  }

  toggle.addEventListener("click", () => {
    if (open) close(); else show();
  });
  document.addEventListener("pointerdown", event => {
    if (open && !library.contains(event.target)) close();
  });
  document.addEventListener("keydown", event => {
    if (open && event.key === "Escape") {
      event.preventDefault();
      close(true);
    }
  });
  library.addEventListener("focusout", event => {
    if (open && event.relatedTarget && !library.contains(event.relatedTarget)) close();
  });
  titles.addEventListener("click", event => {
    if (event.target.closest("a")) close();
  });
})();
