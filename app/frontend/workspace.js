/* Navigation and presentation state are independent of recommendation requests. */
const Workspace = (() => {
  const el = (id) => document.getElementById(id);
  const pages = {
    overview: ["Overview", "Your workspace, at a glance", "A clear picture of your research and where to go next."],
    publications: ["My publications", "Your research record", "Publications found through your connected academic profile."],
    recommendations: ["Discover research", "New directions", "Papers and publishing venues related to your research interests."],
    collaborators: ["Collaborators", "Build your research network", "Explore researchers with shared interests, locally or around the world."],
    growth: ["Growth plan", "Small steps, longer-term progress", "Suggested actions and saved snapshots of your academic profile."],
    profile: ["My profile", "Your academic identity", "Review your research interests, connected profiles and source data."],
    researchers: ["Find a researcher", "Explore the academic community", "Search OpenAlex and explore public research profiles."],
    models: ["Recommendation settings", "Your research preferences", "Choose a model and understand how recommendations are generated."],
  };
  let authenticated = false;
  let activePage = "overview";
  let discoveryPage = "papers";
  let publicationPage = 1;
  const pageSize = 6;
  const mobileMenu = window.matchMedia("(max-width: 700px)");

  function syncMenuAccess() {
    const open = el("workspace-sidebar").classList.contains("is-open");
    el("workspace-sidebar").inert = mobileMenu.matches && !open;
    document.querySelector(".workspace-body").inert = mobileMenu.matches && open;
  }

  function closeMenu() {
    el("workspace-sidebar").classList.remove("is-open");
    el("sidebar-backdrop").classList.add("hidden");
    el("menu-button").setAttribute("aria-expanded", "false");
    syncMenuAccess();
  }

  function route(focus = false) {
    const requested = location.hash.replace(/^#\/?/, "");
    if (!authenticated) {
      const mode = requested === "register" ? "register" : "login";
      el("login-panel").classList.toggle("hidden", mode !== "login");
      el("register-panel").classList.toggle("hidden", mode !== "register");
      document.querySelectorAll("[data-auth-view]").forEach((link) => {
        if (link.dataset.authView === mode) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
      });
      document.title = (mode === "register" ? "Create account" : "Sign in") + " | Academic Visibility";
      return;
    }
    activePage = Object.hasOwn(pages, requested) ? requested : "overview";
    if (requested !== activePage) history.replaceState(null, "", "#/" + activePage);
    document.querySelectorAll("[data-page]").forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.page !== activePage);
    });
    document.querySelectorAll("[data-view]").forEach((link) => {
      if (link.dataset.view === activePage) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    const [title, eyebrow, description] = pages[activePage];
    el("page-title").textContent = title;
    el("breadcrumb-title").textContent = title;
    el("page-eyebrow").textContent = eyebrow;
    el("page-description").textContent = description;
    document.title = title + " | Academic Visibility";
    closeMenu();
    if (focus) {
      el("page-title").focus({ preventScroll: true });
      window.scrollTo({ top: 0, behavior: "instant" });
    }
  }

  function setAccess(value) {
    authenticated = value;
    if (!value) {
      closeMenu();
      const dialog = el("collaborator-profile-panel");
      if (dialog.open) dialog.close();
    }
    route();
  }

  function navigate(page) {
    if (!Object.hasOwn(pages, page)) return;
    if (location.hash === "#/" + page) route(true);
    else location.hash = "/" + page;
  }

  function setDiscovery(page) {
    discoveryPage = page;
    document.querySelectorAll("[data-discovery]").forEach((button) => {
      const selected = button.dataset.discovery === page;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    document.querySelectorAll("[data-discovery-panel]").forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.discoveryPanel !== page);
    });
  }

  function initials(name) {
    return String(name || "Researcher").trim().split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
  }

  function filterPublications(reset = true) {
    if (reset) publicationPage = 1;
    const query = el("publication-search").value.trim().toLocaleLowerCase();
    const cards = [...el("works-list").children];
    const matching = cards.filter((card) => card.dataset.publication === "true" && card.textContent.toLocaleLowerCase().includes(query));
    const totalPages = Math.max(1, Math.ceil(matching.length / pageSize));
    publicationPage = Math.min(publicationPage, totalPages);
    cards.forEach((card) => {
      const position = matching.indexOf(card);
      const visible = position >= (publicationPage - 1) * pageSize && position < publicationPage * pageSize;
      card.classList.toggle("hidden", card.dataset.publication === "true" ? !visible : Boolean(query));
    });
    el("works-count").textContent = matching.length + (matching.length === 1 ? " publication" : " publications");
    el("works-empty-filter").classList.toggle("hidden", !query || matching.length > 0);
    const pagination = el("works-pagination");
    pagination.replaceChildren();
    if (totalPages <= 1) return;
    const previous = document.createElement("button");
    previous.type = "button";
    previous.className = "button button-secondary";
    previous.textContent = "Previous";
    previous.disabled = publicationPage === 1;
    previous.addEventListener("click", () => { publicationPage--; filterPublications(false); el("page-title").scrollIntoView(); });
    const label = document.createElement("span");
    label.textContent = "Page " + publicationPage + " of " + totalPages;
    const next = document.createElement("button");
    next.type = "button";
    next.className = "button button-secondary";
    next.textContent = "Next";
    next.disabled = publicationPage === totalPages;
    next.addEventListener("click", () => { publicationPage++; filterPublications(false); el("page-title").scrollIntoView(); });
    pagination.append(previous, label, next);
  }

  function foldExplanations(root) {
    root.querySelectorAll(".feed-card").forEach((card) => {
      const extra = [...card.children].filter((child) =>
        child.matches(".compact-list, .model-score-grid, .evidence-block"));
      if (!extra.length) return;
      const details = document.createElement("details");
      details.className = "match-details";
      const summary = document.createElement("summary");
      summary.textContent = "Why this recommendation?";
      details.append(summary, ...extra);
      const button = card.querySelector(".collaborator-profile-button");
      card.insertBefore(details, button);
    });
  }

  function previewRow(title, description) {
    const row = document.createElement("article");
    row.className = "preview-row";
    const heading = document.createElement("h4");
    heading.textContent = title;
    const note = document.createElement("p");
    note.textContent = description;
    row.append(heading, note);
    return row;
  }

  function updateDashboard(data, comparison) {
    el("overview-affiliation").textContent = data.researcher.affiliation || "Affiliation not available";
    el("profile-avatar").textContent = initials(data.researcher.name);
    el("nav-works-count").textContent = data.recent_works.length;
    el("paper-count").textContent = (data.recommended_works || []).length;
    el("venue-count").textContent = data.opportunities.length;
    el("collaborator-count").textContent = data.collaborators.length + (data.collaborators.length === 1 ? " candidate" : " candidates");
    el("comparison-banner").classList.toggle("hidden", !comparison);
    el("comparison-label").textContent = "Exploring " + data.researcher.name + "'s public profile";
    document.querySelector(".workspace-label").textContent = comparison ? "Public profile" : "Personal workspace";
    document.querySelector('[data-view="publications"]').childNodes.forEach((node) => {
      if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) node.textContent = comparison ? "Publications" : "My publications";
    });
    document.querySelector('[data-view="profile"]').childNodes.forEach((node) => {
      if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) node.textContent = comparison ? "Researcher profile" : "My profile";
    });
    pages.publications[0] = comparison ? "Publications" : "My publications";
    pages.profile[0] = comparison ? "Researcher profile" : "My profile";
    pages.publications[2] = comparison ? "Recent publications from this researcher's public profile." : "Publications found through your connected academic profile.";
    const works = el("overview-works");
    works.replaceChildren(...data.recent_works.slice(0, 2).map((work) =>
      previewRow(work.title, [work.year, work.venue].filter(Boolean).join(" / "))));
    if (!data.recent_works.length) works.append(previewRow("No publications connected yet", "Your research interests can still help build recommendations."));
    const people = el("overview-people");
    people.replaceChildren(...data.collaborators.slice(0, 2).map((person) => {
      const row = previewRow(person.name, person.affiliation);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "button button-secondary";
      button.textContent = "View profile";
      button.addEventListener("click", () => openCollaboratorProfile(person.researcher_id).catch(showError));
      row.append(button);
      return row;
    }));
    if (!data.collaborators.length) people.append(previewRow("No matches in this scope", "Try a broader search in Collaborators."));
    [...el("works-list").children].forEach((card) => { card.dataset.publication = String(data.recent_works.length > 0); });
    filterPublications();
    ["recommended-work-list", "opportunity-list", "collaborator-list"].forEach((id) => foldExplanations(el(id)));
    setDiscovery(discoveryPage);
    route();
  }

  function compactGrowth() {
    const grid = el("promotion-tracker-list");
    const diagnostics = [...grid.children].filter((card) => card.matches(".ml-signal-card, .gap-card, .visibility-twin-card"));
    if (!diagnostics.length) return;
    const details = document.createElement("details");
    details.className = "growth-diagnostics";
    const summary = document.createElement("summary");
    summary.textContent = "Profile diagnosis and model signals";
    const content = document.createElement("div");
    content.className = "promotion-grid";
    content.append(...diagnostics);
    details.append(summary, content);
    grid.append(details);
  }

  el("menu-button").addEventListener("click", () => {
    const open = el("workspace-sidebar").classList.toggle("is-open");
    el("sidebar-backdrop").classList.toggle("hidden", !open);
    el("menu-button").setAttribute("aria-expanded", String(open));
    syncMenuAccess();
    if (open) el("workspace-sidebar").querySelector('[aria-current="page"]').focus();
  });
  el("sidebar-backdrop").addEventListener("click", closeMenu);
  mobileMenu.addEventListener("change", closeMenu);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && el("workspace-sidebar").classList.contains("is-open")) {
      closeMenu();
      el("menu-button").focus();
    }
    if (event.key === "Tab" && mobileMenu.matches && el("workspace-sidebar").classList.contains("is-open")) {
      const controls = [...el("workspace-sidebar").querySelectorAll("a, button:not(:disabled)")];
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  el("publication-search").addEventListener("input", () => filterPublications());
  document.querySelector(".skip-link").addEventListener("click", (event) => {
    event.preventDefault();
    if (authenticated) el("page-title").focus();
    else document.querySelector(".auth-form-panel:not(.hidden) input").focus();
  });
  document.querySelectorAll("[data-discovery]").forEach((button) => {
    button.addEventListener("click", () => setDiscovery(button.dataset.discovery));
  });
  document.querySelectorAll("[data-view]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(link.dataset.view);
    });
  });
  el("dismiss-notice").addEventListener("click", () => el("app-notice").classList.add("hidden"));
  window.addEventListener("hashchange", () => route(true));
  syncMenuAccess();
  route();
  return { setAccess, navigate, updateDashboard, foldExplanations, compactGrowth, initials };
})();
