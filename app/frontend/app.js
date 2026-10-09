const state = {
  researchers: [],
  selectedResearcherId: null,
  registeredProfile: null,
  currentUser: null,
  authToken: null,
  collaboratorScope: "global",
  embeddingModel: "specter2",
  dashboardContext: "cabinet",
};

const AUTH_TOKEN_STORAGE_KEY = "auth_token";
const byId = (id) => document.getElementById(id);
let dashboardLoadId = 0;
let collaboratorLoadId = 0;
let searchLoadId = 0;
let hasDashboard = false;
const completedActions = new Set();

function prepareDashboard() {
  byId("app-notice").classList.add("hidden");
  byId("dashboard-state").classList.toggle("hidden", hasDashboard);
  byId("dashboard-content").setAttribute("aria-busy", "true");
}

async function runTask(button, task) {
  if (button?.disabled) return;
  if (button) button.disabled = true;
  try {
    await task();
  } catch (error) {
    showError(error);
  } finally {
    if (button) button.disabled = button.dataset.completed === "true" ||
      (button.id === "snapshot-button" && state.dashboardContext !== "cabinet");
    byId("dashboard-content").setAttribute("aria-busy", "false");
  }
}

async function fetchJson(url, options = {}) {
  const headers = {
    ...(options.headers || {}),
  };
  if (state.authToken) {
    headers.Authorization = `Bearer ${state.authToken}`;
  }
  const response = await fetch(url, { ...options, headers, cache: "no-store" });
  if (!response.ok) {
    const error = new Error(response.status === 401
      ? "Your session has expired. Please sign in again."
      : "We could not load this information. Please try again.");
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function metricCard(label, value, note = "", tone = "") {
  return `
    <article class="metric-card ${tone}">
      <span>${label}</span>
      <strong>${value}</strong>
      ${note ? `<small>${note}</small>` : ""}
    </article>
  `;
}

function scorePercent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function scoreLabel(value, lowerIsBetter = false) {
  const score = Number(value) || 0;
  if (lowerIsBetter) {
    if (score <= 0.25) return "low";
    if (score <= 0.55) return "medium";
    return "high";
  }
  if (score >= 0.75) return "strong";
  if (score >= 0.45) return "developing";
  return "needs work";
}

function compactScore(value) {
  const number = Number(value) || 0;
  return number.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function externalLink(url, label) {
  if (!url) return "";
  try {
    const target = new URL(url);
    if (!["https:", "http:"].includes(target.protocol)) return "";
    const anchor = document.createElement("a");
    anchor.className = "inline-link";
    anchor.href = target.href;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    anchor.textContent = label;
    return anchor.outerHTML;
  } catch {
    return "";
  }
}

function modelScoreGrid(modelScores = []) {
  if (!modelScores.length) {
    return "";
  }
  return `
    <div class="model-score-grid" aria-label="Model comparison scores">
      ${modelScores.map((model) => `
        <div class="model-score-chip">
          <span>${model.name}</span>
          <strong>${compactScore(model.score)}</strong>
          <small>${model.role}</small>
        </div>
      `).join("")}
    </div>
  `;
}

function splitCsv(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function scopeLabel(scope) {
  const labels = {
    global: "Global",
    same_country: "Same country",
    same_affiliation: "Same affiliation",
  };
  return labels[scope] || labels.global;
}

function scopeDescription(scope) {
  const descriptions = {
    global: "Researchers around the world with related research interests.",
    same_country: "Researchers whose country matches the country in this profile.",
    same_affiliation: "Researchers linked to the same university or organization.",
  };
  return descriptions[scope] || descriptions.global;
}

function embeddingModelLabel(model) {
  const labels = {
    hashing: "Hashing baseline",
    scibert: "SciBERT",
    specter2: "SPECTER2",
  };
  return labels[model] || labels.specter2;
}

function embeddingModelDescription(model, status = "") {
  const descriptions = {
    hashing: "Hashing baseline is a fast local vector model for comparison.",
    scibert: "SciBERT uses scientific BERT embeddings stored in PostgreSQL/pgvector.",
    specter2: "SPECTER2 uses scientific-paper embeddings stored in PostgreSQL/pgvector.",
  };
  return status || descriptions[model] || descriptions.specter2;
}

function renderScopeControls(scope) {
  state.collaboratorScope = scope || "global";
  document.querySelectorAll("[data-collaborator-scope]").forEach((button) => {
    button.classList.toggle("active", button.dataset.collaboratorScope === state.collaboratorScope);
    button.setAttribute("aria-pressed", String(button.dataset.collaboratorScope === state.collaboratorScope));
  });
  byId("collaborator-scope-note").textContent = scopeDescription(state.collaboratorScope);
}

function renderEmbeddingControls(model, status = "") {
  state.embeddingModel = model || "specter2";
  document.querySelectorAll("[data-embedding-model]").forEach((button) => {
    button.classList.toggle("active", button.dataset.embeddingModel === state.embeddingModel);
    button.setAttribute("aria-pressed", String(button.dataset.embeddingModel === state.embeddingModel));
  });
  byId("embedding-model-note").textContent = embeddingModelDescription(state.embeddingModel, status);
  byId("active-model-label").textContent = embeddingModelLabel(state.embeddingModel);
}

function setAccessMode(isRegistered) {
  byId("onboarding-shell").classList.toggle("hidden", isRegistered);
  byId("app-shell").classList.toggle("hidden", !isRegistered);
  Workspace.setAccess(isRegistered);
}

function renderRegisteredUserCard(data) {
  const summary = `${data.position_title} at ${data.affiliation}. Top domains: ${data.domain_candidates.map((item) => item.name).join(", ") || "not available"}.`;
  const accountLine = state.currentUser ? `Account: ${state.currentUser.email}` : "Account session active.";
  byId("current-user-name").textContent = data.full_name;
  byId("user-avatar").textContent = Workspace.initials(data.full_name);
  byId("current-user-summary").textContent = summary;

  const card = byId("registered-user-card");
  card.classList.remove("hidden");
  card.innerHTML = `
    <div class="section-head slim">
      <div>
        <p class="section-kicker">Current user</p>
        <h4>${data.full_name}</h4>
      </div>
      <span class="badge">${data.profile_completeness} completeness</span>
    </div>
    <div class="stack-list">
      <article class="feed-card">
        <div class="feed-header">
          <span class="type-pill">field</span>
          <strong>${data.visibility_score}</strong>
        </div>
        <h4>${data.academic_field}</h4>
        <p>${summary}</p>
        <div class="feed-meta">${accountLine}</div>
        <div class="feed-meta">Data sources: ${data.profile_data_sources.join(", ")}</div>
      </article>
    </div>
  `;
}

function renderResearchers() {
  const container = byId("researcher-list");
  container.innerHTML = "";
  byId("researcher-count").textContent = `${state.researchers.length} ${state.researchers.length === 1 ? "result" : "results"}`;

  state.researchers.forEach((researcher) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `researcher-item ${state.selectedResearcherId === researcher.researcher_id ? "active" : ""}`;
    button.innerHTML = `
      <strong>${researcher.name}</strong>
      <span>${researcher.affiliation}</span>
      <span>${researcher.works_count} works - ${researcher.cited_by_count} citations</span>
    `;
    const action = document.createElement("span");
    action.className = "result-action";
    action.textContent = "Explore profile";
    button.appendChild(action);
    button.addEventListener("click", () => runTask(button, async () => {
      await selectResearcher(researcher.researcher_id);
      Workspace.navigate("overview");
    }));
    container.appendChild(button);
  });
}

function dashboardQueryParams(extra = {}) {
  const params = new URLSearchParams({
    collaborator_scope: state.collaboratorScope,
    embedding_model: state.embeddingModel,
    ...extra,
  });
  return params.toString();
}

function renderDashboard(data) {
  hasDashboard = true;
  byId("dashboard-content").setAttribute("aria-busy", "false");
  byId("dashboard-state").classList.add("hidden");
  byId("dashboard-content").classList.remove("hidden");
  renderScopeControls(data.collaborator_scope || state.collaboratorScope);
  renderEmbeddingControls(data.embedding_model || state.embeddingModel, data.embedding_model_status || "");
  byId("collaborator-model-note").textContent = data.collaborator_model_status || "";

  byId("profile-name").textContent = data.researcher.name;
  byId("profile-stage").textContent = data.analytics.ecr_status ? "Early-career signals" : "Established profile";
  byId("profile-summary").innerHTML = `
    <div class="summary-block profile-anchor">
      <span>Affiliation</span>
      <strong>${data.researcher.affiliation}</strong>
      <small>From your profile and connected academic sources.</small>
    </div>
    <div class="summary-block">
      <span>ORCID</span>
      <strong>${data.researcher.orcid || "Not available"}</strong>
      <small>${data.researcher.orcid ? "Connected identity source." : "Add ORCID to reduce uncertainty."}</small>
    </div>
    <div class="summary-block">
      <span>Research fingerprint</span>
      <strong>${data.researcher.topics.map((topic) => topic.name).join(", ") || "Not available"}</strong>
      <small>These topics drive venue and collaborator search.</small>
    </div>
  `;

  byId("metrics-grid").innerHTML = [
    metricCard(
      "Profile strength",
      compactScore(data.analytics.profile_strength),
      `${scoreLabel(data.analytics.profile_strength)} - based on works, citations, and h-index`,
      "metric-primary"
    ),
    metricCard(
      "Cold-start risk",
      compactScore(data.analytics.cold_start_risk),
      `${scoreLabel(data.analytics.cold_start_risk, true)} risk - lower is better`,
      "metric-risk"
    ),
    metricCard(
      "Publications",
      data.analytics.publication_count,
      "Found in your connected research record",
      "metric-neutral"
    ),
    metricCard(
      "Collaboration readiness",
      compactScore(data.analytics.collaboration_readiness),
      `${scoreLabel(data.analytics.collaboration_readiness)} - topic profile is usable for matchmaking`,
      "metric-success"
    ),
  ].join("");

  byId("dashboard-insight").innerHTML = `
    <article class="insight-card">
      <div>
        <span class="type-pill">What this means</span>
        <h4>${data.analytics.ecr_status ? "Early-career profile with usable publication evidence" : "Established profile with stronger historical evidence"}</h4>
      </div>
      <p>
        The system found ${data.analytics.publication_count} verified works and a
        ${scoreLabel(data.analytics.collaboration_readiness)} collaboration signal.
        Cold-start risk is ${scoreLabel(data.analytics.cold_start_risk, true)}, so recommendations should be treated as
        ${data.analytics.cold_start_risk > 0.55 ? "exploratory until more identifiers and works are connected." : "usable, but still worth validating manually."}
      </p>
    </article>
  `;

  byId("works-list").innerHTML = data.recent_works.map((item) => `
    <article class="feed-card">
      <div class="feed-header">
        <span class="type-pill">${item.source_type || "work"}</span>
        <strong>${item.year || "n/a"}</strong>
      </div>
      <h4>${item.title}</h4>
      <p>${item.venue || "Unknown venue"}</p>
      <div class="feed-meta">${item.cited_by_count} citations</div>
      <div class="feed-meta">${item.topics.join(", ") || "No topic tags"}</div>
      ${externalLink(item.landing_page_url, "Open article")}
    </article>
  `).join("") || `
    <article class="feed-card">
      <div class="feed-header">
        <span class="type-pill">works</span>
        <strong>cold start</strong>
      </div>
      <h4>No verified publications were loaded</h4>
      <p>This personal cabinet is currently operating from registration data and connected profile links only.</p>
    </article>
  `;

  byId("recommended-work-list").innerHTML = (data.recommended_works || []).length ? data.recommended_works.map((item) => `
    <article class="feed-card transformer-card">
      <div class="feed-header">
        <span class="type-pill">${item.opportunity_type}</span>
        <strong>${compactScore(item.score)} match</strong>
      </div>
      <h4>${item.title}</h4>
      <p>${item.description}</p>
      <div class="feed-meta">Model: ${embeddingModelLabel(data.embedding_model || state.embeddingModel)}</div>
      <div class="feed-meta">Year/source: ${item.region}</div>
      <ul class="compact-list">
        ${item.explanation.reasons.map((reason) => `<li>${reason}</li>`).join("")}
      </ul>
      ${modelScoreGrid(item.explanation.model_scores)}
      ${externalLink(item.homepage_url, "Open paper")}
    </article>
  `).join("") : `
    <article class="feed-card transformer-card">
      <div class="feed-header">
        <span class="type-pill">Papers</span>
        <strong>${embeddingModelLabel(data.embedding_model || state.embeddingModel)}</strong>
      </div>
      <h4>No paper recommendations available yet</h4>
      <p>This model has not returned paper matches for your profile. Check its availability or try another model in settings.</p>
      <a class="inline-link" href="#/models">Review recommendation settings</a>
    </article>
  `;

  byId("opportunity-list").innerHTML = data.opportunities.length ? data.opportunities.map((item) => `
    <article class="feed-card">
      <div class="feed-header">
        <span class="type-pill">${item.opportunity_type}</span>
        <strong>${compactScore(item.score)} match</strong>
      </div>
      <h4>${item.title}</h4>
      <p>${item.description}</p>
      <div class="feed-meta">${item.region}</div>
      <ul class="compact-list">
        ${item.explanation.reasons.map((reason) => `<li>${reason}</li>`).join("")}
      </ul>
      ${modelScoreGrid(item.explanation.model_scores)}
      ${externalLink(item.homepage_url, "Visit venue website")}
    </article>
  `).join("") : `
    <article class="feed-card">
      <div class="feed-header">
        <span class="type-pill">venues</span>
        <strong>no match</strong>
      </div>
      <h4>No real OpenAlex venues were found for this profile yet</h4>
      <p>Add more specific research keywords, methods, ORCID, or publication goals to broaden the search context.</p>
    </article>
  `;

  byId("collaborator-list").innerHTML = data.collaborators.length ? data.collaborators.map((item) => `
    <article class="feed-card">
      <div class="feed-header">
        <span class="type-pill">collaborator</span>
        <strong>${compactScore(item.score)} match</strong>
      </div>
      <h4>${item.name}</h4>
      <p>${item.affiliation}</p>
      <div class="feed-meta">${item.country_code || "Unknown country"}</div>
      <div class="feed-meta">Filter match: ${item.scope_match || scopeLabel(state.collaboratorScope)}</div>
      <ul class="compact-list">
        ${item.explanation.reasons.map((reason) => `<li>${reason}</li>`).join("")}
      </ul>
      ${modelScoreGrid(item.explanation.model_scores)}
      ${(item.evidence_works || []).length ? `
        <div class="evidence-block">
          <strong>Evidence works</strong>
          ${(item.evidence_works || []).slice(0, 2).map((work) => `
            <p>${work.title} ${work.year ? `(${work.year})` : ""}</p>
          `).join("")}
        </div>
      ` : ""}
      <button class="button button-secondary collaborator-profile-button" type="button" data-researcher-id="${item.researcher_id}">
        View profile
      </button>
    </article>
  `).join("") : `
    <article class="feed-card">
      <div class="feed-header">
        <span class="type-pill">collaborators</span>
        <strong>${scopeLabel(state.collaboratorScope)}</strong>
      </div>
      <h4>No collaborator candidates matched this filter</h4>
      <p>The recommender searches recent real papers and extracts authors from them. Try Global mode for a broader search, or add more specific profile keywords and ORCID data.</p>
    </article>
  `;

  document.querySelectorAll(".collaborator-profile-button").forEach((button) => {
    button.addEventListener("click", () => openCollaboratorProfile(button.dataset.researcherId).catch(showError));
  });
  Workspace.updateDashboard(data, state.dashboardContext === "researcher");
}

function shortList(items, limit = 4) {
  return items.slice(0, limit).join(", ") || "Not available";
}

function renderCollaboratorProfile(data) {
  const panel = byId("collaborator-profile-panel");
  const content = byId("collaborator-profile-content");
  byId("collaborator-profile-name").textContent = data.researcher.name;

  content.innerHTML = `
    <div class="profile-summary">
      <div class="summary-block">
        <span>Affiliation</span>
        <strong>${data.researcher.affiliation}</strong>
      </div>
      <div class="summary-block">
        <span>Metrics</span>
        <strong>${data.researcher.works_count} works, ${data.researcher.cited_by_count} citations, h-index ${data.researcher.h_index}</strong>
      </div>
      <div class="summary-block">
        <span>Topics</span>
        <strong>${shortList(data.researcher.topics.map((topic) => topic.name), 6)}</strong>
      </div>
    </div>
    <div class="two-column">
      <section>
        <div class="section-head slim">
          <div>
            <p class="section-kicker">Recent Works</p>
            <h4>What this collaborator publishes</h4>
          </div>
        </div>
        <div class="stack-list">
          ${data.recent_works.slice(0, 5).map((item) => `
            <article class="feed-card">
              <div class="feed-header">
                <span class="type-pill">${item.source_type || "work"}</span>
                <strong>${item.year || "n/a"}</strong>
              </div>
              <h4>${item.title}</h4>
              <p>${item.venue || "Unknown venue"}</p>
              <div class="feed-meta">${item.cited_by_count} citations</div>
              <div class="feed-meta">${shortList(item.topics, 3)}</div>
              ${externalLink(item.landing_page_url, "Open article")}
            </article>
          `).join("") || `
            <article class="feed-card">
              <h4>No recent works loaded</h4>
              <p>OpenAlex did not return publications for this candidate.</p>
            </article>
          `}
        </div>
      </section>
      <section>
        <div class="section-head slim">
          <div>
            <p class="section-kicker">Collaboration Fit</p>
            <h4>Potential shared venues</h4>
          </div>
        </div>
        <div class="stack-list">
          ${data.opportunities.slice(0, 4).map((item) => `
            <article class="feed-card">
              <div class="feed-header">
                <span class="type-pill">${item.opportunity_type}</span>
                <strong>${item.score}</strong>
              </div>
              <h4>${item.title}</h4>
              <p>${item.description}</p>
              <ul class="compact-list">
                ${item.explanation.reasons.map((reason) => `<li>${reason}</li>`).join("")}
              </ul>
              ${modelScoreGrid(item.explanation.model_scores)}
            </article>
          `).join("") || `
            <article class="feed-card">
              <h4>No source suggestions loaded</h4>
              <p>The profile is visible, but source recommendations are unavailable right now.</p>
            </article>
          `}
        </div>
      </section>
    </div>
  `;
  Workspace.foldExplanations(content);
  if (!panel.open) panel.showModal();
}

async function openCollaboratorProfile(researcherId) {
  if (!researcherId) {
    return;
  }
  const requestId = ++collaboratorLoadId;
  const panel = byId("collaborator-profile-panel");
  if (!panel.open) panel.showModal();
  byId("collaborator-profile-name").textContent = "Loading collaborator...";
  byId("collaborator-profile-content").innerHTML = `
    <article class="feed-card">
      <h4>Loading profile</h4>
      <p>Fetching publications, topics, and source recommendations from OpenAlex.</p>
    </article>
  `;
  try {
    const dashboard = await fetchJson(
      `/researchers/${encodeURIComponent(researcherId)}/dashboard?${dashboardQueryParams({ top_k: 4 })}`
    );
    if (requestId === collaboratorLoadId && panel.open) renderCollaboratorProfile(dashboard);
  } catch (error) {
    if (requestId !== collaboratorLoadId || !panel.open) return;
    byId("collaborator-profile-name").textContent = "Profile unavailable";
    byId("collaborator-profile-content").textContent = "This profile could not be loaded. Close this window and try again.";
  }
}

function renderCabinetMeta(cabinet) {
  const tracker = cabinet.promotion_tracker;
  byId("promotion-tracker-list").innerHTML = tracker ? `
    <article class="promotion-card promotion-score-card">
      <span>Promotion score</span>
      <strong>${compactScore(tracker.promotion_score)}</strong>
      <p>Overall readiness for academic profile promotion. Higher is better.</p>
      <div class="score-mini">Gap to target: ${compactScore(tracker.visibility_gap)}</div>
    </article>
    <article class="promotion-card">
        <span>Venue opportunity score</span>
      <strong>${compactScore(tracker.citation_growth_potential)}</strong>
        <p>A profile-based indicator, not a forecast of future citations.</p>
    </article>
    <article class="promotion-card">
        <span>Collaboration opportunity</span>
      <strong>${compactScore(tracker.collaboration_growth_potential)}</strong>
      <p>How strong the current profile is for finding relevant co-authors.</p>
    </article>
    <article class="promotion-card promotion-actions-card">
      <span>Progress loop</span>
      <strong>30 days</strong>
      <p>Save a baseline today, complete actions, then compare score changes later.</p>
    </article>
    <article class="promotion-card visibility-twin-card">
      <span>Visibility Twin</span>
      <strong>Digital profile diagnosis</strong>
      <p>${tracker.visibility_twin_summary}</p>
    </article>
    ${(tracker.ml_signals || []).map((signal) => `
      <article class="promotion-card ml-signal-card">
        <span>${signal.name}</span>
        <strong>${signal.value}</strong>
        <p>${signal.description}</p>
      </article>
    `).join("")}
    ${(tracker.gap_dimensions || []).map((dimension) => `
      <article class="promotion-card gap-card">
        <div class="gap-topline">
          <span>${dimension.name}</span>
          <strong>${scorePercent(dimension.current)}</strong>
        </div>
        <div class="gap-bar" aria-label="${dimension.name} gap">
          <div style="width: ${Math.round((dimension.current || 0) * 100)}%"></div>
        </div>
        <div class="gap-meta">
          <span>Current score: ${compactScore(dimension.current)}</span>
          <span>Gap to target: ${compactScore(dimension.gap)}</span>
        </div>
        <p>${dimension.explanation}</p>
      </article>
    `).join("")}
    <article class="promotion-card roadmap-card">
      <span>Roadmap</span>
      <strong>${(tracker.roadmap || []).length} steps</strong>
      <div class="roadmap-list">
        ${(tracker.roadmap || []).map((step) => `
          <div class="roadmap-step">
            <small>${step.phase}</small>
            <h4>${step.title}</h4>
            <p>${step.objective}</p>
            <div class="feed-meta">Metric: ${step.metric}</div>
          </div>
        `).join("")}
      </div>
    </article>
    ${(tracker.recommended_actions || []).map((action) => `
      <article class="promotion-card action-card priority-${action.priority}">
        <div class="feed-header">
          <span class="type-pill">${action.category}</span>
          <strong>${action.priority}</strong>
        </div>
        <h4>${action.title}</h4>
        <p>${action.rationale}</p>
        <div class="feed-meta">Expected impact score: ${compactScore(action.expected_impact)}</div>
        <button class="button button-secondary action-status-button" type="button" data-action-title="${action.title}">
          Mark as completed
        </button>
      </article>
    `).join("")}
  ` : `
    <article class="feed-card">
      <h4>Promotion tracker is not available</h4>
      <p>Complete registration and connect ORCID to generate promotion actions.</p>
    </article>
  `;

  byId("recommendation-summary-list").innerHTML = `
    <article class="feed-card engine-card">
      <div class="feed-header">
        <span class="type-pill">${cabinet.recommendation_summary.mode}</span>
        <strong>${cabinet.personalized_dashboard.analytics.publication_count} works</strong>
      </div>
      <h4>Profile basis</h4>
      <p>${cabinet.recommendation_summary.profile_basis}</p>
      <div class="engine-flow">
        ${cabinet.recommendation_summary.explanation_steps.map((step, index) => `
          <div class="engine-step">
            <span>${index + 1}</span>
            <p>${step}</p>
          </div>
        `).join("")}
      </div>
    </article>
  `;

  document.querySelectorAll(".action-status-button").forEach((button) => {
    if (completedActions.has(button.dataset.actionTitle)) {
      button.disabled = true;
      button.dataset.completed = "true";
      button.textContent = "Completed";
    }
    button.addEventListener("click", () => runTask(button, async () => {
      const accountToken = state.authToken;
      await recordCompletedAction(button.dataset.actionTitle);
      if (state.authToken !== accountToken) return;
      completedActions.add(button.dataset.actionTitle);
      button.dataset.completed = "true";
      button.textContent = "Completed";
      button.setAttribute("aria-label", "Action completed");
    }));
  });

  byId("connected-profile-list").innerHTML = cabinet.connected_profiles.length
    ? cabinet.connected_profiles.map((item) => `
      <article class="feed-card">
        <div class="feed-header">
          <span class="type-pill">${item.platform}</span>
          <strong>${item.status}</strong>
        </div>
        <h4>${item.platform}</h4>
        <p>${item.note}</p>
          ${externalLink(item.url, "Open " + item.platform + " profile")}
      </article>
    `).join("")
    : `
      <article class="feed-card">
        <div class="feed-header">
          <span class="type-pill">profiles</span>
          <strong>missing</strong>
        </div>
        <h4>No connected academic links yet</h4>
        <p>Add ORCID, Google Scholar, or ResearchGate links to enrich the personal cabinet automatically.</p>
      </article>
    `;

  byId("project-list").innerHTML = cabinet.verified_projects.length
    ? cabinet.verified_projects.map((item) => `
      <article class="feed-card">
        <div class="feed-header">
          <span class="type-pill">project</span>
        </div>
        <h4>${item}</h4>
      </article>
    `).join("")
    : `
      <article class="feed-card">
        <div class="feed-header">
          <span class="type-pill">projects</span>
          <strong>not verified</strong>
        </div>
        <h4>No project records were verified from connected academic profiles</h4>
        <p>OpenAlex and the current public profile integration expose publications more reliably than project metadata.</p>
      </article>
        `;
  Workspace.compactGrowth();
}

function renderPromotionProgress(progress) {
  const container = byId("promotion-progress-list");
  if (!progress || !progress.latest) {
    container.innerHTML = `
      <article class="feed-card progress-card">
        <div class="feed-header">
          <span class="type-pill">progress</span>
          <strong>no snapshots</strong>
        </div>
        <h4>No promotion baseline yet</h4>
        <p>Create a snapshot after reviewing the cabinet. Later snapshots will show deltas in visibility, citations, works, h-index, and completed actions.</p>
      </article>
    `;
    return;
  }

  const deltas = progress.deltas || {};
  const hasProgress = Object.values(deltas).some((value) => Number(value) !== 0);
  container.innerHTML = `
    <article class="feed-card progress-card">
      <div class="feed-header">
        <span class="type-pill">progress</span>
        <strong>${progress.snapshots.length} snapshots</strong>
      </div>
      <h4>${hasProgress ? "Promotion progress delta" : "No measurable progress yet"}</h4>
      <p>
        ${hasProgress
          ? "The latest snapshot changed compared with the first saved baseline."
          : "Snapshots were saved, but profile metrics did not change yet. Complete an action or wait for OpenAlex metrics to update, then save a new snapshot."}
      </p>
      <div class="metrics-grid registration-metrics">
        ${metricCard("Promotion delta", deltas.promotion_score ?? 0)}
        ${metricCard("Visibility delta", deltas.visibility_score ?? 0)}
        ${metricCard("Works delta", deltas.works_count ?? 0)}
        ${metricCard("Citations delta", deltas.citation_count ?? 0)}
        ${metricCard("h-index delta", deltas.h_index ?? 0)}
        ${metricCard("Actions done", progress.latest.completed_actions ?? 0)}
      </div>
      <p>Latest snapshot: ${new Date(progress.latest.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}</p>
    </article>
  `;
}

async function loadPromotionProgress() {
  if (!state.authToken) {
    return;
  }
  const requestId = dashboardLoadId;
  const progress = await fetchJson("/auth/me/progress");
  if (requestId === dashboardLoadId && state.dashboardContext === "cabinet" && state.authToken) {
    renderPromotionProgress(progress);
  }
}

async function createPromotionSnapshot() {
  if (!state.authToken || state.dashboardContext !== "cabinet") {
    return;
  }
  byId("promotion-progress-list").innerHTML = `
    <article class="feed-card progress-card">
      <h4>Creating snapshot</h4>
      <p>Refreshing cabinet metrics and saving the current promotion state.</p>
    </article>
  `;
  await fetchJson(`/auth/me/snapshots?${dashboardQueryParams()}`, { method: "POST" });
  await loadPromotionProgress();
}

async function recordCompletedAction(title) {
  if (!state.authToken || !title) {
    return;
  }
  await fetchJson("/auth/me/actions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, status: "completed" }),
  });
  await createPromotionSnapshot();
}


function buildRegistrationPayload(form) {
  const formData = new FormData(form);
  const numberFields = new Set([
    "years_in_research",
  ]);
  const csvFields = new Set([
    "research_keywords",
    "thematic_clusters",
    "methods",
    "target_journals",
    "collaboration_goals",
    "publication_goals",
    "platforms",
    "preferred_languages",
  ]);

  const payload = {};
  for (const [key, rawValue] of formData.entries()) {
    const value = typeof rawValue === "string" ? rawValue.trim() : rawValue;
    if (csvFields.has(key)) {
      payload[key] = splitCsv(value);
    } else if (numberFields.has(key)) {
      payload[key] = Number(value || 0);
    } else {
      payload[key] = value || "";
    }
  }
  return payload;
}

async function loadAuthenticatedCabinet() {
  const requestId = ++dashboardLoadId;
  setAccessMode(true);
  prepareDashboard();
  const session = await fetchJson(`/auth/me/cabinet?${dashboardQueryParams()}`);
  if (requestId !== dashboardLoadId) return;
  const cabinet = session.cabinet;
  const data = cabinet.registered_profile;
  state.currentUser = session.user;
  state.registeredProfile = data;
  state.selectedResearcherId = null;
  state.dashboardContext = "cabinet";
  setAccessMode(true);
  renderResearchers();
  renderRegisteredUserCard(data);
  renderDashboard(cabinet.personalized_dashboard);
  renderCabinetMeta(cabinet);
  byId("snapshot-button").disabled = false;
  await loadPromotionProgress();
}

async function performSearch() {
  const query = byId("search-input").value.trim();
  if (!query) {
    return;
  }
  const requestId = ++searchLoadId;
  byId("search-hint").textContent = "Searching public profiles...";
  const payload = await fetchJson(`/researchers/search?q=${encodeURIComponent(query)}`);
  if (requestId !== searchLoadId) return;
  state.researchers = payload;
  renderResearchers();
  byId("search-hint").textContent = payload.length
    ? "Choose a researcher to explore their profile. Your personal workspace is kept separately."
    : "No researchers found. Try a different spelling or a full name.";
}

async function selectResearcher(researcherId) {
  const requestId = ++dashboardLoadId;
  prepareDashboard();
  const dashboard = await fetchJson(
    `/researchers/${encodeURIComponent(researcherId)}/dashboard?${dashboardQueryParams()}`
  );
  if (requestId !== dashboardLoadId) return;
  state.selectedResearcherId = researcherId;
  state.dashboardContext = "researcher";
  renderResearchers();
  renderDashboard(dashboard);
  byId("promotion-tracker-list").innerHTML = `
    <article class="promotion-card promotion-actions-card">
      <span>Comparison mode</span>
      <strong>Read-only</strong>
      <p>Promotion actions are generated only for the registered personal cabinet.</p>
    </article>
  `;
  byId("snapshot-button").disabled = true;
  byId("promotion-progress-list").replaceChildren();
  byId("connected-profile-list").replaceChildren();
  byId("project-list").replaceChildren();
  byId("recommendation-summary-list").replaceChildren();
  byId("registered-user-card").classList.add("hidden");
  byId("current-user-summary").textContent = "You are exploring a public profile. Return to your workspace to see account details.";
  const note = document.createElement("p");
  note.className = "section-note";
  note.textContent = "Personal account information is available only in your own workspace.";
  ["connected-profile-list", "project-list", "recommendation-summary-list"].forEach((id) => {
    byId(id).append(note.cloneNode(true));
  });
}

async function submitRegistration(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = buildRegistrationPayload(form);
  byId("registration-status").textContent = "Creating your account and connecting your research profile...";

  const response = await fetch("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(response.status === 409
      ? "An account with this email already exists. Please sign in."
      : "We could not create the account. Check your details and try again.");
  }

  const session = await response.json();
  const data = session.profile;
  state.authToken = session.access_token;
  state.currentUser = session.user;
  state.registeredProfile = data;
  window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, session.access_token);
  byId("registration-status").textContent = "Account and profile stored successfully. Access granted.";
  setAccessMode(true);
  await loadAuthenticatedCabinet();
  form.reset();
}

async function submitLogin(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData(form);
  byId("login-status").textContent = "Checking account...";
  const response = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email: String(formData.get("email") || "").trim(),
      password: String(formData.get("password") || ""),
    }),
  });

  if (!response.ok) {
    throw new Error("Login failed. Check email and password.");
  }

  const session = await response.json();
  state.authToken = session.access_token;
  state.currentUser = session.user;
  state.registeredProfile = session.profile;
  window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, session.access_token);
  byId("login-status").textContent = "Signed in. Loading personal cabinet...";
  await loadAuthenticatedCabinet();
  form.reset();
}

async function logout() {
  ++dashboardLoadId;
  ++collaboratorLoadId;
  ++searchLoadId;
  if (state.authToken) {
    await fetch("/auth/logout", {
      method: "POST",
      headers: { Authorization: `Bearer ${state.authToken}` },
    }).catch(() => {});
  }
  state.registeredProfile = null;
  state.currentUser = null;
  state.authToken = null;
  state.researchers = [];
  state.selectedResearcherId = null;
  state.collaboratorScope = "global";
  state.embeddingModel = "specter2";
  state.dashboardContext = "cabinet";
  hasDashboard = false;
  completedActions.clear();
  renderScopeControls(state.collaboratorScope);
  renderEmbeddingControls(state.embeddingModel);
  window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  byId("researcher-list").innerHTML = "";
  byId("researcher-count").textContent = "0 results";
  byId("dashboard-content").classList.add("hidden");
  byId("dashboard-state").classList.remove("hidden");
  byId("dashboard-state").innerHTML = `
    <span class="loading-mark"></span>
    <h3>Preparing your workspace</h3>
    <p>Finding your publications and research recommendations. This may take a moment.</p>
  `;
  byId("publication-search").value = "";
  byId("search-input").value = "";
  byId("login-status").textContent = "";
  byId("registration-status").textContent = "";
  byId("registration-result").replaceChildren();
  byId("registration-result").classList.add("hidden");
  history.replaceState(null, "", "#/login");
  setAccessMode(false);
}

async function changeCollaboratorScope(scope) {
  state.collaboratorScope = scope || "global";
  renderScopeControls(state.collaboratorScope);
  byId("collaborator-list").innerHTML = `
    <article class="feed-card">
      <h4>Refreshing collaborator recommendations</h4>
      <p>Searching OpenAlex with ${scopeLabel(state.collaboratorScope)} scope.</p>
    </article>
  `;
  if (state.dashboardContext === "researcher" && state.selectedResearcherId) {
    await selectResearcher(state.selectedResearcherId);
  } else {
    await loadAuthenticatedCabinet();
  }
}

async function changeEmbeddingModel(model) {
  state.embeddingModel = model || "specter2";
  renderEmbeddingControls(state.embeddingModel);
  byId("recommended-work-list").innerHTML = `
    <article class="feed-card transformer-card">
      <h4>Refreshing ${embeddingModelLabel(state.embeddingModel)} recommendations</h4>
      <p>Building a query embedding and searching PostgreSQL/pgvector.</p>
    </article>
  `;
  if (state.dashboardContext === "researcher" && state.selectedResearcherId) {
    await selectResearcher(state.selectedResearcherId);
  } else {
    await loadAuthenticatedCabinet();
  }
}

async function bootstrapAccess() {
  const token = window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
  if (!token) {
    setAccessMode(false);
    return;
  }
  try {
    state.authToken = token;
    await loadAuthenticatedCabinet();
  } catch (error) {
    if (error.status === 401) {
      state.authToken = null;
      state.currentUser = null;
      window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
      setAccessMode(false);
    }
    showError(error);
  }
}

byId("researcher-search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runTask(byId("search-button"), performSearch);
});
byId("registration-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runTask(event.currentTarget.querySelector('[type="submit"]'), () => submitRegistration(event));
});
byId("login-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runTask(event.currentTarget.querySelector('[type="submit"]'), () => submitLogin(event));
});
byId("my-cabinet-button").addEventListener("click", () => runTask(byId("my-cabinet-button"), async () => {
  await loadAuthenticatedCabinet();
  Workspace.navigate("overview");
}));
document.querySelector(".sidebar-account").addEventListener("click", (event) => {
  event.preventDefault();
  runTask(null, async () => {
    if (state.dashboardContext === "researcher") await loadAuthenticatedCabinet();
    Workspace.navigate("profile");
  });
});
byId("snapshot-button").addEventListener("click", () => runTask(byId("snapshot-button"), createPromotionSnapshot));
byId("switch-profile-button").addEventListener("click", () => runTask(byId("switch-profile-button"), logout));
document.querySelectorAll("[data-collaborator-scope]").forEach((button) => {
  button.addEventListener("click", () => refreshRecommendations(() => changeCollaboratorScope(button.dataset.collaboratorScope)));
});
document.querySelectorAll("[data-embedding-model]").forEach((button) => {
  button.addEventListener("click", () => refreshRecommendations(() => changeEmbeddingModel(button.dataset.embeddingModel)));
});
byId("close-collaborator-profile").addEventListener("click", () => {
  byId("collaborator-profile-panel").close();
});
byId("collaborator-profile-panel").addEventListener("close", () => { ++collaboratorLoadId; });
byId("collaborator-profile-panel").addEventListener("click", (event) => {
  const panel = event.currentTarget;
  const bounds = panel.getBoundingClientRect();
  if (event.target === panel && (event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom)) panel.close();
});
byId("retry-button").addEventListener("click", () => runTask(byId("retry-button"), async () => {
  if (state.dashboardContext === "researcher" && state.selectedResearcherId) await selectResearcher(state.selectedResearcherId);
  else await loadAuthenticatedCabinet();
}));

async function refreshRecommendations(task) {
  const controls = [...document.querySelectorAll("[data-collaborator-scope], [data-embedding-model]")];
  if (controls.some((button) => button.disabled)) return;
  controls.forEach((button) => { button.disabled = true; });
  try {
    await task();
  } catch (error) {
    showError(error);
  } finally {
    controls.forEach((button) => { button.disabled = false; });
    byId("dashboard-content").setAttribute("aria-busy", "false");
  }
}

function showError(error) {
  if (error.status === 401) {
    ++dashboardLoadId;
    state.authToken = null;
    state.currentUser = null;
    state.registeredProfile = null;
    hasDashboard = false;
    completedActions.clear();
    window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
    byId("dashboard-content").classList.add("hidden");
    setAccessMode(false);
  }
  if (state.authToken) {
    byId("app-notice-text").textContent = error.message;
    byId("app-notice").classList.remove("hidden");
    byId("dashboard-state").classList.add("hidden");
  }
  byId("registration-status").textContent = error.message;
  byId("login-status").textContent = error.message;
}

bootstrapAccess().catch(showError);
