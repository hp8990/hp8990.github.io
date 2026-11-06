(function () {
  "use strict";

  var SITE = window.SITE || {};
  var PROJECTS = window.PROJECTS || [];
  var TIMELINE = window.TIMELINE || [];
  var ARTICLES = window.ARTICLES || [];
  var PINNED = window.WRITING_PINNED || [];
  var VALUE = window.VALUE || { points: [], results: [] };
  var I18N = window.I18N || {};
  var root = document.documentElement;

  function store(key, val) {
    try {
      if (val === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, val);
    } catch (e) { return null; }
  }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  /* ---------------- Language ---------------- */
  var lang = root.lang === "fr" ? "fr" : "en";
  function t(key) { return (I18N[lang] && I18N[lang][key]) || (I18N.en && I18N.en[key]) || ""; }
  function pick(v) { return v == null ? "" : (typeof v === "string" ? v : (v[lang] || v.en || "")); }

  function applyLang() {
    root.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach(function (n) {
      var v = t(n.getAttribute("data-i18n")); if (v) n.textContent = v;
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(function (n) {
      n.setAttribute("placeholder", t(n.getAttribute("data-i18n-placeholder")));
    });
    document.querySelectorAll("[data-i18n-alt]").forEach(function (n) {
      n.setAttribute("alt", t(n.getAttribute("data-i18n-alt")));
    });
    document.querySelectorAll(".lang-switch button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-lang") === lang));
    });
    document.querySelectorAll("[data-i18n-aria]").forEach(function (n) {
      n.setAttribute("aria-label", t(n.getAttribute("data-i18n-aria")));
    });
    document.title = t("meta.title");
    var md = document.querySelector('meta[name="description"]');
    if (md) md.setAttribute("content", t("meta.desc"));
    var tt = document.getElementById("themeToggle");
    if (tt) tt.setAttribute("aria-label", t("theme.label"));
    // CV buttons only appear when a CV path is set in content.js
    var cv = pick(SITE.links && SITE.links.cv);
    document.querySelectorAll(".cv-link").forEach(function (a) {
      if (cv) { a.setAttribute("href", cv); a.setAttribute("hreflang", lang); }
      var host = a.closest(".cv-item") || a;
      host.hidden = !cv;
    });
    document.querySelectorAll(".cv-link .file-info").forEach(function (n) {
      var size = pick(SITE.cvSize) || "";
      if (lang === "fr") size = size.replace(/\bKB\b/, "Ko").replace(/\bMB\b/, "Mo");
      n.textContent = "(" + t("pdf") + (size ? ", " + size : "") + ")";
    });
    renderProjects(false);
    renderValue();
    renderTimeline();
    renderRepos();
    renderArticles();
    renderHarness();
    markNewTabs(document);
  }

  /* Links opening a new tab announce it, and show a small arrow */
  function markNewTabs(scope) {
    scope.querySelectorAll('a[target="_blank"]').forEach(function (a) {
      var sr = a.querySelector(".nt-text");
      if (!sr) {
        var icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        icon.setAttribute("class", "nt-icon");
        icon.setAttribute("viewBox", "0 0 12 12");
        icon.setAttribute("aria-hidden", "true");
        icon.setAttribute("focusable", "false");
        icon.innerHTML = '<path d="M4 2h6v6M10 2L3 9"/>';
        sr = el("span", "visually-hidden nt-text");
        var host = a.querySelector(".repo-name") || a;
        host.appendChild(icon);
        host.appendChild(sr);
      }
      sr.textContent = " " + t("newTab");
      sr.setAttribute("lang", lang);
    });
  }
  document.querySelectorAll(".lang-switch button").forEach(function (b) {
    b.addEventListener("click", function () {
      lang = b.getAttribute("data-lang");
      store("lang", lang);
      applyLang();
    });
  });

  /* ---------------- Theme ---------------- */
  var themeBtn = document.getElementById("themeToggle");
  function isDark() {
    var set = root.getAttribute("data-theme");
    if (set) return set === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function syncTheme() { if (themeBtn) themeBtn.setAttribute("aria-pressed", String(isDark())); }
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    store("theme", next);
    syncTheme();
  });
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", syncTheme);
  }
  syncTheme();

  /* ---------------- Projects ---------------- */
  var filter = "all";
  var list = document.getElementById("projects");

  function renderProjects(animate) {
    if (!list) return;
    var side = document.getElementById("sideProjects");
    var sideWrap = document.getElementById("sideWrap");
    list.textContent = "";
    if (side) side.textContent = "";
    var sideShown = 0;
    PROJECTS.forEach(function (p) {
      var isSide = p.kind === "personal" && side;
      var li = el("li", "project");
      li.id = "project-" + p.id;
      li.dataset.track = p.track;
      if (filter !== "all" && p.track !== filter) li.hidden = true;
      if (animate && !li.hidden) li.classList.add("is-in");

      var meta = el("div", "project-meta");
      if (p.year) meta.appendChild(el("span", "yr", p.year));
      var tr = el("span", "p-track");
      var dot = el("span", "track-dot t-" + p.track); dot.setAttribute("aria-hidden", "true");
      tr.appendChild(dot);
      tr.appendChild(document.createTextNode(t("track." + p.track)));
      meta.appendChild(tr);
      if (!isSide) meta.appendChild(el("span", "kind", t("kind." + p.kind)));   // the heading already says it
      li.appendChild(meta);

      var body = el("div", "project-body");
      body.appendChild(el("h3", null, pick(p.title)));
      if (p.org) body.appendChild(el("div", "project-org", pick(p.org)));
      body.appendChild(el("p", null, pick(p.summary)));
      if (p.tags && p.tags.length) {
        var tags = el("ul", "project-tags");
        p.tags.forEach(function (tg) { tags.appendChild(el("li", null, tg)); });
        body.appendChild(tags);
      }
      if (p.links && p.links.length) {
        var links = el("div", "project-links");
        p.links.forEach(function (l) {
          var a = el("a", null, pick(l.label)); a.href = l.href;
          if (/^https?:/.test(l.href)) { a.target = "_blank"; a.rel = "noopener"; }
          links.appendChild(a);
        });
        body.appendChild(links);
      }
      li.appendChild(body);

      if (p.image) {
        var fig = el("figure", "project-figure");
        var img = el("img"); img.src = p.image; img.alt = pick(p.imageAlt); img.loading = "lazy"; img.decoding = "async";
        fig.appendChild(img); li.appendChild(fig);
      }
      if (isSide) { side.appendChild(li); if (!li.hidden) sideShown++; }
      else list.appendChild(li);
    });
    if (sideWrap) sideWrap.hidden = !sideShown;
  }

  document.querySelectorAll("#filters button").forEach(function (b) {
    b.addEventListener("click", function () {
      filter = b.getAttribute("data-filter");
      document.querySelectorAll("#filters button").forEach(function (x) {
        x.setAttribute("aria-pressed", String(x === b));
      });
      renderProjects(true);
      markNewTabs(list);
      var shown = PROJECTS.filter(function (p) { return filter === "all" || p.track === filter; }).length;
      var st = document.getElementById("filterStatus");
      if (st) st.textContent = t("filter.status").replace("{n}", shown).replace("{total}", PROJECTS.length);
    });
  });

  /* ---------------- What I bring ---------------- */
  function renderValue() {
    var pts = document.getElementById("valuePoints");
    var res = document.getElementById("valueResults");
    if (!pts || !res) return;
    pts.textContent = ""; res.textContent = "";
    VALUE.points.forEach(function (v) {
      var li = el("li", "vpoint");
      li.appendChild(el("h3", null, pick(v.title)));
      li.appendChild(el("p", null, pick(v.body)));
      pts.appendChild(li);
    });
    VALUE.results.forEach(function (r) {
      var li = el("li", "result");
      var fig = el("span", "result-figure", lang === "fr" && r.figureFr ? r.figureFr : r.figure);
      var sr = pick(r.figureSr);
      if (sr) {
        fig.setAttribute("aria-hidden", "true");
        li.appendChild(el("span", "visually-hidden", sr + " "));
      }
      li.appendChild(fig);
      li.appendChild(el("span", "result-label", pick(r.label)));
      var ctx = el("span", "result-context");
      if (r.project && document.getElementById("project-" + r.project)) {
        var a = el("a", null, pick(r.context));
        a.href = "#project-" + r.project;
        a.setAttribute("aria-label", pick(r.context) + ", " + t("value.see").toLowerCase());
        ctx.appendChild(a);
      } else {
        ctx.textContent = pick(r.context);
      }
      li.appendChild(ctx);
      res.appendChild(li);
    });
    var sec = document.getElementById("value");
    if (sec) sec.hidden = !VALUE.points.length && !VALUE.results.length;
  }

  /* ---------------- Timeline ---------------- */
  function renderTimeline() {
    var ol = document.getElementById("timeline");
    if (!ol) return;
    ol.textContent = "";
    TIMELINE.forEach(function (it) {
      var li = el("li", "tl" + (it.edu ? " edu" : ""));
      li.appendChild(el("div", "tl-when", pick(it.when)));
      var box = el("div");
      var what = el("div", "tl-what", pick(it.what));
      if (it.edu) what.appendChild(el("span", "tl-edu", t("path.edu")));
      box.appendChild(what);
      box.appendChild(el("div", "tl-where", pick(it.where)));
      li.appendChild(box);
      ol.appendChild(li);
    });
  }

  /* ---------------- Optional contact links ---------------- */
  document.querySelectorAll("[data-link]").forEach(function (li) {
    var href = SITE.links && SITE.links[li.getAttribute("data-link")];
    var a = li.querySelector("a");
    if (href && a) a.href = href;
    li.hidden = !href;
  });

  /* ---------------- Contact email (optional) ---------------- */
  if (SITE.email) {
    var cl = document.getElementById("contactLinks");
    if (cl) {
      var li = el("li"); var a = el("a", null, SITE.email);
      a.href = "mailto:" + SITE.email; li.appendChild(a);
      cl.insertBefore(li, cl.children[1] || null);
    }
  }

  /* ---------------- GitHub repos (live) ---------------- */
  var repoState = { status: "loading", repos: [] };
  function renderRepos() {
    var ul = document.getElementById("repoList");
    if (!ul) return;
    ul.textContent = "";
    if (repoState.status !== "ok" || !repoState.repos.length) {
      var key = repoState.status === "loading" ? "repos.loading" : repoState.status === "error" ? "repos.error" : "repos.empty";
      ul.appendChild(el("li", "repo-status", t(key)));
      return;
    }
    var fmt;
    try { fmt = new Intl.DateTimeFormat(lang === "fr" ? "fr-FR" : "en-GB", { day: "numeric", month: "short", year: "numeric" }); } catch (e) {}
    repoState.repos.forEach(function (r) {
      var li = el("li", "repo");
      var a = el("a"); a.href = r.html_url; a.target = "_blank"; a.rel = "noopener";
      a.appendChild(el("span", "repo-name", r.name));
      a.appendChild(el("span", "repo-desc", r.description || ""));
      var meta = el("span", "repo-meta");
      if (r.language) meta.appendChild(el("span", null, r.language));
      var d = new Date(r.pushed_at);
      meta.appendChild(el("span", null, t("repos.updated") + " " + (fmt ? fmt.format(d) : d.toISOString().slice(0, 10))));
      a.appendChild(meta);
      li.appendChild(a); ul.appendChild(li);
    });
    markNewTabs(ul);
  }
  function loadRepos() {
    var user = SITE.githubUser;
    var pick = (SITE.githubRepos || []).map(function (n) { return String(n).toLowerCase(); });
    var sec = document.getElementById("repos");
    if (!user || !pick.length) { if (sec) sec.hidden = true; return; }   // hidden unless repos are chosen
    if (sec) sec.hidden = false;
    var all = document.getElementById("reposAll");
    if (all) all.href = "https://github.com/" + encodeURIComponent(user);
    if (!window.fetch) { repoState.status = "error"; renderRepos(); return; }
    var cacheKey = "repos:" + user;
    try {
      var cached = JSON.parse(sessionStorage.getItem(cacheKey) || "null");
      if (cached && Date.now() - cached.at < 30 * 60 * 1000) {
        repoState = { status: cached.repos.length ? "ok" : "empty", repos: cached.repos };
        renderRepos(); return;
      }
    } catch (e) {}
    fetch("https://api.github.com/users/" + encodeURIComponent(user) + "/repos?sort=pushed&per_page=30", {
      headers: { Accept: "application/vnd.github+json" }
    })
      .then(function (res) { if (!res.ok) throw new Error(res.status); return res.json(); })
      .then(function (data) {
        var repos = (Array.isArray(data) ? data : [])
          .filter(function (r) { return pick.indexOf(r.name.toLowerCase()) !== -1; })
          .sort(function (a, b) { return pick.indexOf(a.name.toLowerCase()) - pick.indexOf(b.name.toLowerCase()); })
          .map(function (r) { return { name: r.name, html_url: r.html_url, description: r.description, language: r.language, pushed_at: r.pushed_at }; });
        repoState = { status: repos.length ? "ok" : "empty", repos: repos };
        try { sessionStorage.setItem(cacheKey, JSON.stringify({ at: Date.now(), repos: repos })); } catch (e) {}
        renderRepos();
      })
      .catch(function () { repoState.status = "error"; renderRepos(); });
  }

  /* ---------------- Articles (DEV, live with fallback) ---------------- */
  var articles = ARTICLES.slice();
  function renderArticles() {
    var ol = document.getElementById("articles");
    if (!ol) return;
    ol.textContent = "";
    var fmt;
    try { fmt = new Intl.DateTimeFormat(lang === "fr" ? "fr-FR" : "en-GB", { day: "numeric", month: "long", year: "numeric" }); } catch (e) {}
    var all = articles.concat(PINNED).slice().sort(function (x, y) {
      return String(y.date).slice(0, 10) < String(x.date).slice(0, 10) ? -1 : 1;
    });
    all.forEach(function (a) {
      var itemLang = a.lang || "en";            // DEV articles are written in English
      var li = el("li", "article");
      li.setAttribute("lang", itemLang);
      var meta = el("div", "article-meta");
      var d = new Date(a.date + (a.date.length === 10 ? "T12:00:00" : ""));
      var time = el("time", null, fmt ? fmt.format(d) : a.date.slice(0, 10));
      time.setAttribute("datetime", a.date.slice(0, 10));
      time.setAttribute("lang", lang);
      meta.appendChild(time);
      if (itemLang !== lang) {
        var note = el("span", "article-lang", t(itemLang === "fr" ? "writing.inFrench" : "writing.inEnglish"));
        note.setAttribute("lang", lang); meta.appendChild(note);
      }
      if (!/^https?:/.test(a.url)) {
        var kind = el("span", "article-lang", t("writing.onSite")); kind.setAttribute("lang", lang); meta.appendChild(kind);
      }
      li.appendChild(meta);
      var body = el("div", "article-body");
      var h = el("h3");
      var link = el("a", null, a.title); link.href = a.url;
      if (/^https?:/.test(a.url)) { link.target = "_blank"; link.rel = "noopener"; }
      h.appendChild(link); body.appendChild(h);
      if (a.summary) {
        var sum = el("p", null, pick(a.summary));
        if (typeof a.summary !== "string") sum.setAttribute("lang", lang);   // our own summary follows the page language
        body.appendChild(sum);
      }
      if (a.tags && a.tags.length) {
        var tags = el("ul", "project-tags");
        a.tags.slice(0, 4).forEach(function (tg) { tags.appendChild(el("li", null, "#" + tg)); });
        body.appendChild(tags);
      }
      li.appendChild(body);
      ol.appendChild(li);
    });
    markNewTabs(ol);
  }
  function loadArticles() {
    var user = SITE.devtoUser;
    if (!user || !window.fetch) return;
    var cacheKey = "devto:" + user;
    function use(list) {
      if (!list || !list.length) return;
      articles = list; renderArticles();
    }
    try {
      var cached = JSON.parse(sessionStorage.getItem(cacheKey) || "null");
      if (cached && Date.now() - cached.at < 30 * 60 * 1000) { use(cached.list); return; }
    } catch (e) {}
    fetch("https://dev.to/api/articles?username=" + encodeURIComponent(user) + "&per_page=6")
      .then(function (res) { if (!res.ok) throw new Error(res.status); return res.json(); })
      .then(function (data) {
        var list = (Array.isArray(data) ? data : []).map(function (a) {
          return {
            title: a.title,
            url: a.url,
            date: a.published_at || a.published_timestamp || "",
            tags: a.tag_list || [],
            summary: a.description || ""
          };
        }).filter(function (a) { return a.title && a.url; });
        try { sessionStorage.setItem(cacheKey, JSON.stringify({ at: Date.now(), list: list })); } catch (e) {}
        use(list);
      })
      .catch(function () { /* keep the built-in list */ });
  }

  /* ---------------- Agent harness demo ---------------- */
  var HARNESS = window.HARNESS || [];
  var hx = { idx: -1, pending: false, log: [], ran: 0, stopped: 0 };
  var hxStep = document.getElementById("hxStep");
  var hxLog = document.getElementById("hxLog");
  var hxNext = document.getElementById("hxNext");
  var hxApprove = document.getElementById("hxApprove");
  var hxReject = document.getElementById("hxReject");
  var hxRestart = document.getElementById("hxRestart");
  var ICON = { allow: "✓", approval: "?", steer: "↻", block: "✕", approved: "✓", rejected: "✕" };

  function clock() {
    var d = new Date();
    return [d.getHours(), d.getMinutes(), d.getSeconds()].map(function (n) { return (n < 10 ? "0" : "") + n; }).join(":");
  }
  function toolName(call) { return String(call).split("(")[0]; }

  // log entries keep keys, not text, so they re-render in the current language
  function addLog(entry) { entry.time = clock(); hx.log.push(entry); }

  function renderHarness() {
    if (!hxStep || !hxLog) return;
    hxStep.textContent = "";
    var step = HARNESS[hx.idx];
    if (hx.idx < 0) {
      hxStep.appendChild(el("p", "hx-intro", t("harness.intro")));
    } else {
      hxStep.appendChild(el("p", "hx-count", t("harness.step").replace("{n}", hx.idx + 1).replace("{total}", HARNESS.length)));
      var prop = el("p", "hx-proposal");
      prop.appendChild(el("span", "hx-who", t("harness.proposes")));
      prop.appendChild(el("span", "hx-action", pick(step.action)));
      hxStep.appendChild(prop);
      var code = el("code", "hx-tool", step.tool);
      code.setAttribute("lang", "en");
      hxStep.appendChild(code);
      var outcome = step.outcome || step.verdict;
      var v = el("div", "hx-verdict v-" + outcome);
      var ic = el("span", "hx-icon", ICON[outcome]); ic.setAttribute("aria-hidden", "true");
      v.appendChild(ic);
      var vt = el("div", "hx-vtext");
      var label = outcome === "approved" ? t("harness.approvedBy") : outcome === "rejected" ? t("harness.rejectedBy") : t("harness.v." + step.verdict);
      vt.appendChild(el("strong", null, label));
      vt.appendChild(el("span", null, " " + pick(step.reason)));
      if (step.verdict === "steer") {
        var rw = el("span", "hx-rewrite", t("harness.rewritten") + " ");
        var c2 = el("code", null, step.steerTo); c2.setAttribute("lang", "en");
        rw.appendChild(c2); vt.appendChild(rw);
      }
      var res = outcome === "approved" ? step.approved : outcome === "rejected" ? step.rejected : step.result;
      if (res && outcome !== "approval") vt.appendChild(el("span", "hx-result", pick(res)));
      v.appendChild(vt);
      hxStep.appendChild(v);
    }
    // controls
    var finished = HARNESS.length > 0 && hx.idx >= HARNESS.length - 1 && !hx.pending;
    if (finished) hxStep.appendChild(el("p", "hx-done", t("harness.done").replace("{ran}", hx.ran).replace("{stopped}", hx.stopped)));
    hxNext.hidden = hx.pending || finished;
    hxNext.textContent = hx.idx < 0 ? t("harness.start") : t("harness.next");
    hxApprove.hidden = hxReject.hidden = !hx.pending;
    // log
    hxLog.textContent = "";
    if (!hx.log.length) hxLog.appendChild(el("li", "hx-log-empty", t("harness.logEmpty")));
    hx.log.forEach(function (e) {
      var li = el("li", "hx-entry v-" + e.kind);
      var tm = el("time", null, e.time); li.appendChild(tm);
      var c = el("code", null, e.tool); c.setAttribute("lang", "en"); li.appendChild(c);
      var txt = e.kind === "approved" ? t("harness.approvedBy") : e.kind === "rejected" ? t("harness.rejectedBy") : t("harness.v." + e.kind);
      var out = el("span", "hx-outcome", txt + (e.res ? ". " + pick(e.res) : "."));
      li.appendChild(out);
      hxLog.appendChild(li);
    });
    return finished;
  }

  function hxAdvance() {
    if (hx.pending) return;
    hx.idx++;
    var step = HARNESS[hx.idx];
    if (!step) { hx.idx = HARNESS.length - 1; return; }
    step.outcome = null;
    if (step.verdict === "approval") {
      hx.pending = true;
      renderHarness();
      hxApprove.focus();
      return;
    }
    if (step.verdict === "allow") hx.ran++;
    else hx.stopped++;
    if (step.verdict === "steer") hx.ran++;
    addLog({ kind: step.verdict, tool: toolName(step.verdict === "steer" ? step.steerTo : step.tool), res: step.result || null });
    var finished = renderHarness();
    if (finished) hxRestart.focus();
  }
  function hxDecide(ok) {
    var step = HARNESS[hx.idx];
    hx.pending = false;
    step.outcome = ok ? "approved" : "rejected";
    if (ok) hx.ran++; else hx.stopped++;
    addLog({ kind: step.outcome, tool: toolName(step.tool), res: ok ? step.approved : step.rejected });
    var finished = renderHarness();
    (finished ? hxRestart : hxNext).focus();
  }
  function hxReset() {
    HARNESS.forEach(function (s) { s.outcome = null; });
    hx = { idx: -1, pending: false, log: [], ran: 0, stopped: 0 };
    renderHarness();
    hxNext.focus();
  }
  if (hxNext) {
    hxNext.addEventListener("click", hxAdvance);
    hxApprove.addEventListener("click", function () { hxDecide(true); });
    hxReject.addEventListener("click", function () { hxDecide(false); });
    hxRestart.addEventListener("click", hxReset);
  }

  /* ---------------- Init ---------------- */
  var yr = document.getElementById("year");
  if (yr) yr.textContent = new Date().getFullYear();
  applyLang();
  loadRepos();
  loadArticles();
})();
