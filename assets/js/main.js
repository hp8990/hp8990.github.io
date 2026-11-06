(function () {
  "use strict";

  var SITE = window.SITE || {};
  var PROJECTS = window.PROJECTS || [];
  var TIMELINE = window.TIMELINE || [];
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
    var tt = document.getElementById("themeToggle");
    if (tt) tt.setAttribute("aria-label", t("theme.toggle"));
    var plot = document.getElementById("demoPlot");
    if (plot) plot.setAttribute("aria-label", t("chart.label"));
    renderProjects(false);
    renderTimeline();
    renderRepos();
    drawDemo();
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
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    store("theme", next);
  });

  /* ---------------- Projects ---------------- */
  var filter = "all";
  var list = document.getElementById("projects");

  function renderProjects(animate) {
    if (!list) return;
    list.textContent = "";
    PROJECTS.forEach(function (p) {
      var li = el("li", "project");
      li.id = "project-" + p.id;
      li.dataset.track = p.track;
      if (filter !== "all" && p.track !== filter) li.hidden = true;
      if (animate && !li.hidden) li.classList.add("is-in");

      var meta = el("div", "project-meta");
      if (p.year) meta.appendChild(el("span", "yr", p.year));
      var dot = el("span", "track-dot t-" + p.track); dot.setAttribute("aria-hidden", "true");
      meta.appendChild(dot);
      meta.appendChild(document.createTextNode(t("kind." + p.kind)));
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
      list.appendChild(li);
    });
  }

  document.querySelectorAll("#filters button").forEach(function (b) {
    b.addEventListener("click", function () {
      filter = b.getAttribute("data-filter");
      document.querySelectorAll("#filters button").forEach(function (x) {
        x.setAttribute("aria-pressed", String(x === b));
      });
      renderProjects(true);
    });
  });

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
  }
  function loadRepos() {
    var user = SITE.githubUser;
    if (!user || !window.fetch) { repoState.status = "error"; renderRepos(); return; }
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
          .filter(function (r) { return !r.fork && !r.archived && r.name.toLowerCase() !== (user + ".github.io").toLowerCase(); })
          .slice(0, 6)
          .map(function (r) { return { name: r.name, html_url: r.html_url, description: r.description, language: r.language, pushed_at: r.pushed_at }; });
        repoState = { status: repos.length ? "ok" : "empty", repos: repos };
        try { sessionStorage.setItem(cacheKey, JSON.stringify({ at: Date.now(), repos: repos })); } catch (e) {}
        renderRepos();
      })
      .catch(function () { repoState.status = "error"; renderRepos(); });
  }

  /* ---------------- Keystroke demo ---------------- */
  var input = document.getElementById("demoInput");
  var plot = document.getElementById("demoPlot");
  var MAX = 18;
  var strokes = [];        // { key, dwell, flight }
  var down = {};           // code -> { t, idx }
  var lastUp = null;
  var lastInput = null;
  var sawKeydown = false;
  var freshIdx = -1;
  var SVGNS = "http://www.w3.org/2000/svg";

  function svg(tag, attrs, text) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function label(k) {
    if (k === " ") return "␣";
    if (k.length === 1) return k;
    return "";
  }
  function avg(arr) {
    var v = arr.filter(function (x) { return x != null; });
    return v.length ? Math.round(v.reduce(function (a, b) { return a + b; }, 0) / v.length) : null;
  }

  function drawDemo() {
    if (!plot) return;
    var W = Math.max(260, Math.round(plot.clientWidth || 480));
    var H = Math.max(140, Math.round(plot.clientHeight || 200));
    plot.setAttribute("viewBox", "0 0 " + W + " " + H);
    while (plot.firstChild) plot.removeChild(plot.firstChild);

    var padL = 34, padR = 6, padT = 8, padB = 22;
    var iw = W - padL - padR, ih = H - padT - padB;
    var vals = [];
    strokes.forEach(function (s) { if (s.dwell != null) vals.push(s.dwell); if (s.flight != null) vals.push(s.flight); });
    var top = Math.max(300, Math.ceil((Math.max.apply(null, vals.concat([0])) * 1.1) / 100) * 100);
    top = Math.min(top, 1200);
    var y = function (v) { return padT + ih - (Math.min(v, top) / top) * ih; };
    var step = top <= 400 ? 100 : top <= 800 ? 200 : 300;

    for (var g = 0; g <= top; g += step) {
      plot.appendChild(svg("line", { "class": "grid", x1: padL, x2: W - padR, y1: y(g), y2: y(g) }));
      plot.appendChild(svg("text", { "class": "axis-label", x: padL - 6, y: y(g) + 3, "text-anchor": "end" }, g === 0 ? "0" : g + "ms"));
    }

    if (!strokes.length) {
      plot.appendChild(svg("text", { "class": "empty", x: padL + iw / 2, y: padT + ih / 2, "text-anchor": "middle" }, t("demo.empty")));
    } else {
      var slot = iw / MAX;
      var bw = Math.max(4, Math.min(18, slot * 0.42));
      var pts = [];
      strokes.forEach(function (s, i) {
        var cx = padL + slot * i + slot / 2;
        if (s.dwell != null) {
          var r = svg("rect", { "class": "bar-dwell" + (i === freshIdx ? " fresh" : ""), x: cx - bw / 2, y: y(s.dwell), width: bw, height: Math.max(1, padT + ih - y(s.dwell)), rx: 2 });
          plot.appendChild(r);
        }
        if (s.flight != null) pts.push([cx, y(s.flight)]);
        var lb = label(s.key);
        if (lb) plot.appendChild(svg("text", { "class": "key-label", x: cx, y: H - 6, "text-anchor": "middle" }, lb));
      });
      if (pts.length > 1) {
        plot.appendChild(svg("polyline", { "class": "line-flight", points: pts.map(function (p) { return p.join(","); }).join(" ") }));
      }
      pts.forEach(function (p) { plot.appendChild(svg("circle", { "class": "dot-flight", cx: p[0], cy: p[1], r: 3.5 })); });
    }

    var ad = avg(strokes.map(function (s) { return s.dwell; }));
    var af = avg(strokes.map(function (s) { return s.flight; }));
    document.getElementById("avgDwell").textContent = ad == null ? "–" : ad + " ms";
    document.getElementById("avgFlight").textContent = af == null ? "–" : af + " ms";
  }

  function push(s) {
    strokes.push(s);
    if (strokes.length > MAX) {
      strokes.shift();
      for (var c in down) if (down[c].idx != null) down[c].idx--;
    }
    freshIdx = strokes.length - 1;
  }

  if (input && plot) {
    input.addEventListener("keydown", function (e) {
      if (e.repeat || e.isComposing || e.keyCode === 229 || !e.key) return;
      if (e.key.length !== 1 && e.key !== "Backspace") return;
      sawKeydown = true;
      var now = performance.now();
      var flight = lastUp != null ? Math.round(now - lastUp) : null;
      if (flight != null && flight > 2000) flight = null;   // long pauses aren't rhythm
      push({ key: e.key === "Backspace" ? "⌫" : e.key, dwell: null, flight: flight });
      down[e.code || e.key] = { t: now, idx: strokes.length - 1 };
      drawDemo();
    });
    input.addEventListener("keyup", function (e) {
      var d = down[e.code || e.key];
      if (!d) return;
      var now = performance.now();
      if (d.idx >= 0 && strokes[d.idx]) strokes[d.idx].dwell = Math.round(now - d.t);
      delete down[e.code || e.key];
      lastUp = now;
      freshIdx = d.idx;
      drawDemo();
    });
    // Touch keyboards: no reliable key up/down, so measure the gap between inputs only
    input.addEventListener("input", function (e) {
      if (sawKeydown) { sawKeydown = false; return; }
      var now = performance.now();
      var ch = (e.data || "").slice(-1);
      var flight = lastInput != null ? Math.round(now - lastInput) : null;
      if (flight != null && flight > 2000) flight = null;
      lastInput = now;
      if (!ch) return;
      push({ key: ch, dwell: null, flight: flight });
      var tn = document.getElementById("touchNote"); if (tn) tn.hidden = false;
      drawDemo();
    });
    document.getElementById("demoReset").addEventListener("click", function () {
      strokes = []; down = {}; lastUp = null; lastInput = null; freshIdx = -1;
      input.value = ""; input.focus(); drawDemo();
    });
    var rt;
    window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(drawDemo, 120); });
  }

  /* ---------------- Init ---------------- */
  var yr = document.getElementById("year");
  if (yr) yr.textContent = new Date().getFullYear();
  applyLang();
  loadRepos();
})();
