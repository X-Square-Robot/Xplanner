/* Renders the auto-generated rollout gallery and the reported-results sections.
   All strings originate from generated JSON, so every dynamic value is inserted
   as a text node rather than markup. */

(function () {
  "use strict";

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function getJSON(path) {
    return fetch(path, { cache: "no-cache" }).then(function (response) {
      if (!response.ok) throw new Error(path + " -> " + response.status);
      return response.json();
    });
  }

  function formatValue(value) {
    return typeof value === "number" ? value.toFixed(2) : null;
  }

  /* ---------- scroll reveal ----------
     The .js class turns the hidden state on only once this script is actually
     running, so the page never depends on the animation to be readable. */

  document.documentElement.classList.add("js");

  var observer = null;
  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-in");
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
  }

  function watch(node) {
    if (!node || node.classList.contains("is-in")) return node;
    if (observer) observer.observe(node);
    else node.classList.add("is-in");
    return node;
  }

  function watchAll() {
    document.querySelectorAll(".reveal").forEach(watch);
  }

  /* Safety net: if the observer never delivers — a prerender, an exotic
     engine, a print pass — reveal whatever is already on screen instead of
     leaving the page blank. */
  window.addEventListener("load", function () {
    window.setTimeout(function () {
      document.querySelectorAll(".reveal:not(.is-in)").forEach(function (node) {
        if (node.getBoundingClientRect().top < window.innerHeight) {
          node.classList.add("is-in");
        }
      });
    }, 1000);
  });

  /* ---------- rollout gallery ---------- */

  function renderMedia(demo) {
    var card = el("div", "video-card");
    if (demo.aspect_ratio) card.style.aspectRatio = demo.aspect_ratio;

    if (demo.video) {
      var video = document.createElement("video");
      video.controls = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.setAttribute("aria-label", demo.name + (demo.kind === "recorded_benchmark_episode"
        ? ": benchmark episode recording" : ": recorded observations and model predictions"));
      if (demo.poster) video.poster = demo.poster;

      var source = document.createElement("source");
      source.src = demo.video;
      source.type = "video/mp4";
      video.appendChild(source);
      video.appendChild(document.createTextNode("Your browser cannot play this rollout video."));
      card.appendChild(video);
      return card;
    }

    var placeholder = el("div", "video-placeholder");
    placeholder.appendChild(el("div", "spinner"));
    placeholder.appendChild(el("span", null, "Rollout video rendering…"));
    card.appendChild(placeholder);
    return card;
  }

  function renderCaption(demo) {
    var caption = el("div", "clip-task");
    caption.appendChild(el("span", null, demo.task));
    if (demo.record) {
      var record = el("a", null, "Raw record");
      record.href = demo.record;
      caption.appendChild(record);
    }
    if (demo.annotated_video || demo.video) {
      var full = el("a", null, "Full episode");
      full.href = demo.annotated_video || demo.video;
      caption.appendChild(full);
    }
    return caption;
  }

  function renderPlan(demo) {
    if (!demo.plan || !demo.plan.length) return null;

    var details = el("details", "clip-plan");
    var count = demo.plan.length;
    details.appendChild(el("summary", null,
      "Model's opening plan · " + count + (count === 1 ? " action" : " actions")));

    var list = document.createElement("ol");
    demo.plan.forEach(function (step) { list.appendChild(el("li", null, step)); });
    details.appendChild(list);
    return details;
  }

  function renderContext(demo, video, observations, context) {
    if (!video || !demo.states || !demo.states.length || demo.context_variant !== "with_memory_with_initial") return renderPlan(demo);
    var panel = el("div", "planning-context");
    var chartGrid = el("div", "chart-grid");
    var offset = demo.recording_offset_seconds || 0;
    var duration = Math.max((demo.duration_seconds || 0) - offset,
      demo.states[demo.states.length - 1].time_seconds, 1);
    var plan = demo.plan || [];
    var actions = demo.states.map(function (state) {
      var current = ((state.output && state.output.predictions) || []).filter(function (item) {
        return item.role === "current";
      })[0];
      return current && current.action && current.action.caption || null;
    });
    var predictionPair = el("div", "prediction-pair");
    var currentUnit = el("section", "prediction-unit");
    currentUnit.appendChild(el("h4", null, "Current action"));
    var currentCaption = el("p", "action-caption");
    var currentProgress = el("progress", "action-progress");
    currentProgress.max = 100;
    currentProgress.setAttribute("aria-label", "Current action progress");
    currentUnit.append(currentProgress, currentCaption);
    var nextUnit = el("section", "prediction-unit next-unit");
    nextUnit.appendChild(el("h4", null, "Next action"));
    var nextCaption = el("p", "action-caption");
    nextUnit.appendChild(nextCaption);
    predictionPair.append(currentUnit, nextUnit);
    observations.appendChild(predictionPair);

    var planSection = el("section", "initial-plan-context");
    planSection.appendChild(el("h4", null, "Initial plan"));
    var planList = el("ol", "context-list plan-list");
    plan.forEach(function (step) { planList.appendChild(el("li", null, step)); });
    planSection.appendChild(planList);
    context.appendChild(planSection);
    var longSection = el("section", "long-memory-context");
    longSection.appendChild(el("h4", null, "Long memory"));
    var memoryList = el("ol", "context-list long-memory-list");
    var emptyMemory = el("p", "memory-empty", "No prior predictions");
    longSection.append(memoryList, emptyMemory);
    context.appendChild(longSection);
    var shortSection = el("section", "short-memory-context");
    shortSection.appendChild(el("h4", null, "Short memory"));
    var shortCaption = el("p", "short-memory-caption");
    shortSection.appendChild(shortCaption);
    context.appendChild(shortSection);
    var progress = demo.states.map(function (state) {
      var value = state.output && state.output.task_progress_percent;
      return typeof value === "number" ? value : null;
    });
    var memory = demo.states.map(function (state) {
      return ((state.memory_input && state.memory_input.long_memory) || []).length;
    });
    var cursors = [];
    function svgNode(tag, attributes, label) {
      var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.keys(attributes || {}).forEach(function (key) { node.setAttribute(key, attributes[key]); });
      if (label !== undefined) node.textContent = label;
      return node;
    }
    function xAt(time) { return 34 + 374 * time / duration; }
    function seek(time) {
      video.pause();
      video.currentTime = time + offset + 0.5 / (demo.fps || 25);
    }
    function chart(title, className, values, maximum, mode, suffix) {
      var card = el("section", "chart-panel " + className);
      card.appendChild(el("h4", null, title));
      var svg = svgNode("svg", { viewBox: "0 0 420 164", role: "slider", tabindex: 0,
        "aria-label": title + " over recording time", "aria-valuemin": 0, "aria-valuemax": duration, "aria-valuenow": 0 });
      function yAt(value) { return 134 - 112 * value / maximum; }
      [0, maximum / 2, maximum].forEach(function (value) {
        svg.appendChild(svgNode("line", { x1: 34, x2: 408, y1: yAt(value), y2: yAt(value), class: "chart-gridline" }));
        svg.appendChild(svgNode("text", { x: 28, y: yAt(value) + 4, "text-anchor": "end", class: "chart-label" },
          Number(value.toFixed(1)) + suffix));
      });
      [0, duration / 2, duration].forEach(function (time) {
        svg.appendChild(svgNode("text", { x: xAt(time), y: 157, "text-anchor": time === 0 ? "start" : time === duration ? "end" : "middle", class: "chart-label" },
          Number(time.toFixed(1)) + " s"));
      });
      var path = "";
      var previous = null;
      values.forEach(function (value, i) {
        if (value === null) { previous = null; return; }
        var x = xAt(demo.states[i].time_seconds);
        var y = yAt(value);
        if (mode === "bars") {
          var bar = svgNode("rect", { x: x - 4, y: y, width: 8, height: 134 - y, class: "memory-bar" });
          bar.appendChild(svgNode("title", {}, value + " prior predictions at " + demo.states[i].time_seconds + " s"));
          svg.appendChild(bar);
        } else {
          path += previous === null ? "M " + x + " " + y : mode === "steps" ? " H " + x + " V " + y : " L " + x + " " + y;
          previous = value;
        }
      });
      if (path) svg.appendChild(svgNode("path", { d: path, class: "chart-line" }));
      var cursor = svgNode("line", { x1: 34, x2: 34, y1: 14, y2: 134, class: "chart-cursor" });
      var dot = svgNode("circle", { r: 4, class: "chart-dot" });
      svg.append(cursor, dot);
      cursors.push({ svg: svg, line: cursor, dot: dot, values: values, yAt: yAt });
      svg.addEventListener("click", function (event) {
        var bounds = svg.getBoundingClientRect();
        var time = Math.max(0, Math.min(duration, ((event.clientX - bounds.left) / bounds.width * 420 - 34) / 374 * duration));
        seek(time);
      });
      svg.addEventListener("keydown", function (event) {
        var at = video.currentTime - offset;
        if (event.key === "ArrowLeft") at -= 1;
        else if (event.key === "ArrowRight") at += 1;
        else if (event.key === "Home") at = 0;
        else if (event.key === "End") at = duration;
        else return;
        event.preventDefault();
        seek(Math.max(0, Math.min(duration, at)));
      });
      card.appendChild(svg);
      chartGrid.appendChild(card);
      return card;
    }
    var timeline = el("section", "chart-panel timeline-panel");
    timeline.appendChild(el("h4", null, "Action timeline"));
    timeline.appendChild(el("p", "chart-key", "Model predictions"));
    var predictionTrack = el("div", "prediction-track");
    var actionColors = [];
    actions.forEach(function (action, i) {
      var key = action ? action.trim().toLowerCase().replace(/\s+/g, " ") : "";
      var color = actionColors.indexOf(key);
      if (color < 0) { color = actionColors.length; actionColors.push(key); }
      var next = demo.states[i + 1];
      var length = (next ? next.time_seconds : duration) - demo.states[i].time_seconds;
      var segment = el("button", "prediction-step");
      segment.type = "button";
      segment.style.flexBasis = (100 * length / duration) + "%";
      segment.style.setProperty("--band-color", ["#4268a0", "#0e7d74", "#bb762e", "#c35d6d"][color % 4]);
      segment.setAttribute("aria-label", action || "No current action in model output");
      segment.appendChild(el("span", "plan-tooltip", demo.states[i].time_seconds.toFixed(2) + " s · " + (action || "No current action in model output")));
      segment.addEventListener("click", function () { seek(demo.states[i].time_seconds); });
      predictionTrack.appendChild(segment);
    });
    var timelineCursor = el("span", "timeline-cursor");
    predictionTrack.appendChild(timelineCursor);
    timeline.appendChild(predictionTrack);
    var timeScale = el("div", "timeline-scale");
    timeScale.append(el("span", null, "0 s"), el("span", null, Number(duration.toFixed(1)) + " s"));
    timeline.appendChild(timeScale);
    var planTrack = el("div", "plan-track");
    planTrack.setAttribute("aria-label", "Initial plan: " + plan.length + " actions");
    plan.forEach(function (step, i) {
      var segment = el("span", "plan-step", i + 1);
      segment.tabIndex = 0;
      segment.setAttribute("aria-label", "Initial plan step " + (i + 1) + ": " + step);
      segment.appendChild(el("span", "plan-tooltip", step));
      planTrack.appendChild(segment);
    });
    timeline.appendChild(el("p", "chart-key", "Initial plan"));
    timeline.appendChild(planTrack);
    chartGrid.appendChild(timeline);
    chart("Predicted task progress", "progress-panel", progress, 100, "line", "%");
    chart("Long memory (items)", "memory-panel", memory, Math.max.apply(null, memory.concat([1])), "bars", "");
    panel.appendChild(chartGrid);
    if (demo.moments && demo.moments.length) {
      var moments = el("nav", "clip-moments");
      moments.setAttribute("aria-label", demo.name + " key observations");
      demo.moments.forEach(function (moment) {
        var button = el("button", "moment-seek", moment.label);
        button.type = "button";
        button.setAttribute("aria-label", moment.label);
        button.title = "Seek to " + moment.time_seconds.toFixed(2) + " s";
        button.addEventListener("click", function () { seek(moment.time_seconds); });
        moments.appendChild(button);
      });
      panel.appendChild(moments);
    }
    var lastIndex = null;
    function update(mediaTime) {
      var at = mediaTime - offset;
      var index = -1;
      demo.states.forEach(function (state, i) {
        if (state.time_seconds <= at + 0.000001) index = i;
      });
      panel.dataset.anchorFrame = index < 0 ? "initial" : String(demo.states[index].frame);
      if (lastIndex !== index) {
        lastIndex = index;
        var state = index < 0 ? null : demo.states[index];
        var predictions = (state && state.output && state.output.predictions) || [];
        var current = predictions.filter(function (item) { return item.role === "current"; })[0];
        var next = predictions.filter(function (item) { return item.role === "next"; })[0];
        currentCaption.textContent = current && current.action && current.action.caption || "No current action in model output";
        nextCaption.textContent = next && next.action && next.action.caption || "No next action in model output";
        var value = current && current.action && current.action.progress_percent;
        currentProgress.hidden = typeof value !== "number";
        if (typeof value === "number") currentProgress.value = value;
        var supplied = (state && state.memory_input) || {};
        var history = supplied.long_memory || [];
        memoryList.replaceChildren();
        history.forEach(function (item) { memoryList.appendChild(el("li", null, item.action)); });
        memoryList.hidden = !history.length;
        emptyMemory.hidden = !!history.length;
        var short = supplied.short_memory;
        shortCaption.textContent = short && short.prediction1 && short.prediction1.action && short.prediction1.action.caption || "No previous prediction";
      }
      var x = xAt(Math.max(0, Math.min(at, duration)));
      cursors.forEach(function (cursor) {
        cursor.svg.setAttribute("aria-valuenow", Math.max(0, Math.min(at, duration)).toFixed(2));
        cursor.line.setAttribute("x1", x);
        cursor.line.setAttribute("x2", x);
        var value = index < 0 ? null : cursor.values[index];
        cursor.dot.style.display = value === null ? "none" : "";
        if (value !== null) {
          cursor.dot.setAttribute("cx", xAt(demo.states[index].time_seconds));
          cursor.dot.setAttribute("cy", cursor.yAt(value));
        }
      });
      timelineCursor.style.left = (100 * Math.max(0, Math.min(at, duration)) / duration) + "%";
      predictionTrack.querySelectorAll(".prediction-step").forEach(function (node, i) {
        node.classList.toggle("is-active", index === i);
      });
    }
    var updateFromClock = function () { update(video.currentTime); };
    video.addEventListener("timeupdate", updateFromClock);
    video.addEventListener("seeked", updateFromClock);
    video.addEventListener("loadedmetadata", updateFromClock);
    if (typeof video.requestVideoFrameCallback === "function") {
      function presentedFrame(_now, metadata) {
        update(metadata.mediaTime);
        video.requestVideoFrameCallback(presentedFrame);
      }
      video.requestVideoFrameCallback(presentedFrame);
    }
    update(video.currentTime);
    return panel;
  }

  function renderClip(demo, index) {
    var clip = el("article", "clip reveal");
    clip.style.setProperty("--rise-delay", (index % 2) * 0.09 + "s");
    var heading = el("header", "episode-heading");
    heading.appendChild(el("h3", "clip-heading", demo.name));
    heading.appendChild(el("span", "episode-environment", demo.environment));
    clip.appendChild(heading);
    clip.appendChild(el("p", "episode-playback-note", demo.playback_note || "Recorded simulation with offline X-Planner predictions."));
    clip.appendChild(renderMedia(demo));
    clip.appendChild(renderCaption(demo));
    var video = clip.querySelector("video");
    var chapters = el("nav", "clip-moments");
    chapters.setAttribute("aria-label", demo.name + " video chapters");
    (demo.chapters || []).forEach(function (chapter) {
      var button = el("button", "moment-seek", chapter.label);
      button.type = "button";
      button.setAttribute("aria-label", chapter.label);
      button.addEventListener("click", function () {
        video.pause();
        video.currentTime = chapter.time_seconds + 0.5 / (demo.fps || 25);
      });
      chapters.appendChild(button);
    });
    clip.appendChild(chapters);
    var explorer = el("details", "episode-explorer");
    explorer.appendChild(el("summary", null, "Explore full plan, memory & timeline"));
    var layout = el("div", "episode-layout");
    var observations = el("div", "episode-observations");
    var context = el("aside", "episode-context");
    context.setAttribute("aria-label", demo.name + " plan and model input memory");
    layout.append(observations, context);
    explorer.appendChild(layout);
    var plan = renderContext(demo, video, observations, context);
    if (plan) explorer.appendChild(plan);
    if (!context.childElementCount) context.hidden = true;
    clip.appendChild(explorer);
    if (demo.review_note) {
      var review = el("details", "clip-plan");
      review.appendChild(el("summary", null, "Review note"));
      review.appendChild(el("p", null, demo.review_note));
      clip.appendChild(review);
    }

    return watch(clip);
  }

  function renderDemos(payload) {
    var grid = document.getElementById("demo-grid");
    if (!grid) return;
    grid.replaceChildren();

    var demos = payload.demos || [];
    if (!demos.length) {
      grid.appendChild(el("p", "deck-note", "No rollout artifacts were published with this build."));
      return;
    }
    var tabs = el("div", "demo-tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Selected planning cases");
    var clips = [];
    var buttons = [];
    function selectCase(index, moveFocus) {
      clips.forEach(function (clip, i) {
        clip.hidden = i !== index;
        buttons[i].setAttribute("aria-selected", String(i === index));
        buttons[i].tabIndex = i === index ? 0 : -1;
        if (i !== index) clip.querySelectorAll("video").forEach(function (video) { video.pause(); });
      });
      clips[index].classList.add("is-in");
      if (moveFocus) buttons[index].focus();
    }
    demos.forEach(function (demo, index) {
      var clip = renderClip(demo, index);
      clip.id = "case-" + demo.slug;
      clip.setAttribute("role", "tabpanel");
      clip.setAttribute("aria-labelledby", "tab-" + demo.slug);
      clips.push(clip);
      var tab = el("button", "demo-tab", demo.name);
      tab.type = "button";
      tab.id = "tab-" + demo.slug;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", clip.id);
      tab.addEventListener("click", function () { selectCase(index, false); });
      tab.addEventListener("keydown", function (event) {
        var target = index;
        if (event.key === "ArrowRight") target = (index + 1) % demos.length;
        else if (event.key === "ArrowLeft") target = (index + demos.length - 1) % demos.length;
        else if (event.key === "Home") target = 0;
        else if (event.key === "End") target = demos.length - 1;
        else return;
        event.preventDefault();
        selectCase(target, true);
      });
      buttons.push(tab);
      tabs.appendChild(tab);
    });
    grid.appendChild(tabs);
    clips.forEach(function (clip) { grid.appendChild(clip); });
    selectCase(0, false);

    var disclaimer = document.getElementById("demo-disclaimer");
    if (disclaimer && payload.disclaimer) {
      var details = el("details", "demo-evidence");
      details.appendChild(el("summary", null, "Provenance & selection"));
      details.appendChild(el("p", null, payload.disclaimer));
      disclaimer.replaceChildren(details);
      if (payload.selection) {
        details.appendChild(document.createTextNode(" Selected qualitative examples: " +
          payload.selection.featured_count + " of " + payload.selection.candidate_count + " candidates. "));
        var record = el("a", null, "Selection review and all candidate records");
        record.href = payload.selection.record;
        details.appendChild(record);
      }
      disclaimer.hidden = false;
    }
  }

  /* ---------- reported results ---------- */

  function renderBenchmarkCases(payload) {
    var grid = document.getElementById("benchmark-cases");
    if (!grid) return;
    grid.replaceChildren();
    (payload.cases || []).forEach(function (item) {
      var figure = el("figure", "benchmark-case");
      figure.appendChild(renderMedia({ name: item.task, video: item.video, poster: item.poster,
        kind: item.kind, aspect_ratio: "4 / 3" }));
      var caption = el("figcaption", null);
      caption.appendChild(el("h3", null, item.task));
      caption.appendChild(el("p", null, item.description));
      var record = el("a", null, "Episode record");
      record.href = item.record;
      caption.appendChild(record);
      figure.appendChild(caption);
      grid.appendChild(figure);
    });
  }

  function oursRow(suite) {
    return (suite.rows || []).filter(function (row) { return row.ours; })[0];
  }

  function bestBaseline(suite) {
    return (suite.rows || [])
      .filter(function (row) { return !row.ours && row.available && typeof row.value === "number"; })
      .sort(function (a, b) { return b.value - a.value; })[0];
  }

  function renderStat(suite, index) {
    var mine = oursRow(suite);
    var best = bestBaseline(suite);

    var stat = el("div", "stat reveal");
    stat.style.setProperty("--rise-delay", index * 0.09 + "s");
    stat.appendChild(el("div", "stat-lab", suite.name));

    var value = mine ? formatValue(mine.value) : null;
    stat.appendChild(el("div", "stat-val", value === null ? "—" : value));

    if (best) {
      stat.appendChild(el("div", "stat-sub", "best baseline " + formatValue(best.value)));
    }
    return watch(stat);
  }

  function renderRow(row) {
    var tr = el("tr", row.ours ? "ours" : null);

    var th = document.createElement("th");
    th.scope = "row";
    th.appendChild(document.createTextNode(row.method));
    if (row.ours) th.appendChild(el("span", "tag-ours", "this work"));
    if (!row.available) th.appendChild(el("span", "tag-unreported", "not reported"));
    tr.appendChild(th);

    var td = document.createElement("td");
    var value = formatValue(row.available ? row.value : null);

    if (value === null) {
      td.appendChild(el("span", "cell-value na", "—"));
    } else {
      var cell = el("div", "cell-meter");
      var meter = el("div", "meter");
      var fill = document.createElement("span");
      fill.style.setProperty("--w", value + "%");
      meter.appendChild(fill);
      cell.appendChild(meter);
      cell.appendChild(el("span", "cell-value", value));
      td.appendChild(cell);
    }

    tr.appendChild(td);
    return tr;
  }

  function renderSuite(suite, metric) {
    var block = el("div", "results-table");
    var caption = document.createElement("caption");
    caption.appendChild(document.createTextNode(suite.name));
    if (suite.tasks && suite.tasks.length) {
      caption.appendChild(el("span", "tasks", suite.tasks.join(" · ")));
    }
    var table = document.createElement("table");
    table.appendChild(caption);
    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");

    var thMethod = document.createElement("th");
    thMethod.scope = "col";
    thMethod.textContent = "Method";
    headRow.appendChild(thMethod);

    var thMetric = document.createElement("th");
    thMetric.scope = "col";
    thMetric.textContent = metric.name;
    headRow.appendChild(thMetric);

    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement("tbody");
    (suite.rows || []).forEach(function (row) { tbody.appendChild(renderRow(row)); });
    table.appendChild(tbody);
    block.appendChild(table);
    return block;
  }

  function renderResults(payload) {
    var grid = document.getElementById("results-grid");
    if (!grid) return;
    grid.replaceChildren();

    var metric = payload.metric || { name: "Score", min: 0, max: 100 };

    var definition = document.getElementById("metric-definition");
    if (definition) {
      var text = (metric.definition || metric.name) + " Range: " + metric.min +
        " to " + metric.max + "; higher is better.";
      definition.textContent = text;
    }

    var suites = payload.suites || [];
    if (!suites.length) {
      grid.appendChild(el("p", "deck-note", "No reported results were published with this build."));
      return;
    }

    var statRow = document.getElementById("stat-row");
    if (statRow) {
      statRow.replaceChildren();
      suites.forEach(function (suite, index) { statRow.appendChild(renderStat(suite, index)); });
    }

    suites.forEach(function (suite) { grid.appendChild(renderSuite(suite, metric)); });

    var box = document.getElementById("limitations");
    var limitations = payload.limitations || [];
    if (box && limitations.length) {
      box.appendChild(el("h3", null, "Limitations"));
      var list = document.createElement("ul");
      limitations.forEach(function (item) { list.appendChild(el("li", null, item)); });
      box.appendChild(list);
      box.hidden = false;
    }
  }

  /* ---------- boot ---------- */

  function fail(id, message) {
    var host = document.getElementById(id);
    if (host) host.replaceChildren(el("p", "deck-note", message));
  }

  watchAll();

  getJSON("data/demos.json").then(renderDemos).catch(function () {
    fail("demo-grid", "Rollout records are unavailable in this build.");
  });

  getJSON("data/results.json").then(renderResults).catch(function () {
    fail("results-grid", "Reported results are unavailable in this build.");
  });
  getJSON("data/benchmark-cases.json").then(renderBenchmarkCases).catch(function () {
    fail("benchmark-cases", "Benchmark case recordings are unavailable.");
  });
}());
