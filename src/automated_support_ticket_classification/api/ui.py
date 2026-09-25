"""A tiny self-contained test page for the classifier.

The HTML lives here as a string rather than as a static file so it needs no
package-data declaration and no extra COPY in the Dockerfile: it travels with
the module wherever the package goes.

No CDN, no build step, no dependencies. Everything is inline so the page works
offline and inside the container.
"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ticket Classifier</title>
<style>
  :root {
    --bg:#0f172a; --card:#1e293b; --line:#334155; --text:#e2e8f0;
    --muted:#94a3b8; --accent:#3b82f6; --ok:#22c55e;
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:32px 20px; background:var(--bg); color:var(--text);
         font-family:"Segoe UI",system-ui,-apple-system,sans-serif; line-height:1.5; }
  .wrap { max-width:760px; margin:0 auto; }
  h1 { font-size:24px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:14px; margin-bottom:24px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:20px; margin-bottom:18px; }
  label { display:block; font-size:13px; color:var(--muted); margin-bottom:8px;
          text-transform:uppercase; letter-spacing:.5px; }
  textarea { width:100%; min-height:88px; background:#0b1220; color:var(--text);
             border:1px solid var(--line); border-radius:8px; padding:12px;
             font-family:inherit; font-size:15px; resize:vertical; }
  textarea:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button { background:var(--accent); color:#fff; border:0; border-radius:8px;
           padding:10px 20px; font-size:15px; font-weight:600; cursor:pointer; }
  button:hover { filter:brightness(1.1); }
  button:disabled { opacity:.5; cursor:default; }
  .examples { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:6px; }
  .ex { background:transparent; border:1px solid var(--line); color:var(--muted);
        border-radius:999px; padding:6px 14px; font-size:13px; font-weight:500; }
  .ex:hover { border-color:var(--accent); color:var(--text); }
  .verdict { display:flex; align-items:baseline; gap:12px; margin-bottom:18px; }
  .label { font-size:30px; font-weight:700; color:var(--ok); }
  .conf { color:var(--muted); font-size:15px; }
  .bar-row { display:grid; grid-template-columns:86px 1fr 58px; align-items:center;
             gap:10px; margin-bottom:7px; font-size:14px; }
  /* display:block matters: these are spans, and width/height do not apply to
     inline elements, so without it every bar renders full width. */
  .track { display:block; background:#0b1220; border-radius:999px;
           height:9px; overflow:hidden; }
  .fill { display:block; background:var(--accent); height:100%;
          border-radius:999px; transition:width .35s ease; }
  .fill.top { background:var(--ok); }
  .pct { text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; }
  .err { color:#f87171; }
  .hint { color:var(--muted); font-size:13px; margin-top:14px; }
  code { background:#0b1220; padding:2px 6px; border-radius:4px; font-size:13px; }
  .hidden { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Support Ticket Classifier</h1>
  <div class="sub">
    TF-IDF plus logistic regression, five categories.
    Calls <code>POST /predict</code> on this service.
  </div>

  <div class="card">
    <label>Try an example</label>
    <div class="examples" id="examples"></div>

    <label style="margin-top:18px">Or write your own ticket</label>
    <textarea id="text"
      placeholder="My order has not arrived and the tracking shows nothing..."></textarea>

    <div class="row">
      <button id="go">Classify</button>
      <span class="conf" id="status"></span>
    </div>
  </div>

  <div class="card hidden" id="result">
    <div class="verdict">
      <span class="label" id="label"></span>
      <span class="conf" id="conf"></span>
    </div>
    <div id="bars"></div>
  </div>

  <div class="hint">
    Interactive API docs at <a href="/docs" style="color:var(--accent)">/docs</a>,
    Prometheus metrics at <a href="/metrics" style="color:var(--accent)">/metrics</a>.
    Try whitespace only to see the 422 guard.
  </div>
</div>

<script>
const EXAMPLES = [
  ["billing",   "I was charged $49 twice on my Visa card this month"],
  ["technical", "The app crashes every time I open the reports page"],
  ["account",   "I cannot reset my password, the email never arrives"],
  ["shipping",  "My order #10231 has not arrived after two weeks"],
  ["general",   "Do you offer a discount for students or non profits"],
];

const $ = (id) => document.getElementById(id);

EXAMPLES.forEach(([name, text]) => {
  const b = document.createElement("button");
  b.className = "ex";
  b.textContent = name;
  b.title = text;
  b.onclick = () => { $("text").value = text; classify(); };
  $("examples").appendChild(b);
});

async function classify() {
  const text = $("text").value;
  $("status").textContent = "classifying...";
  $("status").className = "conf";
  $("go").disabled = true;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    if (res.status === 422) {
      $("result").classList.add("hidden");
      $("status").textContent = "422: text must not be empty";
      $("status").className = "conf err";
      return;
    }
    if (!res.ok) throw new Error("HTTP " + res.status);
    render(await res.json());
    $("status").textContent = "";
  } catch (e) {
    $("result").classList.add("hidden");
    $("status").textContent = "request failed: " + e.message;
    $("status").className = "conf err";
  } finally {
    $("go").disabled = false;
  }
}

function render(body) {
  $("label").textContent = body.label;
  $("conf").textContent = (body.confidence * 100).toFixed(1) + "% confident";
  const sorted = Object.entries(body.all_scores).sort((a, b) => b[1] - a[1]);
  $("bars").innerHTML = sorted.map(([name, p], i) => `
    <div class="bar-row">
      <span>${name}</span>
      <span class="track"><span class="fill ${i === 0 ? "top" : ""}"
        style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="pct">${(p * 100).toFixed(1)}%</span>
    </div>`).join("");
  $("result").classList.remove("hidden");
}

$("go").onclick = classify;
$("text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classify();
});
</script>
</body>
</html>
"""
