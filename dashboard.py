"""ATLAS Mission Control dashboard - localhost-only live view.

stdlib http.server; no external JS/CSS (works offline, no CDN). Read-only
against all trading state via mission_control.py. Serves:
  /            self-contained HTML dashboard (auto-refresh)
  /api/data    full JSON payload

Usage: .venv\\Scripts\\python.exe dashboard.py [--port 8787]
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mission_control import (
    HEARTBEAT_PATH, MARKET_DB, SIGNALS_PATH, build_status, expected_value,
    read_jsonl, split_signals,
)

VERDICT_WINDOW = "Jul 28 - Aug 6, 2026"


def latest_payouts() -> list[dict]:
    """Most recent payout per quoted asset/kind, read-only."""
    if not MARKET_DB.exists():
        return []
    try:
        import duckdb
        conn = duckdb.connect(str(MARKET_DB), read_only=True)
        try:
            rows = conn.execute(
                """SELECT asset, kind, payout FROM payout_snapshots
                   WHERE ts_epoch = (SELECT max(ts_epoch) FROM payout_snapshots)
                   ORDER BY payout DESC"""
            ).fetchall()
        finally:
            conn.close()
        return [{"asset": a, "kind": k, "payout": float(p)} for a, k, p in rows]
    except Exception:
        return []


def build_payload() -> dict:
    status = build_status()
    heartbeats = read_jsonl(HEARTBEAT_PATH, tail=720)  # ~12h of cycles
    parts = split_signals(read_jsonl(SIGNALS_PATH))
    signals = [
        {
            "ts": r.get("ts"), "asset": r.get("asset"), "action": r.get("action"),
            "p_up": r.get("p_up"), "meta_p": r.get("meta_p"),
            "payout": r.get("payout"), "mode": r.get("mode"),
            "order_id": r.get("order_id"),
            "skipped": r.get("trade_skipped"),
            "ev": round(expected_value(float(r["p_up"]), float(r["payout"])), 4)
                  if r.get("p_up") is not None and r.get("payout") is not None else None,
        }
        for r in parts["signals"]
    ]
    settled = [
        {"ts": r.get("ts"), "asset": r.get("asset"), "action": r.get("action"),
         "result": r.get("result"), "profit": r.get("profit"),
         "meta_p": r.get("meta_p")}
        for r in parts["settled"]
    ]
    return {
        "status": status,
        "heartbeats": [{"ts": h.get("ts"), "max_conf": h.get("max_conf"),
                        "assets": h.get("assets"), "signals": h.get("signals")}
                       for h in heartbeats],
        "signals": signals[-500:],
        "settled": settled[-500:],
        "payouts": latest_payouts(),
        "verdict_window": VERDICT_WINDOW,
        "server_time": int(time.time()),
    }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>ATLAS Mission Control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg:#07090b; --panel:#0c1013; --panel2:#0a0d10; --line:#1c2329;
  --line2:#2a333b; --fg:#cfd8dd; --dim:#6e7b85; --faint:#48535c;
  --amber:#f5a83c; --amber2:#ffcf87; --green:#67d17c; --red:#ff6a5f;
  --warnc:#e3b341;
  --disp:"Bahnschrift","Segoe UI Variable Display","Segoe UI",sans-serif;
  --mono:"Cascadia Mono","Cascadia Code",Consolas,monospace;
}
* { box-sizing:border-box; margin:0; padding:0; }
html { background:var(--bg); }
body {
  color:var(--fg); font:13px/1.5 var(--disp); padding:22px 26px 40px;
  min-height:100vh;
  background:
    radial-gradient(1100px 500px at 18% -10%, rgba(245,168,60,.055), transparent 60%),
    radial-gradient(900px 600px at 100% 0%, rgba(103,209,124,.035), transparent 55%),
    var(--bg);
}
/* CRT scanlines + vignette, barely-there */
body::before { content:""; position:fixed; inset:0; pointer-events:none; z-index:9;
  background:repeating-linear-gradient(0deg, rgba(255,255,255,.014) 0 1px, transparent 1px 3px); }
body::after { content:""; position:fixed; inset:0; pointer-events:none; z-index:9;
  background:radial-gradient(120% 90% at 50% 40%, transparent 60%, rgba(0,0,0,.38)); }

/* ---------- masthead ---------- */
header { display:flex; align-items:center; gap:22px; flex-wrap:wrap;
  padding-bottom:16px; margin-bottom:18px; border-bottom:1px solid var(--line);
  animation:reveal .5s ease both; }
.brand { display:flex; align-items:baseline; gap:12px; }
.brand h1 { font-size:26px; font-weight:700; letter-spacing:.22em; color:var(--amber2);
  text-shadow:0 0 22px rgba(245,168,60,.35); }
.brand small { font-family:var(--mono); font-size:10px; letter-spacing:.32em;
  color:var(--dim); text-transform:uppercase; }
.lamp { width:14px; height:14px; border-radius:50%; position:relative; flex:none; }
.lamp::after { content:""; position:absolute; inset:-7px; border-radius:50%;
  border:1px solid var(--line2); }
.lamp.HEALTHY  { background:var(--green); box-shadow:0 0 14px var(--green);
  animation:pulse 2.6s ease-in-out infinite; }
.lamp.WARNING  { background:var(--warnc); box-shadow:0 0 14px var(--warnc);
  animation:pulse 1.4s ease-in-out infinite; }
.lamp.CRITICAL { background:var(--red); box-shadow:0 0 16px var(--red);
  animation:pulse .6s ease-in-out infinite; }
@keyframes pulse { 50% { opacity:.55; } }
#tier { font-family:var(--mono); font-size:15px; letter-spacing:.24em; }
#tier.HEALTHY { color:var(--green); } #tier.WARNING { color:var(--warnc); }
#tier.CRITICAL { color:var(--red); }
.tele { font-family:var(--mono); font-size:11px; color:var(--dim); letter-spacing:.04em; }
.tele b { color:var(--fg); font-weight:400; }
.clockbox { margin-left:auto; text-align:right; }
#clock { font-family:var(--mono); font-size:19px; color:var(--amber2);
  letter-spacing:.14em; text-shadow:0 0 12px rgba(245,168,60,.3); }
#count { font-family:var(--mono); font-size:10px; color:var(--dim); letter-spacing:.14em; }
#reasons { font-family:var(--mono); font-size:11.5px; color:var(--warnc);
  letter-spacing:.03em; margin:-6px 0 14px; min-height:1em; }

/* ---------- grid & panels ---------- */
.grid { display:grid; gap:14px;
  grid-template-columns:repeat(12, 1fr); }
.card { grid-column:span 4; background:linear-gradient(180deg, rgba(255,255,255,.018), transparent 45%), var(--panel);
  border:1px solid var(--line); padding:14px 16px 12px; position:relative;
  animation:reveal .55s ease both; }
.card:nth-child(2) { animation-delay:.05s } .card:nth-child(3) { animation-delay:.1s }
.card:nth-child(4) { animation-delay:.15s } .card:nth-child(5) { animation-delay:.2s }
.card:nth-child(6) { animation-delay:.25s } .card:nth-child(7) { animation-delay:.3s }
.card:nth-child(8) { animation-delay:.35s } .card:nth-child(9) { animation-delay:.4s }
@keyframes reveal { from { opacity:0; transform:translateY(7px); } }
/* blueprint corner ticks */
.card::before, .card::after { content:""; position:absolute; width:7px; height:7px;
  border:1px solid var(--line2); pointer-events:none; }
.card::before { top:-1px; left:-1px; border-right:0; border-bottom:0; }
.card::after  { bottom:-1px; right:-1px; border-left:0; border-top:0; }
.card.w8 { grid-column:span 8; } .card.w12 { grid-column:span 12; }
.card.w6 { grid-column:span 6; }
@media (max-width:1100px){ .card, .card.w6, .card.w8 { grid-column:span 6; } }
@media (max-width:720px){ .card, .card.w6, .card.w8, .card.w12 { grid-column:span 12; } }
h2 { font-size:10px; font-weight:600; color:var(--dim); text-transform:uppercase;
  letter-spacing:.24em; margin-bottom:10px; display:flex; gap:8px; align-items:center; }
h2 .idx { color:var(--amber); font-family:var(--mono); font-size:9px; }
h2 .sub { margin-left:auto; color:var(--faint); letter-spacing:.1em; font-size:9px;
  text-transform:none; }

/* ---------- readouts ---------- */
.kv { display:flex; justify-content:space-between; align-items:baseline;
  padding:4px 0; border-bottom:1px dashed rgba(255,255,255,.05); }
.kv:last-child { border-bottom:0; }
.kv span { color:var(--dim); font-size:12px; }
.kv b { font-family:var(--mono); font-weight:400; font-size:15px; color:var(--amber2); }
.kv b.g { color:var(--green); } .kv b.r { color:var(--red); } .kv b.plain { color:var(--fg); }
.meter { height:3px; background:var(--line); margin-top:8px; position:relative; }
.meter i { position:absolute; inset:0 auto 0 0; background:var(--amber);
  box-shadow:0 0 8px rgba(245,168,60,.6); transition:width .8s ease; }
canvas { width:100%; height:168px; display:block; }

/* ---------- ledger table ---------- */
table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:11.5px; }
th { color:var(--faint); font-weight:400; text-transform:uppercase; font-size:9.5px;
  letter-spacing:.18em; text-align:left; padding:4px 8px 7px;
  border-bottom:1px solid var(--line2); }
td { padding:4px 8px; border-bottom:1px solid rgba(255,255,255,.04); color:var(--fg); }
tr:hover td { background:rgba(245,168,60,.045); }
td.dim { color:var(--dim); } .call { color:var(--green); } .put { color:var(--red); }
.empty { color:var(--faint); font-family:var(--mono); font-size:11px;
  letter-spacing:.12em; padding:26px 0; text-align:center; }
</style></head><body>
<header>
  <div class="lamp HEALTHY" id="lamp"></div>
  <div class="brand"><h1>ATLAS</h1><small>Mission Control · demo / practice only</small></div>
  <div id="tier">—</div>
  <div class="tele" id="hb">awaiting telemetry…</div>
  <div class="clockbox"><div id="clock">--:--:--</div><div id="count"></div></div>
</header>
<div id="reasons"></div>
<div class="grid">
  <div class="card"><h2><span class="idx">01</span>Forward Test<span class="sub" id="vwindow"></span></h2>
    <div id="fwd"></div>
    <div class="meter"><i id="fwdbar" style="width:0%"></i></div>
    <div class="tele" style="margin-top:6px">counts only — verdicts: forward_eval.py, run once</div></div>
  <div class="card"><h2><span class="idx">02</span>Label Fidelity<span class="sub">broker vs candle</span></h2>
    <div id="fid"></div>
    <div class="meter"><i id="fidbar" style="width:0%"></i></div></div>
  <div class="card"><h2><span class="idx">03</span>Signals &amp; Orders</h2><div id="sig"></div></div>
  <div class="card w8"><h2><span class="idx">04</span>Model Confidence<span class="sub">max |p−.5| per cycle</span></h2><canvas id="conf"></canvas></div>
  <div class="card"><h2><span class="idx">05</span>Signals · UTC Hour</h2><canvas id="hours"></canvas></div>
  <div class="card"><h2><span class="idx">06</span>Equity<span class="sub">settled demo $</span></h2><canvas id="equity"></canvas></div>
  <div class="card"><h2><span class="idx">07</span>Win Rate · Asset</h2><canvas id="wr"></canvas></div>
  <div class="card"><h2><span class="idx">08</span>Live Payouts</h2><canvas id="pay"></canvas></div>
  <div class="card w12"><h2><span class="idx">09</span>Signal Ledger<span class="sub">latest 25</span></h2>
    <table id="recent"><thead><tr><th>time utc</th><th>asset</th><th>side</th>
    <th>p_up</th><th>meta_p</th><th>ev</th><th>payout</th><th>mode</th><th>order</th></tr></thead>
    <tbody></tbody></table></div>
</div>
<script>
const AMBER="#f5a83c", AMBER2="#ffcf87", GREEN="#67d17c", RED="#ff6a5f",
      DIM="#6e7b85", FAINT="#48535c", LINE="#1c2329";
const MONO="10px 'Cascadia Mono',Consolas,monospace";
function el(id){ return document.getElementById(id); }
function fmtAge(s){ if(s==null) return "n/a"; if(s<120) return s+"s";
  if(s<7200) return Math.round(s/60)+"m"; return (s/3600).toFixed(1)+"h"; }
function utc(ts){ return new Date(ts*1000).toISOString().slice(5,16).replace("T"," "); }

/* mission clock + verdict countdown, ticking client-side */
const VERDICT_OPEN = Date.UTC(2026,6,28);
function tick(){
  const now = new Date();
  el("clock").textContent = now.toISOString().slice(11,19)+" UTC";
  const dms = VERDICT_OPEN - now.getTime();
  el("count").textContent = dms > 0
    ? "T−" + Math.floor(dms/86400000) + "d " +
      String(Math.floor(dms/3600000)%24).padStart(2,"0") + "h to verdict window"
    : "verdict window OPEN — forward_eval.py runs once";
}
setInterval(tick, 1000); tick();

function chart(id, draw){ const c = el(id), ctx = c.getContext("2d");
  c.width = c.clientWidth*devicePixelRatio;
  c.height = c.clientHeight*devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  draw(ctx, c.clientWidth, c.clientHeight); }

function frame(ctx,w,h){
  ctx.strokeStyle=LINE; ctx.lineWidth=1; ctx.setLineDash([1,3]);
  for(let i=1;i<4;i++){ const y=8+(h-30)*i/4;
    ctx.beginPath(); ctx.moveTo(34,y); ctx.lineTo(w-6,y); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.strokeStyle="#242d34";
  ctx.beginPath(); ctx.moveTo(34,8); ctx.lineTo(34,h-22); ctx.lineTo(w-6,h-22); ctx.stroke();
}
function noData(ctx,w,h){ ctx.fillStyle=FAINT; ctx.font=MONO;
  ctx.textAlign="center"; ctx.fillText("· no telemetry yet ·", w/2, h/2);
  ctx.textAlign="left"; }

function line(id, pts, color, fmt){ chart(id,(ctx,w,h)=>{ frame(ctx,w,h);
  if(!pts.length){ noData(ctx,w,h); return; }
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
  const x0=Math.min(...xs), x1=Math.max(...xs)||1;
  let y0=Math.min(...ys,0), y1=Math.max(...ys); if(y0===y1){ y1=y0+1; }
  const X=t=>34+(w-42)*(t-x0)/((x1-x0)||1), Y=v=>h-22-(h-30)*(v-y0)/(y1-y0);
  const grad=ctx.createLinearGradient(0,8,0,h-22);
  grad.addColorStop(0,color+"33"); grad.addColorStop(1,color+"00");
  ctx.beginPath(); ctx.moveTo(X(pts[0][0]),Y(pts[0][1]));
  pts.forEach(p=>ctx.lineTo(X(p[0]),Y(p[1])));
  ctx.lineTo(X(pts[pts.length-1][0]),h-22); ctx.lineTo(X(pts[0][0]),h-22);
  ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
  ctx.shadowColor=color; ctx.shadowBlur=7;
  ctx.strokeStyle=color; ctx.lineWidth=1.4; ctx.beginPath();
  pts.forEach((p,i)=> i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1])));
  ctx.stroke(); ctx.shadowBlur=0;
  const last=pts[pts.length-1];
  ctx.fillStyle=color; ctx.beginPath();
  ctx.arc(X(last[0]),Y(last[1]),2.4,0,7); ctx.fill();
  ctx.fillStyle=DIM; ctx.font=MONO;
  const F=fmt||(v=>v.toFixed(2));
  ctx.fillText(F(y1), 2, 14); ctx.fillText(F(y0), 2, h-22);
  ctx.fillStyle=FAINT;
  ctx.fillText(utc(x0), 36, h-8); ctx.textAlign="right";
  ctx.fillText(utc(x1), w-6, h-8); ctx.textAlign="left"; }); }

function bars(id, labels, values, color, fmt){ chart(id,(ctx,w,h)=>{ frame(ctx,w,h);
  if(!values.length){ noData(ctx,w,h); return; }
  const y1=Math.max(...values)||1, n=values.length,
        slot=(w-44)/n, bw=Math.max(2, slot-3);
  values.forEach((v,i)=>{ const x=36+i*slot, bh=(h-30)*v/y1;
    const g=ctx.createLinearGradient(0,h-22-bh,0,h-22);
    g.addColorStop(0,color); g.addColorStop(1,color+"55");
    ctx.fillStyle=g; ctx.fillRect(x,h-22-bh,bw,bh);
    if(bh>2){ ctx.fillStyle="#fff3"; ctx.fillRect(x,h-22-bh,bw,1); } });
  ctx.fillStyle=DIM; ctx.font=MONO;
  ctx.fillText((fmt||(v=>v))(y1), 2, 14);
  const step=Math.ceil(n/8);
  ctx.fillStyle=FAINT;
  labels.forEach((L,i)=>{ if(i%step===0)
    ctx.fillText(String(L).slice(0,7), 36+i*slot, h-8); }); });
}

const kv=(k,v,cls)=>`<div class="kv"><span>${k}</span><b class="${cls||""}">${v}</b></div>`;

async function refresh(){
  const d = await (await fetch("/api/data")).json();
  const s = d.status;
  el("tier").textContent = s.tier;
  el("tier").className = s.tier;
  el("lamp").className = "lamp "+s.tier;
  el("reasons").textContent = (s.reasons||[]).map(r=>"▲ "+r).join("   ");
  el("vwindow").textContent = d.verdict_window;
  const hb = s.heartbeat;
  el("hb").innerHTML = hb.last
    ? `heartbeat <b>${fmtAge(hb.age_s)}</b> ago · assets <b>${hb.last.assets}</b> · max_conf <b>${hb.last.max_conf}</b>`
    : "no heartbeat yet";

  const fp = s.forward_progress, target = 100;
  el("fwd").innerHTML = Object.entries(fp).map(([k,v])=>kv(k,v)).join("");
  el("fwdbar").style.width = Math.min(100, 100*(fp["H2p ev0.03"]||0)/target)+"%";

  const f = s.fidelity||{};
  el("fid").innerHTML =
    kv("settled orders", `${f.settled_orders||0} / ${f.target_trades||100}`) +
    kv("agree", f.agree||0, "g") + kv("disagree", f.disagree||0, "r") +
    kv("agreement", f.agreement_rate==null?"—":(100*f.agreement_rate).toFixed(1)+"%") +
    kv("undetermined", f.undetermined||0, "plain");
  el("fidbar").style.width = Math.min(100, 100*(f.settled_orders||0)/(f.target_trades||100))+"%";

  const g = s.signals;
  el("sig").innerHTML =
    kv("signals logged", g.total) + kv("orders placed", g.orders_placed) +
    kv("settled", g.settled, "g") + kv("OTC skipped · by design", g.otc_skipped, "plain");

  line("conf", d.heartbeats.filter(h=>h.max_conf!=null).map(h=>[h.ts,h.max_conf]),
       AMBER, v=>v.toFixed(3));

  const byHour = Array(24).fill(0);
  d.signals.forEach(x=>{ byHour[new Date(x.ts*1000).getUTCHours()]++; });
  bars("hours", [...Array(24).keys()], byHour, AMBER);

  let cum=0; const eq = d.settled.filter(x=>x.profit!=null)
    .map(x=>[x.ts, (cum+=x.profit)]);
  line("equity", eq, cum>=0?GREEN:RED, v=>"$"+v.toFixed(2));

  const per={};
  d.settled.forEach(x=>{ if(!x.result) return;
    (per[x.asset]=per[x.asset]||{w:0,n:0});
    if(x.result==="win") per[x.asset].w++;
    if(x.result!=="equal") per[x.asset].n++; });
  const assets=Object.keys(per).filter(a=>per[a].n>0);
  bars("wr", assets, assets.map(a=>100*per[a].w/per[a].n), GREEN, v=>v.toFixed(0)+"%");

  const pays=(d.payouts||[]).slice(0,18);
  bars("pay", pays.map(p=>p.asset), pays.map(p=>100*p.payout), AMBER2, v=>v.toFixed(0)+"%");

  const rows = d.signals.slice(-25).reverse();
  el("recent").querySelector("tbody").innerHTML = rows.length ? rows
    .map(x=>`<tr><td class="dim">${utc(x.ts)}</td><td>${x.asset}</td>
      <td class="${x.action==="binary_call"?"call":"put"}">${x.action==="binary_call"?"▲ CALL":"▼ PUT"}</td>
      <td>${x.p_up}</td><td class="dim">${x.meta_p??"—"}</td>
      <td>${x.ev??"—"}</td><td class="dim">${x.payout}</td>
      <td class="dim">${x.mode}${x.skipped?" ·skip":""}</td>
      <td class="dim">${x.order_id??"—"}</td></tr>`).join("")
    : `<tr><td colspan="9"><div class="empty">· ledger empty — the meta filter is being selective, which is the edge ·</div></td></tr>`;
}
refresh(); setInterval(refresh, 60000);
addEventListener("resize", ()=>refresh());
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/data"):
            body = json.dumps(build_payload(), default=str).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the console quiet
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"ATLAS Mission Control -> http://127.0.0.1:{args.port}  (Ctrl-C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
