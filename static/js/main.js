/* Shared helpers for all pages */
const $ = (s, r = document) => r.querySelector(s);
const LV_ORDER = {Low: 0, Medium: 1, High: 2};
const badge = l => `<span class="lvl lvl-${l}">${l}</span>`;
const gauge = (pct, lvl) => `<div class="gauge"><div class="lvl-bg-${lvl}" style="width:${pct}%"></div></div>`;
const statusBadge = s => {
  const m = {completed: 'success', processing: 'primary pulse', queued: 'secondary', error: 'danger', interrupted: 'warning', cancelled: 'secondary'};
  return `<span class="badge text-bg-${m[s] || 'secondary'}">${s}</span>`;
};
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  let j = {};
  try { j = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok) throw new Error(j.error || `${r.status} ${r.statusText}`);
  return j;
}
/* Poll an analysis session until all roads are finished (plus one extra tick to catch the final plan). */
function pollSession(sid, render) {
  let idle = 0;
  (async function tick() {
    try {
      const ov = await getJSON(`/api/session/${sid}/overview`);
      render(ov);
      idle = ov.active ? 0 : idle + 1;
      if (idle >= 2) return;
    } catch (e) { console.error(e); }
    setTimeout(tick, 1500);
  })();
}
function refreshImages(sel, active) {
  document.querySelectorAll(sel).forEach(img => { img.src = img.dataset.src + '?t=' + Date.now(); });
}
function autoRefreshCharts(ov, sel) {
  const n = ov.roads.reduce((a, r) => a + r.frames, 0);
  if (n !== autoRefreshCharts.last || !ov.active) { autoRefreshCharts.last = n; refreshImages(sel); }
}
function levelFromWorst(list) { return list.reduce((a, l) => LV_ORDER[l] > LV_ORDER[a] ? l : a, 'Low'); }
function renderPlanTable(plan, el) {
  if (!plan) { el.innerHTML = '<p class="small-muted mb-0">Signal plan appears once at least two roads have been analysed.</p>'; return; }
  el.innerHTML = `<table class="table table-sm align-middle mb-2"><thead><tr><th>Road</th><th>Density</th><th>Avg PCU</th><th>Green</th><th>Share</th><th>Fixed</th></tr></thead><tbody>` +
    plan.roads.map(r => `<tr><td><b>${r.road}</b></td><td>${badge(r.level)}</td><td>${r.score}</td><td><b>${r.green}s</b></td><td>${r.share_pct}%</td><td class="text-muted">${r.fixed_green}s</td></tr>`).join('') +
    `</tbody></table><div class="small-muted">Cycle ${plan.cycle}s (incl. ${plan.yellow}s yellow per phase) · min green ${plan.min_green}s / max ${plan.max_green}s · est. avg wait <b>${plan.avg_wait_fixed}s → ${plan.avg_wait_adaptive}s</b> (${plan.reduction_pct}% lower, model estimate)</div>`;
}
/* Queue simulator: same arrivals, fixed vs adaptive green times */
class QueueSim {
  constructor(el, plan, mode) {
    this.el = el; this.mode = mode; this.t = 0;
    const n = plan.roads.length, tot = plan.roads.reduce((a, r) => a + r.score + 0.5, 0);
    this.cap = 1.0;                                    // vehicles discharged per green second
    this.roads = plan.roads.map(r => ({name: r.road, q: 0, arr: 0.75 * plan.green_budget * this.cap * ((r.score + 0.5) / tot) / plan.cycle,
      green: mode === 'adaptive' ? r.green : plan.green_budget / n}));
    this.yellow = plan.yellow; this.cycle = this.roads.reduce((a, r) => a + r.green + this.yellow, 0);
    el.innerHTML = this.roads.map((r, i) => `<tr><td><b>${r.name}</b></td>
      <td><span class="light r"></span><span class="light y"></span><span class="light g"></span></td>
      <td class="count" style="width:48px">–</td><td style="width:45%"><div class="qbar"><div style="width:0"></div></div></td><td class="q small-muted">0</td></tr>`).join('');
  }
  step() {
    this.t = (this.t + 1) % this.cycle;
    let acc = 0, phaseIdx = 0, within = 0;
    for (let i = 0; i < this.roads.length; i++) {
      const len = this.roads[i].green + this.yellow;
      if (this.t < acc + len) { phaseIdx = i; within = this.t - acc; break; }
      acc += len;
    }
    const rows = this.el.querySelectorAll('tr');
    this.roads.forEach((r, i) => {
      r.q += r.arr;
      const green = i === phaseIdx && within < r.green, yellow = i === phaseIdx && !green;
      if (green) r.q = Math.max(0, r.q - this.cap);
      const tr = rows[i], L = tr.querySelectorAll('.light');
      L[0].classList.toggle('on', !green && !yellow); L[1].classList.toggle('on', yellow); L[2].classList.toggle('on', green);
      tr.querySelector('.count').textContent = i === phaseIdx ? (green ? r.green - within : this.yellow - (within - r.green)) + 's' : '';
      const pct = Math.min(100, r.q / 40 * 100), bar = tr.querySelector('.qbar');
      bar.firstElementChild.style.width = pct + '%'; bar.classList.toggle('over', pct > 70);
      tr.querySelector('.q').textContent = Math.round(r.q) + ' veh';
    });
  }
  totalQueue() { return this.roads.reduce((a, r) => a + r.q, 0); }
}
