"use strict";

const ACTIONS = { STAY: 0, UP: 1, DOWN: 2, LEFT: 3, RIGHT: 4 };
const DELTAS = { 0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1] };
const COLORS = {
  bg: "#f8fafb",
  grid: "#cbd5df",
  visited: "#b8d8f2",
  obstacle: "#262b30",
  target: "#39a96b",
  foundTarget: "#a9d9bc",
  path: "rgba(15,139,141,.25)",
  agents: ["#3867d6", "#c44536", "#7d5fff", "#f4b942", "#0f8b8d", "#8d6e63"],
};

class RNG {
  constructor(seed) { this.state = seed >>> 0; }
  next() { this.state = (1664525 * this.state + 1013904223) >>> 0; return this.state / 4294967296; }
  int(max) { return Math.floor(this.next() * max); }
}

const $ = (id) => document.querySelector(id);
const els = {
  gridSize: $("#grid-size"), numAgents: $("#num-agents"), numTargets: $("#num-targets"),
  numObstacles: $("#num-obstacles"), maxSteps: $("#max-steps"), policy: $("#policy"),
  seed: $("#seed"), speed: $("#speed"), resetBtn: $("#reset-btn"), stepBtn: $("#step-btn"),
  playBtn: $("#play-btn"), episodeBtn: $("#episode-btn"), exportBtn: $("#export-btn"),
  runState: $("#run-state"), canvas: $("#world-canvas"), metricStep: $("#metric-step"),
  metricReward: $("#metric-reward"), metricTargets: $("#metric-targets"), metricCollisions: $("#metric-collisions"),
  metricCoverage: $("#metric-coverage"), metricMeanReward: $("#metric-mean-reward"),
  agentTable: $("#agent-table"), eventLog: $("#event-log"),
};
const ctx = els.canvas.getContext("2d");
let sim = null;
let timer = null;
let history = [];
let logs = [];

function readConfig() {
  const config = {
    gridSize: clamp(intValue(els.gridSize), 5, 40),
    numAgents: clamp(intValue(els.numAgents), 1, 12),
    numTargets: clamp(intValue(els.numTargets), 0, 40),
    numObstacles: clamp(intValue(els.numObstacles), 0, 300),
    maxSteps: clamp(intValue(els.maxSteps), 1, 2000),
    policy: els.policy.value,
    seed: intValue(els.seed) || 1,
  };
  const capacity = config.gridSize * config.gridSize;
  if (config.numAgents + config.numTargets + config.numObstacles > capacity) {
    config.numObstacles = Math.max(0, capacity - config.numAgents - config.numTargets);
    els.numObstacles.value = String(config.numObstacles);
  }
  return config;
}

function createWorld(config) {
  const rng = new RNG(config.seed);
  const positions = [];
  for (let row = 0; row < config.gridSize; row += 1) for (let col = 0; col < config.gridSize; col += 1) positions.push({ row, col });
  shuffle(positions, rng);
  const agents = Array.from({ length: config.numAgents }, (_, i) => {
    const pos = positions.pop();
    return { id: `A${i}`, row: pos.row, col: pos.col, reward: 0, path: [{ row: pos.row, col: pos.col }] };
  });
  const targets = Array.from({ length: config.numTargets }, () => ({ ...positions.pop(), found: false }));
  const obstacles = new Set(Array.from({ length: config.numObstacles }, () => {
    const pos = positions.pop();
    return key(pos.row, pos.col);
  }));
  const visited = new Set(agents.map((agent) => key(agent.row, agent.col)));
  return { config, rng, agents, targets, obstacles, visited, step: 0, totalReward: 0, collisions: 0, done: false };
}

function resetWorld() {
  stop();
  sim = createWorld(readConfig());
  history = [];
  logs = [];
  log("World reset");
  record();
  render();
}

function stepWorld() {
  if (!sim || sim.done) return;
  sim.step += 1;
  const actions = chooseActions();
  const current = new Map(sim.agents.map((a) => [a.id, { row: a.row, col: a.col }]));
  const proposed = new Map();
  const rewards = new Map(sim.agents.map((a) => [a.id, -0.1]));
  const agentCollisions = new Set();

  for (const agent of sim.agents) {
    const [dr, dc] = DELTAS[actions.get(agent.id) ?? 0];
    const next = { row: agent.row + dr, col: agent.col + dc };
    if (!inBounds(next.row, next.col) || sim.obstacles.has(key(next.row, next.col))) {
      proposed.set(agent.id, current.get(agent.id));
      rewards.set(agent.id, rewards.get(agent.id) - 5);
      sim.collisions += 1;
    } else {
      proposed.set(agent.id, next);
    }
  }

  for (const agent of sim.agents) {
    const here = proposed.get(agent.id);
    if (sim.agents.filter((other) => same(proposed.get(other.id), here)).length > 1) agentCollisions.add(agent.id);
  }
  for (let i = 0; i < sim.agents.length; i += 1) for (let j = i + 1; j < sim.agents.length; j += 1) {
    const a = sim.agents[i], b = sim.agents[j];
    if (same(proposed.get(a.id), current.get(b.id)) && same(proposed.get(b.id), current.get(a.id))) {
      agentCollisions.add(a.id); agentCollisions.add(b.id);
    }
  }
  for (const agent of sim.agents) if (agentCollisions.has(agent.id)) {
    proposed.set(agent.id, current.get(agent.id));
    rewards.set(agent.id, rewards.get(agent.id) - 5);
    sim.collisions += 1;
  }

  for (const agent of sim.agents) {
    const before = current.get(agent.id);
    const next = proposed.get(agent.id);
    agent.row = next.row; agent.col = next.col; agent.path.push({ row: agent.row, col: agent.col });
    const cell = key(agent.row, agent.col);
    if (!same(before, next) && !sim.visited.has(cell)) { sim.visited.add(cell); rewards.set(agent.id, rewards.get(agent.id) + 1); }
    for (const target of sim.targets) if (!target.found && target.row === agent.row && target.col === agent.col) {
      target.found = true; rewards.set(agent.id, rewards.get(agent.id) + 10); log(`${agent.id} found target at (${target.row}, ${target.col})`);
    }
    agent.reward += rewards.get(agent.id);
    sim.totalReward += rewards.get(agent.id);
  }

  sim.done = sim.targets.every((t) => t.found) || sim.step >= sim.config.maxSteps;
  if (sim.done) { stop(); log(sim.targets.every((t) => t.found) ? "Episode complete" : "Max steps reached"); }
  record(); render();
}

function chooseActions() {
  const actions = new Map();
  for (const agent of sim.agents) {
    if (sim.config.policy === "random") actions.set(agent.id, sim.rng.int(5));
    else if (sim.config.policy === "frontier") actions.set(agent.id, frontier(agent));
    else actions.set(agent.id, greedy(agent));
  }
  return actions;
}

function greedy(agent) {
  const targets = sim.targets.filter((t) => !t.found);
  if (!targets.length) return 0;
  targets.sort((a, b) => manhattan(agent, a) - manhattan(agent, b));
  return actionToward(agent, targets[0]);
}

function frontier(agent) {
  const cells = [
    { action: 1, row: agent.row - 1, col: agent.col }, { action: 2, row: agent.row + 1, col: agent.col },
    { action: 3, row: agent.row, col: agent.col - 1 }, { action: 4, row: agent.row, col: agent.col + 1 },
  ].filter((c) => inBounds(c.row, c.col) && !sim.obstacles.has(key(c.row, c.col)));
  const unvisited = cells.filter((c) => !sim.visited.has(key(c.row, c.col)));
  if (unvisited.length) return unvisited[sim.rng.int(unvisited.length)].action;
  return greedy(agent);
}

function actionToward(from, to) {
  const dr = to.row - from.row, dc = to.col - from.col;
  if (Math.abs(dr) >= Math.abs(dc) && dr !== 0) return dr > 0 ? 2 : 1;
  if (dc !== 0) return dc > 0 ? 4 : 3;
  return 0;
}

function render() { renderWorld(); renderMetrics(); renderAgents(); renderLogs(); }

function renderWorld() {
  const dpr = window.devicePixelRatio || 1;
  const rect = els.canvas.getBoundingClientRect();
  els.canvas.width = Math.max(600, Math.floor(rect.width * dpr));
  els.canvas.height = Math.max(600, Math.floor(rect.height * dpr));
  const width = els.canvas.width, height = els.canvas.height;
  const size = sim.config.gridSize, pad = 24 * dpr, board = Math.min(width, height) - pad * 2;
  const cell = board / size, left = (width - board) / 2, top = (height - board) / 2;
  ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, width, height);

  for (let row = 0; row < size; row += 1) for (let col = 0; col < size; col += 1) {
    const x = left + col * cell, y = top + row * cell;
    ctx.fillStyle = sim.visited.has(key(row, col)) ? COLORS.visited : "#fff";
    ctx.fillRect(x, y, cell, cell);
    if (sim.obstacles.has(key(row, col))) { ctx.fillStyle = COLORS.obstacle; ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2); }
  }
  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = Math.max(1, dpr);
  for (let i = 0; i <= size; i += 1) {
    ctx.beginPath(); ctx.moveTo(left + i * cell, top); ctx.lineTo(left + i * cell, top + board); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(left, top + i * cell); ctx.lineTo(left + board, top + i * cell); ctx.stroke();
  }
  for (const target of sim.targets) {
    ctx.fillStyle = target.found ? COLORS.foundTarget : COLORS.target;
    ctx.beginPath(); ctx.arc(left + target.col * cell + cell / 2, top + target.row * cell + cell / 2, cell * .24, 0, Math.PI * 2); ctx.fill();
  }
  sim.agents.forEach((agent, index) => {
    const x = left + agent.col * cell + cell / 2, y = top + agent.row * cell + cell / 2;
    ctx.fillStyle = COLORS.agents[index % COLORS.agents.length]; ctx.beginPath(); ctx.arc(x, y, cell * .34, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = `${Math.max(10, cell * .32)}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(index), x, y);
  });
}

function renderMetrics() {
  const found = sim.targets.filter((t) => t.found).length;
  const traversable = sim.config.gridSize * sim.config.gridSize - sim.obstacles.size;
  const coverage = sim.visited.size / Math.max(1, traversable);
  els.metricStep.textContent = `${sim.step} / ${sim.config.maxSteps}`;
  els.metricReward.textContent = sim.totalReward.toFixed(1);
  els.metricTargets.textContent = `${found} / ${sim.targets.length}`;
  els.metricCollisions.textContent = String(sim.collisions);
  els.metricCoverage.textContent = `${Math.round(coverage * 100)}%`;
  els.metricMeanReward.textContent = (sim.totalReward / Math.max(1, sim.step)).toFixed(2);
  els.runState.textContent = sim.done ? "Done" : timer ? "Running" : "Ready";
  els.runState.classList.toggle("running", Boolean(timer));
}

function renderAgents() {
  els.agentTable.innerHTML = sim.agents.map((a) => `<tr><td>${a.id}</td><td>${a.row}</td><td>${a.col}</td><td>${a.reward.toFixed(1)}</td></tr>`).join("");
}
function renderLogs() { els.eventLog.innerHTML = logs.map((item) => `<li>${item}</li>`).join(""); }
function record() { const traversable = sim.config.gridSize * sim.config.gridSize - sim.obstacles.size; history.push({ step: sim.step, totalReward: sim.totalReward, coverage: sim.visited.size / Math.max(1, traversable) }); }
function play() { if (timer) { stop(); render(); return; } if (sim.done) resetWorld(); els.playBtn.textContent = "Pause"; timer = setInterval(stepWorld, Math.max(20, 420 - Number(els.speed.value) * 13)); render(); }
function stop() { if (timer) clearInterval(timer); timer = null; els.playBtn.textContent = "Run"; }
function runEpisode() { stop(); while (sim && !sim.done) stepWorld(); }
function exportCsv() { const rows = ["step,total_reward,coverage_ratio", ...history.map((h) => `${h.step},${h.totalReward.toFixed(4)},${h.coverage.toFixed(6)}`)]; const blob = new Blob([rows.join("\n")], { type: "text/csv" }); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = `swarmgym_studio_${Date.now()}.csv`; link.click(); URL.revokeObjectURL(url); }
function log(message) { logs.unshift(`[${sim?.step ?? 0}] ${message}`); logs = logs.slice(0, 30); }
function key(row, col) { return `${row},${col}`; }
function inBounds(row, col) { return row >= 0 && col >= 0 && row < sim.config.gridSize && col < sim.config.gridSize; }
function same(a, b) { return a.row === b.row && a.col === b.col; }
function manhattan(a, b) { return Math.abs(a.row - b.row) + Math.abs(a.col - b.col); }
function intValue(el) { return Number.parseInt(el.value, 10) || 0; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function shuffle(items, rng) { for (let i = items.length - 1; i > 0; i -= 1) { const j = rng.int(i + 1); [items[i], items[j]] = [items[j], items[i]]; } }

[els.gridSize, els.numAgents, els.numTargets, els.numObstacles, els.maxSteps, els.seed].forEach((el) => el.addEventListener("change", resetWorld));
els.policy.addEventListener("change", resetWorld);
els.resetBtn.addEventListener("click", resetWorld);
els.stepBtn.addEventListener("click", stepWorld);
els.playBtn.addEventListener("click", play);
els.episodeBtn.addEventListener("click", runEpisode);
els.exportBtn.addEventListener("click", exportCsv);
window.addEventListener("resize", render);
resetWorld();
