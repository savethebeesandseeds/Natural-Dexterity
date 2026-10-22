const $ = id => document.getElementById(id);
const number = (value, digits = 0) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits});
const duration = value => {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  const seconds = Math.max(0, Math.round(value));
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60].map(v => String(v).padStart(2, '0')).join(':');
};
const text = (id, value) => { $(id).textContent = value; };
let previousConfig = '';
let previousResults = '';
let previousExperiments = '';
let previousComparisons = '';
const initialization = c => c.training_mode === 'shared' ? 'Shared encoder / separate heads' : ({cold_start: 'Scratch', feature_transfer_fresh_heads_and_adam: 'Pong features / fresh heads', model_and_adam_with_fresh_environments_and_rng: 'Checkpoint continuation'}[c.initialization_mode || c.continuation] || c.initialization_mode || '—');
function renderExperiments(experiments) {
  const runs = experiments.runs || [];
  $('experiment-panel').hidden = !runs.length;
  const serialized = JSON.stringify(runs);
  if (serialized === previousExperiments) return;
  previousExperiments = serialized;
  $('experiment-results').replaceChildren(...runs.map(run => {
    const tr = document.createElement('tr'); tr.title = run.run_dir || '';
    for (const value of [run.game_title || run.game_id, run.initialization, number(run.target_decisions), number(run.mean_return, 2), number(run.truncated_games), duration(run.training_seconds), run.passed ? 'PASS' : 'Below target']) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
}
function renderComparisons(comparisons) {
  const reports = comparisons.reports || [];
  $('comparison-panel').hidden = !reports.length;
  const serialized = JSON.stringify(reports);
  if (serialized === previousComparisons) return;
  previousComparisons = serialized;
  $('comparison-results').replaceChildren(...reports.map(report => {
    const tr = document.createElement('tr'); tr.title = 'Report SHA-256: ' + report.report_sha256;
    for (const value of [report.game_title || report.game_id, number(report.scratch_mean, 2), number(report.transfer_mean, 2), number(report.delta, 2), number(report.scratch_truncated) + ' / ' + number(report.transfer_truncated), report.full_game_comparison ? '20 full games' : 'Frame-capped episodes']) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
  const first = reports[0];
  text('comparison-note', first ? number(first.target_decisions) + ' target decisions per policy; Pong source pretraining cost: ' + number(first.source_pretraining_decisions) + ' decisions. One training seed per game. Capped returns do not establish full-game superiority.' : '');
}
function renderShared(progress, retention, games) {
  const active = progress.training_mode === 'shared';
  $('shared-panel').hidden = !active;
  if (!active) return;
  $('shared-results').replaceChildren(...Object.entries(progress.per_game || {}).map(([id, p]) => {
    const e = retention.games?.[id] || {}, tr = document.createElement('tr');
    const title = games.find(game => game.id === id)?.title || id;
    const criterion = e.complete == null ? 'Pending' : !e.complete ? 'Incomplete' : e.before_passed ? e.criterion_retained ? 'Retained' : 'Not retained' : e.after_passed ? 'Target reached' : 'Target missed';
    for (const value of [title, number(p.agent_decisions) + ' / ' + number(p.target_steps), number(p.recent_mean_raw_return, 2) + ' · n=' + number(p.recent_mean_sample_count), number(e.before_mean_raw_return, 2), number(e.after_mean_raw_return, 2), number(e.paired_mean_raw_return_delta, 2), number(e.before_truncated_games) + ' / ' + number(e.after_truncated_games), criterion]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
}
function renderResults(games, proofs, activeGame, evaluation, experiments) {
  const results = {};
  for (const run of experiments.runs || []) results[run.game_id] = run;
  if (evaluation.status === 'complete') results[activeGame.id] = evaluation;
  Object.assign(results, proofs);
  const key = JSON.stringify([games, results]);
  if (key === previousResults) return;
  previousResults = key;
  $('game-results').replaceChildren(...games.map(game => {
    const result = results[game.id], criterion = game.criterion;
    const target = game.id === 'pong' ? '≥' + criterion.min_positive_games + '/' + criterion.episodes + ' wins, mean >0' : 'Mean ≥' + number(criterion.mean_raw_return_threshold);
    const measured = !result ? 'Pending' : (result.passed ? 'PASS' : 'Below target') + ' · mean ' + number(result.mean_return, 2) + (result.truncated_games ? ' · ' + number(result.truncated_games) + ' capped' : '');
    const tr = document.createElement('tr');
    for (const value of [game.title, target, measured, result ? number(result.episodes_completed) : '—']) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
}

function renderConfig(c) {
  const serialized = JSON.stringify(c);
  if (serialized === previousConfig) return;
  previousConfig = serialized;
  const p = c.preprocessing || {};
  const shared = c.training_mode === 'shared';
  const actions = shared ? (c.shared_games || []).map(game => game.title + ': ' + number(game.expected_actions)).join(' / ') : number(c.action_count);
  const rows = [
    ['Algorithm', c.algorithm ?? '—'], ['Environments', number(c.environments)],
    ['Rollout / environment', number(c.rollout_steps)], ['Minibatch', number(c.minibatch_size)], ['Epochs / update', number(c.epochs)],
    ['Learning rate', c.learning_rate ?? '—'], ['Discount γ', c.gamma ?? '—'], ['GAE λ', c.gae_lambda ?? '—'],
    ['PPO clip', c.clip_range ?? '—'], ['Entropy coefficient', c.entropy_coefficient ?? '—'],
    ['Observation', c.preprocessing ? '4 × 84 × 84 pixels' : '—'], ['Action repeat', number(p.frame_skip)],
    ['Sticky action probability', p.sticky_action_probability == null ? '—' : number(p.sticky_action_probability * 100) + '%'],
    [shared ? 'Actions / game' : 'Actions', actions], ['Session seed', number(c.seed)], ['Initialization', initialization(c)]
  ];
  if (c.training_mode === 'shared') rows.push(
    ['Environments / game', number(c.environments_per_game)],
    ['Evaluation time cap / game', duration(c.evaluation_max_seconds_per_game)]
  );
  $('config-values').replaceChildren(...rows.map(([label, value]) => {
    const row = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = label; dd.textContent = String(value); row.append(dt, dd); return row;
  }));
  text('runtime', 'C++20 · LibTorch ' + (c.torch_version ?? '—'));
  text('parent-steps', number(c.previous_trained_steps ?? 0));
  text('parent-hash', c.parent_checkpoint_sha256 ?? 'None');
  text('source-steps', number(c.transfer?.source_pretraining_decisions));
  text('source-hash', c.transfer?.source_checkpoint_sha256 ?? 'None');
  text('rom-hash', c.rom_sha256 ?? '—');
}

async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    if (!response.ok) throw Error('Status request failed: HTTP ' + response.status);
    const data = await response.json(), p = data.progress || {}, c = data.config || {}, playback = data.playback || {};
    const game = data.game || c.game_spec || {id: 'pong', title: 'Pong', criterion: {episodes: 20, min_positive_games: 16, mean_raw_return_threshold: 0, mean_strict: true}};
    const criterion = game.criterion, proofs = data.proofs || {};
    const currentEvaluation = data.evaluation || {};
    const e = currentEvaluation;
    const shared = c.training_mode === 'shared';
    const title = shared ? (c.shared_games || []).map(entry => entry.title).join(' + ') : game.title;
    text('game-title', '/ ' + title); document.title = 'Atari / ' + title;
    $('frame').alt = 'Live Atari ' + (playback.game_title || game.title) + ' game';
    const steps = shared ? p.steps : p.cumulative_agent_decisions ?? p.steps;
    const target = p.target_steps == null ? null : (shared ? 0 : p.previous_trained_steps || 0) + p.target_steps;
    const percent = steps == null || !target ? null : steps / target * 100;
    const states = {initializing: 'Initializing', training: 'Training', evaluating_before: 'Evaluating initial policies', evaluating: 'Evaluating', complete: 'Complete', evaluation_incomplete: 'Evaluation incomplete', failed: 'Failed', stopped: 'Stopped', interrupted: 'Interrupted', complete_without_evaluation: 'Complete · no evaluation'};
    text('state', states[p.status] || p.status || 'Idle');
    text('updated', new Date().toLocaleTimeString(undefined, {hour12: false}));
    text('run', data.run ?? 'No run');
    text('device', (p.device ?? c.device ?? '—').toUpperCase());
    text('budget', number(steps) + ' / ' + number(target) + (shared ? ' new decisions · all games' : ' decisions'));
    text('percent', percent == null ? '—' : number(percent, 1) + '%');
    $('progress').value = steps ?? 0; $('progress').max = target || 1;
    text('fps', p.status === 'interrupted' ? '—' : number(p.decisions_per_second ?? p.fps));
    text('mean', number(p.recent_mean_raw_return ?? p.recent_mean_reward, 2));
    text('mean-label', (shared ? 'Pong train return' : 'Train return') + (p.recent_mean_sample_count == null ? ' · current block' : ' · n=' + number(p.recent_mean_sample_count)));
    text('updates', number(p.updates)); text('episodes', number(p.episodes));
    text('session-steps', number(p.steps));
    text('elapsed', duration(p.elapsed_seconds) + ' / ' + duration(c.max_seconds));
    const speed = p.decisions_per_second ?? p.fps;
    const decisionEta = speed > 0 && target != null && steps != null ? Math.max(0, target - steps) / speed : null;
    const timeEta = c.max_seconds == null || p.elapsed_seconds == null ? null : Math.max(0, c.max_seconds - p.elapsed_seconds);
    const eta = decisionEta == null ? timeEta : timeEta == null ? decisionEta : Math.min(decisionEta, timeEta);
    text('eta', p.status === 'training' ? (eta == null ? '—' : '≈ ' + duration(eta)) : '—');
    text('checkpoint-label', shared ? 'Saved Pong checkpoint · Pong decisions' : 'Saved checkpoint · total decisions');
    text('checkpoint', number(p.checkpoint_steps));
    text('loss', p.status === 'training' ? number(p.loss, 6) : '—');
    text('policy', playback.policy ?? 'Waiting for checkpoint');
    text('play-score', number(playback.score)); text('play-episode', number(playback.episode));
    const hasEvaluation = Object.keys(e).length > 0;
    text('evaluation-label', shared ? game.title + ' final evaluation' : 'Final evaluation');
    text('verdict', !hasEvaluation ? 'Pending' : e.passed ? 'PASS' : e.status === 'incomplete' ? 'Incomplete' : e.verdict === 'target_not_met' ? 'FAIL' : 'Insufficient data');
    text('eval-games', number(e.episodes_completed) + ' / ' + number(e.episodes_requested ?? criterion.episodes));
    text('eval-positive-label', game.id === 'pong' ? 'Wins · required ≥' + number(criterion.min_positive_games) : 'Positive-return games');
    text('eval-mean-label', 'Mean return · required ' + (criterion.mean_strict ? '>' : '≥') + number(criterion.mean_raw_return_threshold));
    text('eval-wins', number(game.id === 'pong' ? e.wins : e.positive_games)); text('eval-mean', number(e.mean_return, 2)); text('eval-truncated', number(e.truncated_games));
    text('evaluation-note', !hasEvaluation ? 'Current run · after training · seed start ' + number(c.evaluation_seeds?.[game.id] ?? c.evaluation_seed ?? game.final_eval_seed) + ' · frozen policy · full games' : 'Current run · seed start ' + number(e.seed_start) + ' · ' + (e.elapsed_seconds == null ? number(e.trained_decisions) + ' trained decisions' : 'elapsed ' + duration(e.elapsed_seconds)) + ' · ' + (e.role ?? e.verdict ?? e.status));
    text('eval-hash', e.checkpoint_sha256 ?? '—');
    renderConfig(c);
    renderResults(data.games || [game], proofs, game, currentEvaluation, data.experiments || {});
    renderExperiments(data.experiments || {});
    renderComparisons(data.comparisons || {});
    renderShared(p, data.shared_evaluation || {}, data.games || []);
    const error = playback.error || p.error || (p.status === 'interrupted' ? 'The learner process has stopped. Saved checkpoints and measurements are preserved.' : '') || data.learner_liveness_error || '';
    text('error', error); $('error').hidden = !error; $('state').dataset.error = String(Boolean(error));
  } catch (error) {
    text('error', error.message); $('error').hidden = false;
    text('state', 'Disconnected'); $('state').dataset.error = 'true';
  }
  setTimeout(refresh, 2000);
}
function nextFrame() {
  const frame = $('frame');
  frame.onload = () => { $('loading').classList.add('hidden'); setTimeout(nextFrame, 65); };
  frame.onerror = () => { $('loading').textContent = 'Frame unavailable · retrying'; $('loading').classList.remove('hidden'); setTimeout(nextFrame, 1000); };
  frame.src = '/frame.png?t=' + Date.now();
}
refresh(); nextFrame();
