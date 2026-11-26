let currentSettings = { base_speed: 30, max_diff: 25, left_trim: 1, right_trim: 1 };
let previewEnabled = true;
let activeInput = null;
let heartbeatTimer = null;

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  return res.json();
}

function fillSelect(id, items, prefer) {
  const el = document.getElementById(id);
  const prev = el.value;
  el.innerHTML = '';
  if (!items || items.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(none found)';
    el.appendChild(opt);
    return;
  }
  items.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = item;
    el.appendChild(opt);
  });
  if (prev && items.includes(prev)) el.value = prev;
  else if (prefer) {
    const match = items.find(x => x.endsWith(prefer));
    if (match) el.value = match;
  }
}

async function loadAssets() {
  const data = await (await fetch('/api/assets')).json();
  fillSelect('selPPO', data.ppo_models, 'ppo_1stackframe.zip');
  fillSelect('selPTH', data.pth_models, 'model_dagger_3.pth');
  fillSelect('selScenario', data.scenarios);
  applySettingsUI(data.settings, data.preview_enabled);
}

function applySettingsUI(s, prevOn) {
  if (!s) return;
  currentSettings = s;
  document.getElementById('val_base_speed').textContent = s.base_speed.toFixed(1) + '%';
  document.getElementById('val_max_diff').textContent = s.max_diff.toFixed(1) + '%';
  document.getElementById('val_left_trim').textContent = s.left_trim.toFixed(2);
  document.getElementById('val_right_trim').textContent = s.right_trim.toFixed(2);
  if (typeof prevOn === 'boolean' && prevOn !== previewEnabled) {
    setPreviewUI(prevOn);
  }
}

function setPreviewUI(on) {
  previewEnabled = on;
  const img = document.getElementById('camImg');
  const off = document.getElementById('camOff');
  const btn = document.getElementById('btnPreview');
  btn.textContent = on ? 'Preview: ON' : 'Preview: OFF';
  btn.classList.toggle('toggle-on', on);
  if (on) {
    off.style.display = 'none';
    img.style.display = 'block';
    img.src = '/stream.mjpg?t=' + Date.now();
  } else {
    img.src = '';
    img.style.display = 'none';
    off.style.display = 'block';
  }
}

async function togglePreview() {
  const next = !previewEnabled;
  setPreviewUI(next);
  await postJSON('/api/settings', { preview_enabled: next });
}

async function nudge(key, delta) {
  const val = +(currentSettings[key] + delta).toFixed(2);
  const res = await postJSON('/api/settings', { [key]: val });
  applySettingsUI(res.settings, res.preview_enabled);
}

async function resetSettings() {
  const res = await postJSON('/api/settings', { reset: true });
  applySettingsUI(res.settings, res.preview_enabled);
}

async function startMode(payload) {
  await postJSON('/api/start', payload);
  pollStatus();
}
function startPPO() {
  startMode({ mode: 'ppo', model: document.getElementById('selPPO').value });
}
function startDagger() {
  startMode({
    mode: 'dagger',
    model: document.getElementById('selPTH').value,
    save_dagger: document.getElementById('chkSaveDagger').checked,
    session_name: document.getElementById('txtSession').value
  });
}
function startCollect() {
  startMode({ mode: 'collect', scenario: document.getElementById('selScenario').value });
}
function startCalibrate() {
  startMode({
    mode: 'calibrate',
    cal_mode: document.getElementById('selCalMode').value,
    duration: parseFloat(document.getElementById('selCalSec').value)
  });
}
async function stopMode() {
  setActiveInput(null);
  await postJSON('/api/stop', {});
  pollStatus();
}

// ---- Drive input with monotonic seq + non-overlapping deadman heartbeat ----
let inputSeq = 0;
let heartbeatInFlight = false;

function sendInputNow(key, isHeartbeat = false) {
  if (isHeartbeat && heartbeatInFlight) return;
  const seq = ++inputSeq;
  if (isHeartbeat) heartbeatInFlight = true;
  fetch('/api/input', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, seq }),
    keepalive: true
  }).catch(() => {}).finally(() => {
    if (isHeartbeat) heartbeatInFlight = false;
  });
}

function setActiveInput(key) {
  if (activeInput === key) return;
  activeInput = key;
  ['forward', 'left', 'right', 'backward'].forEach(k => {
    const b = document.getElementById('pad-' + k);
    if (b) b.classList.toggle('active', k === key);
  });
  sendInputNow(key, false);
  if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
  if (key !== null) {
    heartbeatTimer = setInterval(() => sendInputNow(activeInput, true), 120);
  }
}

const keyMap = {
  ArrowUp: 'forward', KeyW: 'forward',
  ArrowLeft: 'left', KeyA: 'left',
  ArrowRight: 'right', KeyD: 'right',
  ArrowDown: 'backward', KeyS: 'backward'
};
const pressedCodes = new Set();

function resolveKeyboardInput() {
  for (const k of ['backward', 'left', 'right', 'forward']) {
    for (const code of pressedCodes) {
      if (keyMap[code] === k) return k;
    }
  }
  return null;
}

window.addEventListener('keydown', (e) => {
  if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
  if (e.code === 'Space' || e.code === 'Escape') {
    e.preventDefault();
    pressedCodes.clear();
    stopMode();
    return;
  }
  if (keyMap[e.code]) {
    e.preventDefault();
    if (e.repeat) return;
    pressedCodes.add(e.code);
    setActiveInput(resolveKeyboardInput());
  }
});

window.addEventListener('keyup', (e) => {
  if (keyMap[e.code]) {
    pressedCodes.delete(e.code);
    setActiveInput(resolveKeyboardInput());
  }
});

window.addEventListener('blur', () => {
  pressedCodes.clear();
  setActiveInput(null);
});

document.querySelectorAll('.pad button[data-key]').forEach(btn => {
  const k = btn.dataset.key;
  btn.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    btn.setPointerCapture(e.pointerId);
    setActiveInput(k);
  });
  const release = (e) => {
    e.preventDefault();
    if (activeInput === k) setActiveInput(null);
  };
  btn.addEventListener('pointerup', release);
  btn.addEventListener('pointercancel', release);
});

async function pollStatus() {
  try {
    const s = await (await fetch('/api/status')).json();
    const banner = document.getElementById('banner');
    banner.className = 'banner-' + (s.banner || 'idle');
    document.getElementById('hdrTitle').textContent = s.title || 'PI CAR';
    document.getElementById('hdrDetail').textContent = s.detail || '';
    document.getElementById('hdrAction').textContent = s.action || 'STOPPED';

    document.getElementById('leftPwm').textContent = s.left_pwm.toFixed(1) + '%';
    document.getElementById('rightPwm').textContent = s.right_pwm.toFixed(1) + '%';
    const leftBar = document.getElementById('leftBar');
    const rightBar = document.getElementById('rightBar');
    leftBar.style.width = Math.min(100, Math.abs(s.left_pwm)) + '%';
    rightBar.style.width = Math.min(100, Math.abs(s.right_pwm)) + '%';
    leftBar.classList.toggle('reverse', s.left_pwm < 0);
    rightBar.classList.toggle('reverse', s.right_pwm < 0);

    // Update S/Down button label dynamically depending on mode
    const revBtn = document.getElementById('pad-backward');
    if (s.mode === 'ppo' || s.mode === 'dagger') {
      revBtn.innerHTML = 'BRAKE (S/&darr;)';
    } else {
      revBtn.innerHTML = '&#9660; REV (S/&darr;)';
    }

    let parts = [];
    if (s.mode === 'ppo') {
      parts.push(`Steering: ${s.extra.steering ?? 0}`);
      parts.push(`Infer: ${s.infer_ms} ms`);
      parts.push(`Overruns: ${s.overruns}`);
    } else if (s.mode === 'dagger') {
      parts.push(`AI: ${s.extra.ai_pred || '-'}`);
      parts.push(`Saved: ${s.extra.saved_session ?? 0} (Total: ${s.extra.dataset_total ?? 0})`);
      parts.push(`Override: ${s.extra.override_s ?? 0}s`);
      parts.push(`Infer: ${s.infer_ms} ms`);
    } else if (s.mode === 'collect') {
      parts.push(`F: ${s.extra.forward ?? 0} | L: ${s.extra.left ?? 0} | R: ${s.extra.right ?? 0}`);
      parts.push(`Session: ${s.extra.total ?? 0} (Folder: ${s.extra.folder_total ?? 0})`);
    } else if (s.mode === 'calibrate') {
      parts.push(`${s.extra.phase || ''} | Remaining: ${s.extra.remaining_s ?? 0}s`);
    } else {
      parts.push(`Mode: ${s.mode} | Overruns: ${s.overruns}`);
    }
    document.getElementById('extraLine').textContent = parts.join('  •  ');

    const toast = document.getElementById('statusToast');
    toast.textContent = s.message || 'Ready.';
    toast.classList.toggle('error', !!s.error);

    ['manual', 'ppo', 'dagger', 'collect', 'calibrate'].forEach(m => {
      document.getElementById('card-' + m).classList.toggle('running', s.mode === m);
    });
    applySettingsUI(s.settings, s.preview_enabled);
  } catch (err) {}
}

let pollTimer = null;
async function pollStatusLoop() {
  await pollStatus();
  pollTimer = setTimeout(pollStatusLoop, 250);
}

loadAssets();
pollStatusLoop();
