/**
 * BrainBeats — Real-Time Closed-Loop Biofeedback Frontend Controller
 */

// Application State
const state = {
  sessionId: null,
  isRunning: false,
  preset: 'calm',
  targetGoal: 'calm',
  sessionStartTime: null,
  timerInterval: null,
  socket: null,
  volume: 0.8,
  isMuted: false,
  synthEnabled: true,
  reverbEnabled: true,
  audioCtx: null,
  reverbNode: null,
  masterGain: null,
  history: [],
  latestFrame: null,
  activeNotes: [],
  oscilloscopeData: Array(8).fill(0).map(() => Array(64).fill(0)),
};

// Scale degrees map for harmonic display
const SCALE_NOTES = {
  0: ['C', 'D', 'E', 'F', 'G', 'A', 'B'], // C Major
  1: ['C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#'],
  2: ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
  3: ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
  4: ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
  5: ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
  6: ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'],
  7: ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
  8: ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
  9: ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
  10: ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
  11: ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#']
};

const MINOR_SCALE_NOTES = {
  0: ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
  9: ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
  2: ['D', 'E', 'F', 'G', 'A', 'Bb', 'C'],
  4: ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
  7: ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'],
  11: ['B', 'C#', 'D', 'E', 'F#', 'G', 'A']
};

// DOM Elements
const elements = {
  connectionBadge: document.getElementById('connectionBadge'),
  connectionText: document.getElementById('connectionText'),
  sessionStateBadge: document.getElementById('sessionStateBadge'),
  sessionStateDot: document.getElementById('sessionStateDot'),
  sessionStateText: document.getElementById('sessionStateText'),
  sessionTimer: document.getElementById('sessionTimer'),
  sessionToggleBtn: document.getElementById('sessionToggleBtn'),
  sessionBtnLabel: document.getElementById('sessionBtnLabel'),
  muteBtn: document.getElementById('muteBtn'),
  muteIcon: document.getElementById('muteIcon'),
  volumeSlider: document.getElementById('volumeSlider'),
  btnPresetCalm: document.getElementById('btnPresetCalm'),
  btnPresetNeutral: document.getElementById('btnPresetNeutral'),
  btnPresetAroused: document.getElementById('btnPresetAroused'),
  goalSelect: document.getElementById('goalSelect'),
  stepManualBtn: document.getElementById('stepManualBtn'),
  downloadMidiBtn: document.getElementById('downloadMidiBtn'),
  downloadWavBtn: document.getElementById('downloadWavBtn'),
  exportSessionBtn: document.getElementById('exportSessionBtn'),
  synthToggleBtn: document.getElementById('synthToggleBtn'),
  reverbToggleBtn: document.getElementById('reverbToggleBtn'),
  alphaRatioLabel: document.getElementById('alphaRatioLabel'),
  valDelta: document.getElementById('valDelta'),
  valTheta: document.getElementById('valTheta'),
  valAlpha: document.getElementById('valAlpha'),
  valBeta: document.getElementById('valBeta'),
  valGamma: document.getElementById('valGamma'),
  barDelta: document.getElementById('barDelta'),
  barTheta: document.getElementById('barTheta'),
  barAlpha: document.getElementById('barAlpha'),
  barBeta: document.getElementById('barBeta'),
  barGamma: document.getElementById('barGamma'),
  emotionBadge: document.getElementById('emotionBadge'),
  valenceVal: document.getElementById('valenceVal'),
  arousalVal: document.getElementById('arousalVal'),
  latentBarsContainer: document.getElementById('latentBarsContainer'),
  scoreRingFg: document.getElementById('scoreRingFg'),
  bioScoreText: document.getElementById('bioScoreText'),
  tempoVal: document.getElementById('tempoVal'),
  densityVal: document.getElementById('densityVal'),
  keyVal: document.getElementById('keyVal'),
  scaleNotesVal: document.getElementById('scaleNotesVal'),
  metronomePulse: document.getElementById('metronomePulse'),
  notesCountBadge: document.getElementById('notesCountBadge'),
  playingNoteName: document.getElementById('playingNoteName'),
  audioPlayer: document.getElementById('audioPlayer'),
  totalFramesVal: document.getElementById('totalFramesVal'),
  dominantBandVal: document.getElementById('dominantBandVal'),
  meanValenceVal: document.getElementById('meanValenceVal'),
  meanTempoVal: document.getElementById('meanTempoVal'),
  toast: document.getElementById('toast'),
  eegCanvas: document.getElementById('eegCanvas'),
  circumplexCanvas: document.getElementById('circumplexCanvas'),
  pianoRollCanvas: document.getElementById('pianoRollCanvas'),
  historyChartCanvas: document.getElementById('historyChartCanvas'),
};

// Canvas Contexts
const eegCtx = elements.eegCanvas.getContext('2d');
const circumplexCtx = elements.circumplexCanvas.getContext('2d');
const pianoRollCtx = elements.pianoRollCanvas.getContext('2d');
const historyCtx = elements.historyChartCanvas.getContext('2d');

// --- Initialization ---
window.addEventListener('DOMContentLoaded', () => {
  initLatentBars();
  initAudioContext();
  setupEventListeners();
  initWebSocket();
  startCanvasRenderLoops();
  checkCurrentSession();
});

// Toast Notifications
function showToast(message, duration = 3000) {
  elements.toast.textContent = message;
  elements.toast.classList.remove('hidden');
  setTimeout(() => elements.toast.classList.add('hidden'), duration);
}

// Latent Bars Setup (12 dimensions)
function initLatentBars() {
  elements.latentBarsContainer.innerHTML = '';
  const labels = ['z0 (Tempo)', 'z1 (Density)', 'z2 (Key)', 'z3 (Mode)', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11'];
  for (let i = 0; i < 12; i++) {
    const item = document.createElement('div');
    item.className = 'z-bar-item';
    item.innerHTML = `
      <span class="z-bar-label">${labels[i].split(' ')[0]}</span>
      <div class="z-bar-track">
        <div class="z-bar-fill" id="zFill${i}" style="height: 50%; top: 25%;"></div>
      </div>
      <span class="z-bar-val" id="zVal${i}">0.0</span>
    `;
    elements.latentBarsContainer.appendChild(item);
  }
}

// Web Audio API Synthesizer Setup
function initAudioContext() {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    state.audioCtx = new AudioContext();
    state.masterGain = state.audioCtx.createGain();
    state.masterGain.gain.setValueAtTime(state.volume, state.audioCtx.currentTime);

    // Create algorithmic reverb using a delay network
    const delay = state.audioCtx.createDelay();
    delay.delayTime.value = 0.22;
    const feedback = state.audioCtx.createGain();
    feedback.gain.value = 0.35;
    delay.connect(feedback);
    feedback.connect(delay);

    state.reverbNode = delay;
    state.masterGain.connect(state.audioCtx.destination);
    state.reverbNode.connect(state.masterGain);
  } catch (e) {
    console.warn('Web Audio not supported or blocked:', e);
  }
}

function playSynthesizerNote(pitch, duration, velocity = 85, startTimeOffset = 0) {
  if (!state.synthEnabled || state.isMuted || !state.audioCtx) return;
  if (state.audioCtx.state === 'suspended') {
    state.audioCtx.resume();
  }

  const now = state.audioCtx.currentTime + startTimeOffset;
  const freq = 440 * Math.pow(2, (pitch - 69) / 12);

  // Dual oscillator for rich warm acoustic piano/chime timbre
  const osc1 = state.audioCtx.createOscillator();
  const osc2 = state.audioCtx.createOscillator();
  const noteGain = state.audioCtx.createGain();

  osc1.type = 'sine';
  osc2.type = 'triangle';
  osc1.frequency.setValueAtTime(freq, now);
  osc2.frequency.setValueAtTime(freq * 1.002, now); // slight chorus detune

  const velScale = (velocity / 127) * 0.25;
  noteGain.gain.setValueAtTime(0.0001, now);
  noteGain.gain.exponentialRampToValueAtTime(velScale, now + 0.015);
  noteGain.gain.exponentialRampToValueAtTime(velScale * 0.4, now + duration * 0.5);
  noteGain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  osc1.connect(noteGain);
  osc2.connect(noteGain);
  noteGain.connect(state.masterGain);

  if (state.reverbEnabled && state.reverbNode) {
    noteGain.connect(state.reverbNode);
  }

  osc1.start(now);
  osc2.start(now);
  osc1.stop(now + duration + 0.1);
  osc2.stop(now + duration + 0.1);
}

// WebSocket Connection
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  try {
    state.socket = new WebSocket(wsUrl);

    state.socket.onopen = () => {
      elements.connectionBadge.classList.remove('status-offline');
      elements.connectionText.textContent = 'Live Connected';
    };

    state.socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'telemetry') {
          handleTelemetry(payload.data);
        } else if (payload.type === 'status') {
          updateStatus(payload.data);
        }
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    state.socket.onclose = () => {
      elements.connectionText.textContent = 'Disconnected (Retrying)';
      setTimeout(initWebSocket, 3000);
    };

    state.socket.onerror = () => {
      elements.connectionText.textContent = 'Offline';
    };
  } catch (e) {
    console.error('WS init failed:', e);
  }
}

// Telemetry Frame Ingestion
function handleTelemetry(frame) {
  state.latestFrame = frame;
  state.history.push(frame);
  if (state.history.length > 80) state.history.shift();

  // 1. Update EEG Waveforms buffer
  if (frame.raw_eeg_preview) {
    state.oscilloscopeData = frame.raw_eeg_preview;
  }

  // 2. Frequency Band Meters
  const bands = frame.band_powers;
  if (bands) {
    elements.valDelta.textContent = Math.round(bands.delta * 100) + '%';
    elements.valTheta.textContent = Math.round(bands.theta * 100) + '%';
    elements.valAlpha.textContent = Math.round(bands.alpha * 100) + '%';
    elements.valBeta.textContent = Math.round(bands.beta * 100) + '%';
    elements.valGamma.textContent = Math.round(bands.gamma * 100) + '%';

    elements.barDelta.style.width = Math.min(100, Math.round(bands.delta * 100)) + '%';
    elements.barTheta.style.width = Math.min(100, Math.round(bands.theta * 100)) + '%';
    elements.barAlpha.style.width = Math.min(100, Math.round(bands.alpha * 100)) + '%';
    elements.barBeta.style.width = Math.min(100, Math.round(bands.beta * 100)) + '%';
    elements.barGamma.style.width = Math.min(100, Math.round(bands.gamma * 100)) + '%';

    const ratio = (bands.alpha / (bands.beta + 0.001)).toFixed(2);
    elements.alphaRatioLabel.innerHTML = `Alpha/Beta Ratio: <strong>${ratio}</strong>`;
  }

  // 3. Neuro-Emotional Coordinates
  elements.valenceVal.textContent = (frame.valence >= 0 ? '+' : '') + frame.valence.toFixed(2);
  elements.arousalVal.textContent = (frame.arousal >= 0 ? '+' : '') + frame.arousal.toFixed(2);

  let emotion = 'Resting Neutral';
  if (frame.valence > 0.2 && frame.arousal < 0.1) emotion = 'Calm & Meditative 🌿';
  else if (frame.valence > 0.2 && frame.arousal >= 0.1) emotion = 'Joyful & Focused ⚡';
  else if (frame.valence <= 0.2 && frame.arousal >= 0.1) emotion = 'Tense & High Arousal ⚠️';
  else if (frame.valence <= 0.2 && frame.arousal < 0.1) emotion = 'Fatigued / Low Energy 💤';
  elements.emotionBadge.textContent = emotion;

  // 4. Latent Vector Bars
  if (frame.latent) {
    frame.latent.forEach((val, i) => {
      const fill = document.getElementById(`zFill${i}`);
      const txt = document.getElementById(`zVal${i}`);
      if (fill && txt) {
        const height = Math.abs(val) * 50;
        const top = val >= 0 ? 50 - height : 50;
        fill.style.height = `${height}%`;
        fill.style.top = `${top}%`;
        fill.style.background = val >= 0 ? 'var(--neon-teal)' : 'var(--neon-rose)';
        txt.textContent = val.toFixed(2);
      }
    });
  }

  // 5. Biofeedback Score Ring
  const score = Math.round(frame.biofeedback_score || 70);
  elements.bioScoreText.textContent = `${score}%`;
  const offset = 251.2 - (251.2 * score) / 100;
  elements.scoreRingFg.style.strokeDashoffset = offset;

  // 6. Music Macro Controls
  elements.tempoVal.innerHTML = `${Math.round(frame.tempo_bpm)} <small>BPM</small>`;
  elements.densityVal.innerHTML = `${frame.notes_per_beat.toFixed(1)} <small>notes/beat</small>`;
  elements.keyVal.textContent = frame.key_name;

  // Metronome pulse rate
  const beatSec = 60 / frame.tempo_bpm;
  elements.metronomePulse.style.animationDuration = `${beatSec}s`;

  // Scale Notes display
  const root = frame.key_root % 12;
  const scaleList = frame.mode === 'minor'
    ? (MINOR_SCALE_NOTES[root] || ['Scale', 'Minor'])
    : (SCALE_NOTES[root] || ['Scale', 'Major']);
  elements.scaleNotesVal.textContent = scaleList.join(' ');

  // 7. Melodic Notes & Synthesizer Playback
  if (frame.notes && frame.notes.length > 0) {
    elements.notesCountBadge.textContent = `${frame.notes.length} Notes Generated`;
    state.activeNotes = frame.notes;

    // Trigger Web Audio playback sequence
    frame.notes.forEach((note) => {
      playSynthesizerNote(note.pitch, note.duration, note.velocity, note.start);
    });

    // Fallback: If Web Audio is disabled and base64 audio is provided, play audio element
    if (!state.synthEnabled && frame.audio_base64 && !state.isMuted) {
      elements.audioPlayer.src = `data:audio/wav;base64,${frame.audio_base64}`;
      elements.audioPlayer.volume = state.volume;
      elements.audioPlayer.play().catch(() => {});
    }

    elements.playingNoteName.textContent = `Playing: ${frame.notes[0].name}`;
  }

  // 8. Statistics Summary
  elements.totalFramesVal.textContent = frame.frame_id;
  elements.dominantBandVal.textContent = getDominantBand(frame.band_powers);
  const avgVal = (state.history.reduce((a, b) => a + b.valence, 0) / state.history.length).toFixed(2);
  elements.meanValenceVal.textContent = (avgVal >= 0 ? '+' : '') + avgVal;
  const avgTempo = Math.round(state.history.reduce((a, b) => a + b.tempo_bpm, 0) / state.history.length);
  elements.meanTempoVal.textContent = `${avgTempo} BPM`;
}

function getDominantBand(bands) {
  if (!bands) return 'Alpha';
  let maxBand = 'alpha';
  let maxVal = -1;
  for (const [k, v] of Object.entries(bands)) {
    if (v > maxVal) {
      maxVal = v;
      maxBand = k;
    }
  }
  return maxBand.charAt(0).toUpperCase() + maxBand.slice(1);
}

// --- Canvas Render Loops ---
function startCanvasRenderLoops() {
  function render() {
    drawOscilloscope();
    drawCircumplex();
    drawPianoRoll();
    drawHistoryChart();
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

// 1. Oscilloscope Canvas
function drawOscilloscope() {
  const w = elements.eegCanvas.width;
  const h = elements.eegCanvas.height;
  eegCtx.fillStyle = '#04060c';
  eegCtx.fillRect(0, 0, w, h);

  const channels = state.oscilloscopeData;
  const chHeight = h / 8;
  const channelLabels = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'];

  for (let ch = 0; ch < 8; ch++) {
    const yCenter = chHeight * ch + chHeight / 2;

    // Draw baseline
    eegCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    eegCtx.lineWidth = 1;
    eegCtx.beginPath();
    eegCtx.moveTo(40, yCenter);
    eegCtx.lineTo(w, yCenter);
    eegCtx.stroke();

    // Channel label
    eegCtx.fillStyle = 'rgba(0, 242, 254, 0.7)';
    eegCtx.font = '10px JetBrains Mono';
    eegCtx.fillText(channelLabels[ch], 8, yCenter + 3);

    // Waveform line
    const data = channels[ch] || [];
    if (data.length > 1) {
      eegCtx.strokeStyle = ch % 2 === 0 ? '#00f5d4' : '#00f2fe';
      eegCtx.lineWidth = 1.5;
      eegCtx.shadowColor = '#00f2fe';
      eegCtx.shadowBlur = 4;
      eegCtx.beginPath();

      const dx = (w - 45) / (data.length - 1);
      for (let i = 0; i < data.length; i++) {
        const x = 45 + i * dx;
        const val = data[i];
        const y = yCenter - val * (chHeight * 0.35);
        if (i === 0) eegCtx.moveTo(x, y);
        else eegCtx.lineTo(x, y);
      }
      eegCtx.stroke();
      eegCtx.shadowBlur = 0;
    }
  }
}

// 2. Circumplex Emotion Plane Canvas
function drawCircumplex() {
  const w = elements.circumplexCanvas.width;
  const h = elements.circumplexCanvas.height;
  circumplexCtx.fillStyle = '#04060c';
  circumplexCtx.fillRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;

  // Quadrant fills
  circumplexCtx.fillStyle = 'rgba(244, 63, 94, 0.06)'; // Top Left: Tense
  circumplexCtx.fillRect(0, 0, cx, cy);
  circumplexCtx.fillStyle = 'rgba(255, 183, 3, 0.06)'; // Top Right: Excited
  circumplexCtx.fillRect(cx, 0, cx, cy);
  circumplexCtx.fillStyle = 'rgba(100, 116, 139, 0.06)'; // Bottom Left: Fatigued
  circumplexCtx.fillRect(0, cy, cx, cy);
  circumplexCtx.fillStyle = 'rgba(16, 185, 129, 0.08)'; // Bottom Right: Calm
  circumplexCtx.fillRect(cx, cy, cx, cy);

  // Axes
  circumplexCtx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
  circumplexCtx.lineWidth = 1;
  circumplexCtx.beginPath();
  circumplexCtx.moveTo(0, cy);
  circumplexCtx.lineTo(w, cy);
  circumplexCtx.moveTo(cx, 0);
  circumplexCtx.lineTo(cx, h);
  circumplexCtx.stroke();

  // Grid circles
  circumplexCtx.beginPath();
  circumplexCtx.arc(cx, cy, cx * 0.45, 0, Math.PI * 2);
  circumplexCtx.arc(cx, cy, cx * 0.85, 0, Math.PI * 2);
  circumplexCtx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  circumplexCtx.stroke();

  // Quadrant Labels
  circumplexCtx.font = '9px Outfit';
  circumplexCtx.fillStyle = 'rgba(255, 255, 255, 0.35)';
  circumplexCtx.fillText('TENSE / AGITATED', 12, 20);
  circumplexCtx.fillText('ALERT / EXCITED', cx + 12, 20);
  circumplexCtx.fillText('FATIGUED / BORED', 12, h - 12);
  circumplexCtx.fillStyle = 'rgba(16, 185, 129, 0.8)';
  circumplexCtx.fillText('CALM / SERENE ⭐', cx + 12, h - 12);

  // Active Coordinate Point
  if (state.latestFrame) {
    const vx = cx + state.latestFrame.valence * (cx * 0.8);
    const vy = cy - state.latestFrame.arousal * (cy * 0.8);

    // Trail
    circumplexCtx.beginPath();
    circumplexCtx.arc(vx, vy, 14, 0, Math.PI * 2);
    circumplexCtx.fillStyle = 'rgba(0, 245, 212, 0.2)';
    circumplexCtx.fill();

    // Dot
    circumplexCtx.beginPath();
    circumplexCtx.arc(vx, vy, 6, 0, Math.PI * 2);
    circumplexCtx.fillStyle = '#00f5d4';
    circumplexCtx.shadowColor = '#00f5d4';
    circumplexCtx.shadowBlur = 12;
    circumplexCtx.fill();
    circumplexCtx.shadowBlur = 0;
  }
}

// 3. Piano Roll Canvas
function drawPianoRoll() {
  const w = elements.pianoRollCanvas.width;
  const h = elements.pianoRollCanvas.height;
  pianoRollCtx.fillStyle = '#04060c';
  pianoRollCtx.fillRect(0, 0, w, h);

  // Draw grid lines
  pianoRollCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  pianoRollCtx.lineWidth = 1;
  const pitchLines = 12;
  for (let p = 0; p < pitchLines; p++) {
    const y = (h / pitchLines) * p;
    pianoRollCtx.beginPath();
    pianoRollCtx.moveTo(0, y);
    pianoRollCtx.lineTo(w, y);
    pianoRollCtx.stroke();
  }

  // Draw notes
  const notes = state.activeNotes;
  if (!notes || notes.length === 0) return;

  const minPitch = 36;
  const maxPitch = 84;
  const totalDuration = notes.reduce((max, n) => Math.max(max, n.start + n.duration), 6.0);

  notes.forEach((n) => {
    const x = (n.start / totalDuration) * (w - 20) + 10;
    const noteW = Math.max(8, (n.duration / totalDuration) * (w - 20));
    const pitchNorm = 1 - (n.pitch - minPitch) / (maxPitch - minPitch);
    const y = pitchNorm * (h - 20) + 5;

    // Glowing note block
    pianoRollCtx.fillStyle = 'rgba(157, 78, 221, 0.85)';
    pianoRollCtx.shadowColor = '#00f2fe';
    pianoRollCtx.shadowBlur = 8;
    pianoRollCtx.fillRect(x, y, noteW, 8);
    pianoRollCtx.shadowBlur = 0;

    // Note pitch label
    pianoRollCtx.fillStyle = '#ffffff';
    pianoRollCtx.font = '8px JetBrains Mono';
    pianoRollCtx.fillText(n.name, x + 2, y + 7);
  });
}

// 4. History Chart Canvas
function drawHistoryChart() {
  const w = elements.historyChartCanvas.width;
  const h = elements.historyChartCanvas.height;
  historyCtx.fillStyle = '#04060c';
  historyCtx.fillRect(0, 0, w, h);

  if (state.history.length < 2) return;

  const len = state.history.length;
  const dx = w / (len - 1);
  const cy = h / 2;

  // Baseline
  historyCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  historyCtx.lineWidth = 1;
  historyCtx.beginPath();
  historyCtx.moveTo(0, cy);
  historyCtx.lineTo(w, cy);
  historyCtx.stroke();

  // Plot Valence (Teal)
  historyCtx.strokeStyle = '#00f5d4';
  historyCtx.lineWidth = 2;
  historyCtx.beginPath();
  state.history.forEach((f, i) => {
    const x = i * dx;
    const y = cy - f.valence * (h * 0.4);
    if (i === 0) historyCtx.moveTo(x, y);
    else historyCtx.lineTo(x, y);
  });
  historyCtx.stroke();

  // Plot Arousal (Rose)
  historyCtx.strokeStyle = '#f43f5e';
  historyCtx.lineWidth = 2;
  historyCtx.beginPath();
  state.history.forEach((f, i) => {
    const x = i * dx;
    const y = cy - f.arousal * (h * 0.4);
    if (i === 0) historyCtx.moveTo(x, y);
    else historyCtx.lineTo(x, y);
  });
  historyCtx.stroke();

  // Plot Tempo/100 (Amber)
  historyCtx.strokeStyle = '#ffb703';
  historyCtx.lineWidth = 1.5;
  historyCtx.beginPath();
  state.history.forEach((f, i) => {
    const x = i * dx;
    const y = h - (f.tempo_bpm / 180) * (h * 0.85);
    if (i === 0) historyCtx.moveTo(x, y);
    else historyCtx.lineTo(x, y);
  });
  historyCtx.stroke();
}

// --- Event Listeners ---
function setupEventListeners() {
  // Session Toggle
  elements.sessionToggleBtn.addEventListener('click', () => {
    if (state.isRunning) {
      stopSession();
    } else {
      startSession();
    }
  });

  // Preset Buttons
  elements.btnPresetCalm.addEventListener('click', () => switchPreset('calm'));
  elements.btnPresetNeutral.addEventListener('click', () => switchPreset('neutral'));
  elements.btnPresetAroused.addEventListener('click', () => switchPreset('aroused'));

  // Goal Selector
  elements.goalSelect.addEventListener('change', (e) => {
    state.targetGoal = e.target.value;
    fetch('/api/sessions/goal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_goal: state.targetGoal }),
    });
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify({ action: 'set_goal', goal: state.targetGoal }));
    }
    showToast(`Biofeedback goal changed to: ${state.targetGoal}`);
  });

  // Manual Step
  elements.stepManualBtn.addEventListener('click', async () => {
    showToast('Evaluating closed-loop cycle...');
    const res = await fetch('/api/sessions/step', { method: 'POST' });
    const data = await res.json();
    handleTelemetry(data);
  });

  // Download MIDI & WAV
  elements.downloadMidiBtn.addEventListener('click', () => {
    if (!state.latestFrame || !state.latestFrame.midi_filename) {
      showToast('No MIDI clip recorded yet. Start a session first.');
      return;
    }
    const sessId = state.sessionId || state.latestFrame.session_id;
    window.open(`/api/sessions/${sessId}/midi/${state.latestFrame.midi_filename}`, '_blank');
  });

  elements.downloadWavBtn.addEventListener('click', () => {
    if (!state.latestFrame || !state.latestFrame.wav_filename) {
      showToast('No WAV clip rendered yet. Start a session first.');
      return;
    }
    const sessId = state.sessionId || state.latestFrame.session_id;
    window.open(`/api/sessions/${sessId}/audio/${state.latestFrame.wav_filename}`, '_blank');
  });

  // Export Session JSON
  elements.exportSessionBtn.addEventListener('click', () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.history, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `brainbeats_session_${Date.now()}.json`);
    dlAnchor.click();
  });

  // Volume & Mute
  elements.muteBtn.addEventListener('click', () => {
    state.isMuted = !state.isMuted;
    elements.muteIcon.textContent = state.isMuted ? '🔇' : '🔊';
    if (state.masterGain) {
      state.masterGain.gain.setValueAtTime(state.isMuted ? 0 : state.volume, state.audioCtx.currentTime);
    }
  });

  elements.volumeSlider.addEventListener('input', (e) => {
    state.volume = parseFloat(e.target.value);
    if (!state.isMuted && state.masterGain) {
      state.masterGain.gain.setValueAtTime(state.volume, state.audioCtx.currentTime);
    }
  });

  // Synth & Reverb Toggles
  elements.synthToggleBtn.addEventListener('click', () => {
    state.synthEnabled = !state.synthEnabled;
    elements.synthToggleBtn.textContent = `Synth: ${state.synthEnabled ? 'ON' : 'OFF'}`;
    elements.synthToggleBtn.classList.toggle('active', state.synthEnabled);
  });

  elements.reverbToggleBtn.addEventListener('click', () => {
    state.reverbEnabled = !state.reverbEnabled;
    elements.reverbToggleBtn.textContent = `Reverb: ${state.reverbEnabled ? 'ON' : 'OFF'}`;
    elements.reverbToggleBtn.classList.toggle('active', state.reverbEnabled);
  });
}

// --- API Actions ---
async function startSession() {
  try {
    if (state.audioCtx && state.audioCtx.state === 'suspended') {
      await state.audioCtx.resume();
    }
    const res = await fetch('/api/sessions/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset: state.preset, target_goal: state.targetGoal, interval: 2.0 }),
    });
    const data = await res.json();
    state.sessionId = data.session.session_id;
    state.isRunning = true;
    state.sessionStartTime = Date.now();
    updateSessionUI(true);
    showToast(`Session ${state.sessionId.slice(0, 8)} started!`);
    startTimer();
    if (data.initial_frame) handleTelemetry(data.initial_frame);
  } catch (err) {
    showToast('Failed to start session: ' + err);
  }
}

async function stopSession() {
  try {
    await fetch('/api/sessions/stop', { method: 'POST' });
    state.isRunning = false;
    updateSessionUI(false);
    stopTimer();
    showToast('Session stopped and saved to storage.');
  } catch (err) {
    showToast('Failed to stop session: ' + err);
  }
}

async function switchPreset(newPreset) {
  state.preset = newPreset;
  [elements.btnPresetCalm, elements.btnPresetNeutral, elements.btnPresetAroused].forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.preset === newPreset);
  });
  await fetch('/api/sessions/preset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset: newPreset }),
  });
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify({ action: 'set_preset', preset: newPreset }));
  }
  showToast(`Switched EEG preset to: ${newPreset.toUpperCase()}`);
}

async function checkCurrentSession() {
  try {
    const res = await fetch('/api/sessions/current');
    const data = await res.json();
    if (data.is_running) {
      state.isRunning = true;
      state.sessionId = data.session_id;
      state.preset = data.preset;
      updateSessionUI(true);
      startTimer();
      if (data.latest_frame) handleTelemetry(data.latest_frame);
    }
  } catch (e) {
    console.log('No current active session');
  }
}

function updateSessionUI(running) {
  elements.sessionStateDot.className = `status-dot ${running ? 'dot-active' : 'dot-inactive'}`;
  elements.sessionStateText.textContent = running ? 'Running' : 'Idle';
  elements.sessionBtnLabel.textContent = running ? 'Stop Session' : 'Start Session';
  elements.sessionToggleBtn.classList.toggle('btn-stop', running);
}

function startTimer() {
  stopTimer();
  state.timerInterval = setInterval(() => {
    if (!state.sessionStartTime) return;
    const elapsed = Math.floor((Date.now() - state.sessionStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    elements.sessionTimer.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopTimer() {
  if (state.timerInterval) {
    clearInterval(state.timerInterval);
    state.timerInterval = null;
  }
}
