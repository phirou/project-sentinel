/**
 * Sentinel Dashboard — client JavaScript
 *
 * Connecte le navigateur au backend via WebSocket, reçoit les RadarFrame
 * en JSON, et anime un affichage radar circulaire sur Canvas.
 *
 * Architecture :
 *   - WebSocket client → état partagé (latestFrame, stats)
 *   - requestAnimationFrame loop → rendu Canvas + mise à jour DOM
 *
 * Le rendu est découplé de la réception : même si on reçoit 10 trames/s,
 * on rend à 60 FPS pour avoir un sweep fluide. Quand une nouvelle trame
 * arrive, on l'affecte à latestFrame et le prochain frame de rendu
 * l'utilisera.
 */

// ════════════════════════════════════════════════════════════════════
// CONSTANTES DE L'AFFICHAGE RADAR
// ════════════════════════════════════════════════════════════════════

// Champ de vue du radar (doit correspondre au LD2450 : ±60°).
const FOV_DEGREES = 120;

// Portée maximale affichée, en mètres. Au-delà, les cibles sortent
// du cercle. 8m correspond à la portée pratique du LD2450.
const MAX_RANGE_METERS = 8;

// Cercles concentriques de l'échelle de distance.
const RANGE_RINGS_METERS = [2, 4, 6, 8];

// Vitesse de rotation du sweep en degrés par seconde.
const SWEEP_SPEED_DEG_PER_SEC = 60;

// Durée de la traînée du sweep en degrés (effet phosphorescent).
const SWEEP_TRAIL_DEGREES = 80;

// Durée pendant laquelle une cible reste affichée même si elle disparaît
// d'une trame, pour éviter le clignotement (en millisecondes).
const TARGET_PERSISTENCE_MS = 500;

// Couleurs (synchronisées avec les variables CSS via getComputedStyle).
let COLORS = {
    primary: '#00ff41',
    dim: '#00cc33',
    faint: '#006619',
    warning: '#ffcc00',
    danger: '#ff3333',
    bg: '#0a0e0d',
};


// ════════════════════════════════════════════════════════════════════
// ÉTAT GLOBAL DE L'APPLICATION
// ════════════════════════════════════════════════════════════════════

const state = {
    // Connexion WebSocket
    ws: null,
    connected: false,
    reconnectDelay: 1000,

    // Dernière trame reçue
    latestFrame: null,
    targetsHistory: new Map(),  // slot → { target, lastSeen }

    // Statistiques cumulées
    framesReceived: 0,
    framesWithTargets: 0,
    parseErrors: 0,
    totalTargets: 0,
    startTime: Date.now(),

    // Cadence de réception (frames par seconde)
    frameTimestamps: [],

    // Animation du sweep
    sweepAngleDeg: 0,
    lastFrameTimestamp: performance.now(),
};


// ════════════════════════════════════════════════════════════════════
// CONNEXION WEBSOCKET
// ════════════════════════════════════════════════════════════════════

function connectWebSocket() {
    const url = `ws://${location.host}/ws/realtime`;
    console.log(`[WS] Connecting to ${url}...`);
    state.ws = new WebSocket(url);

    state.ws.onopen = () => {
        console.log('[WS] Connected');
        state.connected = true;
        state.reconnectDelay = 1000;
        updateConnectionStatus(true);
    };

    state.ws.onclose = () => {
        console.log('[WS] Disconnected');
        state.connected = false;
        updateConnectionStatus(false);

        // Reconnexion exponentielle avec backoff (max 10s).
        const delay = Math.min(state.reconnectDelay, 10000);
        setTimeout(connectWebSocket, delay);
        state.reconnectDelay *= 2;
    };

    state.ws.onerror = (err) => {
        console.error('[WS] Error:', err);
    };

    state.ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'radar.frame') {
                handleFrame(msg.data);
            }
        } catch (err) {
            console.error('[WS] Failed to parse message:', err);
        }
    };
}


function handleFrame(frame) {
    state.latestFrame = frame;
    state.framesReceived++;
    state.totalTargets += frame.target_count;
    if (frame.has_targets) {
        state.framesWithTargets++;
    }

    // Mise à jour de la cadence (rolling window de 2 secondes).
    const now = performance.now();
    state.frameTimestamps.push(now);
    state.frameTimestamps = state.frameTimestamps.filter(t => now - t < 2000);

    // Mise à jour de l'historique des cibles avec persistance.
    for (const target of frame.targets) {
        state.targetsHistory.set(target.slot, {
            target: target,
            lastSeen: now,
        });
    }

    // Nettoyage des cibles trop anciennes.
    for (const [slot, entry] of state.targetsHistory) {
        if (now - entry.lastSeen > TARGET_PERSISTENCE_MS) {
            state.targetsHistory.delete(slot);
        }
    }
}


// ════════════════════════════════════════════════════════════════════
// MISE À JOUR DU DOM
// ════════════════════════════════════════════════════════════════════

function updateConnectionStatus(connected) {
    const dot = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');
    if (connected) {
        dot.classList.add('live');
        text.textContent = 'LIVE';
        text.classList.remove('disconnected');
    } else {
        dot.classList.remove('live');
        text.textContent = 'DISCONNECTED';
        text.classList.add('disconnected');
    }
}


function updateTargetsList() {
    const container = document.getElementById('targets-list');
    const targets = state.latestFrame ? state.latestFrame.targets : [];

    document.getElementById('target-count').textContent = targets.length;

    if (targets.length === 0) {
        container.innerHTML = '<div class="target-empty">— No targets in field of view —</div>';
        return;
    }

    container.innerHTML = targets.map(t => {
        const stateClass = t.state.toLowerCase();
        const arrow = t.state === 'approaching' ? '▼' :
                     t.state === 'receding' ? '▲' : '■';
        return `
            <div class="target-card ${stateClass}">
                <div class="target-id">T${t.slot}</div>
                <div class="target-info">
                    <div class="target-info-row">
                        <span class="label">DIST</span>
                        <span>${t.distance_m.toFixed(2)} m</span>
                    </div>
                    <div class="target-info-row">
                        <span class="label">AZIM</span>
                        <span>${t.angle_degrees >= 0 ? '+' : ''}${t.angle_degrees.toFixed(1)}°</span>
                    </div>
                    <div class="target-info-row">
                        <span class="label">VEL</span>
                        <span>${arrow} ${Math.abs(t.speed_ms).toFixed(2)} m/s</span>
                    </div>
                </div>
                <div class="target-state ${stateClass}">${arrow}</div>
            </div>
        `;
    }).join('');
}


function updateTelemetry() {
    document.getElementById('tlm-frames').textContent = state.framesReceived;
    document.getElementById('tlm-with-targets').textContent = state.framesWithTargets;
    document.getElementById('tlm-errors').textContent = state.parseErrors;

    const avg = state.framesReceived > 0
        ? (state.totalTargets / state.framesReceived).toFixed(2)
        : '0.00';
    document.getElementById('tlm-avg').textContent = avg;

    // Uptime
    const uptimeSec = Math.floor((Date.now() - state.startTime) / 1000);
    const min = Math.floor(uptimeSec / 60);
    const sec = uptimeSec % 60;
    document.getElementById('tlm-uptime').textContent =
        `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;

    // Cadence (Hz)
    const rate = state.frameTimestamps.length / 2.0;
    document.getElementById('status-rate').textContent = `${rate.toFixed(1)} Hz`;

    // Horloge
    const now = new Date();
    document.getElementById('status-clock').textContent =
        now.toLocaleTimeString('en-GB');
}


// ════════════════════════════════════════════════════════════════════
// RENDU DU RADAR (CANVAS)
// ════════════════════════════════════════════════════════════════════

const canvas = document.getElementById('radar-canvas');
const ctx = canvas.getContext('2d');

// Géométrie courante du radar (recalculée au resize).
let radarGeom = { cx: 0, cy: 0, radius: 0 };


function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Centre du radar : en bas au milieu (le radar regarde vers le haut).
    // On garde un peu de marge en haut et en bas.
    radarGeom.cx = rect.width / 2;
    radarGeom.cy = rect.height * 0.92;
    // Rayon = distance du centre au bord supérieur, avec marge.
    radarGeom.radius = Math.min(
        rect.width / 2 * 0.95,
        rect.height * 0.88
    );
}


/**
 * Convertit (distance en mètres, angle en degrés) en coordonnées canvas (x, y).
 * Convention : angle = 0 droit devant (vers le haut du canvas),
 * angle positif = à droite, négatif = à gauche.
 */
function polarToCanvas(distance_m, angle_deg) {
    const r = (distance_m / MAX_RANGE_METERS) * radarGeom.radius;
    // -90° pour que 0° pointe vers le haut au lieu de la droite.
    const angleRad = (angle_deg - 90) * Math.PI / 180;
    return {
        x: radarGeom.cx + r * Math.cos(angleRad),
        y: radarGeom.cy + r * Math.sin(angleRad),
    };
}


function drawGrid() {
    const { cx, cy, radius } = radarGeom;

    // Cercles de distance
    ctx.strokeStyle = COLORS.faint;
    ctx.lineWidth = 1;
    ctx.font = '10px monospace';
    ctx.fillStyle = COLORS.dim;

    for (const range of RANGE_RINGS_METERS) {
        const r = (range / MAX_RANGE_METERS) * radius;
        ctx.beginPath();
        // Demi-cercle (uniquement la partie supérieure, là où le radar voit).
        ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
        ctx.stroke();

        // Étiquette de distance
        ctx.fillText(`${range}m`, cx + 4, cy - r + 12);
    }

    // Lignes radiales (graduations angulaires tous les 30°).
    ctx.strokeStyle = COLORS.faint;
    for (let angle = -60; angle <= 60; angle += 30) {
        const end = polarToCanvas(MAX_RANGE_METERS, angle);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();

        // Étiquette d'angle
        const label = polarToCanvas(MAX_RANGE_METERS * 1.05, angle);
        ctx.fillStyle = COLORS.dim;
        ctx.textAlign = 'center';
        ctx.fillText(
            `${angle >= 0 ? '+' : ''}${angle}°`,
            label.x,
            label.y
        );
    }
    ctx.textAlign = 'start';

    // Bord du FoV (cône) en plus marqué.
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1.5;
    const leftEdge = polarToCanvas(MAX_RANGE_METERS, -60);
    const rightEdge = polarToCanvas(MAX_RANGE_METERS, 60);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(leftEdge.x, leftEdge.y);
    ctx.moveTo(cx, cy);
    ctx.lineTo(rightEdge.x, rightEdge.y);
    ctx.stroke();

    // Arc extérieur du FoV
    ctx.beginPath();
    const startAngle = (-60 - 90) * Math.PI / 180;
    const endAngle = (60 - 90) * Math.PI / 180;
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.stroke();

    // Symbole du radar au centre
    ctx.fillStyle = COLORS.primary;
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
    ctx.fill();
}


function drawSweep() {
    const { cx, cy, radius } = radarGeom;

    // Sweep limité au FoV : on n'anime que dans la zone -60° à +60°.
    // L'angle oscille entre ces bornes plutôt que de tourner à 360°.
    const sweepDuration = (FOV_DEGREES * 2) / SWEEP_SPEED_DEG_PER_SEC;
    const t = (performance.now() / 1000) % sweepDuration;
    const halfDuration = sweepDuration / 2;

    let sweepAngle;
    if (t < halfDuration) {
        // Phase aller : -60° → +60°
        sweepAngle = -60 + (t / halfDuration) * 120;
    } else {
        // Phase retour : +60° → -60°
        sweepAngle = 60 - ((t - halfDuration) / halfDuration) * 120;
    }
    state.sweepAngleDeg = sweepAngle;

    // Trail dégradé (effet phosphorescent).
    for (let offset = 0; offset < SWEEP_TRAIL_DEGREES; offset += 2) {
        const angle = sweepAngle - offset;
        if (angle < -60 || angle > 60) continue;

        const alpha = 0.4 * (1 - offset / SWEEP_TRAIL_DEGREES);
        ctx.strokeStyle = `rgba(0, 255, 65, ${alpha})`;
        ctx.lineWidth = 2;
        const end = polarToCanvas(MAX_RANGE_METERS, angle);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
    }

    // Ligne du sweep elle-même, plus brillante.
    ctx.strokeStyle = COLORS.primary;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 8;
    ctx.shadowColor = COLORS.primary;
    const end = polarToCanvas(MAX_RANGE_METERS, sweepAngle);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.shadowBlur = 0;
}


function drawTargets() {
    const now = performance.now();

    for (const [slot, entry] of state.targetsHistory) {
        const { target, lastSeen } = entry;
        const age = now - lastSeen;
        const fade = 1 - (age / TARGET_PERSISTENCE_MS);

        const pos = polarToCanvas(target.distance_m, target.angle_degrees);

        // Halo extérieur (effet glow)
        const haloAlpha = 0.3 * fade;
        ctx.fillStyle = target.state === 'approaching'
            ? `rgba(255, 204, 0, ${haloAlpha})`
            : `rgba(0, 255, 65, ${haloAlpha})`;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 14, 0, 2 * Math.PI);
        ctx.fill();

        // Point central
        ctx.fillStyle = target.state === 'approaching'
            ? COLORS.warning
            : COLORS.primary;
        ctx.shadowBlur = 6;
        ctx.shadowColor = ctx.fillStyle;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Étiquette : "T0 3.2m"
        ctx.fillStyle = COLORS.primary;
        ctx.font = '11px monospace';
        ctx.fillText(
            `T${target.slot} ${target.distance_m.toFixed(1)}m`,
            pos.x + 10,
            pos.y - 8
        );
    }
}


function renderRadar() {
    const rect = canvas.getBoundingClientRect();

    // Fond noir avec léger fade pour effet rémanent (trails).
    ctx.fillStyle = 'rgba(15, 22, 20, 0.15)';
    ctx.fillRect(0, 0, rect.width, rect.height);

    drawGrid();
    drawSweep();
    drawTargets();
}


// ════════════════════════════════════════════════════════════════════
// BOUCLE D'ANIMATION PRINCIPALE
// ════════════════════════════════════════════════════════════════════

function animationLoop() {
    renderRadar();
    updateTargetsList();
    updateTelemetry();
    requestAnimationFrame(animationLoop);
}


// ════════════════════════════════════════════════════════════════════
// AMORÇAGE
// ════════════════════════════════════════════════════════════════════

function init() {
    // Synchronise les couleurs JS avec les variables CSS pour cohérence.
    const styles = getComputedStyle(document.documentElement);
    COLORS.primary = styles.getPropertyValue('--color-primary').trim() || COLORS.primary;
    COLORS.dim     = styles.getPropertyValue('--color-dim').trim()     || COLORS.dim;
    COLORS.faint   = styles.getPropertyValue('--color-faint').trim()   || COLORS.faint;
    COLORS.warning = styles.getPropertyValue('--color-warning').trim() || COLORS.warning;
    COLORS.danger  = styles.getPropertyValue('--color-danger').trim()  || COLORS.danger;
    COLORS.bg      = styles.getPropertyValue('--bg-primary').trim()    || COLORS.bg;

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    connectWebSocket();
    requestAnimationFrame(animationLoop);
}


// On attend que le DOM soit prêt avant d'initialiser.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}