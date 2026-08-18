const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const dampFactor = (speed, dt) => 1 - Math.exp(-speed * dt);
const angleDelta = (from, to) => Math.atan2(Math.sin(to - from), Math.cos(to - from));

const CAR_SVG = `
  <svg viewBox="0 0 100 180" aria-hidden="true">
    <defs>
      <linearGradient id="scarlet" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0" stop-color="#ff5b55"/><stop offset=".48" stop-color="#e62938"/><stop offset="1" stop-color="#a80e25"/>
      </linearGradient>
      <linearGradient id="carbon" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#313744"/><stop offset="1" stop-color="#10131a"/>
      </linearGradient>
      <filter id="car-shadow" x="-40%" y="-30%" width="180%" height="190%">
        <feDropShadow dx="0" dy="6" stdDeviation="5" flood-color="#141824" flood-opacity=".3"/>
      </filter>
    </defs>
    <g filter="url(#car-shadow)">
      <path class="car-trail" d="M38 171 32 180M62 171l6 9"/>
      <g class="wheel wheel-front-left"><rect x="5" y="30" width="18" height="35" rx="8"/><path d="M10 35v25M18 35v25"/></g>
      <g class="wheel wheel-front-right"><rect x="77" y="30" width="18" height="35" rx="8"/><path d="M82 35v25M90 35v25"/></g>
      <g class="wheel wheel-rear-left"><rect x="1" y="112" width="23" height="43" rx="9"/><path d="M7 118v31M17 118v31"/></g>
      <g class="wheel wheel-rear-right"><rect x="76" y="112" width="23" height="43" rx="9"/><path d="M82 118v31M92 118v31"/></g>

      <g class="suspension" fill="none" stroke="#171b25" stroke-width="3">
        <path d="M20 38 43 58M20 58l23-8M80 38 57 58M80 58 57 50M22 121l22 9M22 145l22-12M78 121l-22 9M78 145l-22-12"/>
      </g>

      <path class="front-wing" d="M10 13h28l7 5h10l7-5h28l-3 11-27 3-6 7h-8l-6-7-27-3z"/>
      <path class="front-wing-accent" d="M13 17h26l4 4H15zm74 0H61l-4 4h28z"/>
      <path class="nose" d="M44 19h12l5 43-7 25h-8l-7-25z"/>
      <path class="body" d="M39 55c-4 10-6 26-5 39l-10 25 11 30h30l11-30-10-25c1-13-1-29-5-39l-11 8z"/>
      <path class="sidepod sidepod-left" d="M35 78 21 86l5 36 16 7 3-39z"/>
      <path class="sidepod sidepod-right" d="m65 78 14 8-5 36-16 7-3-39z"/>
      <path class="sf-white" d="M41 62c2 7 1 17-2 29l5 20h12l5-20c-3-12-4-22-2-29l-9 4z"/>
      <path class="sf-white sf-side-left" d="M27 88 38 81l2 15-11 10z"/>
      <path class="sf-white sf-side-right" d="m73 88-11-7-2 15 11 10z"/>
      <path class="floor" d="M28 101h12l3 43H27l-8-24zm44 0H60l-3 43h16l8-24z"/>
      <path class="engine-cover" d="M41 105h18l7 42-16 14-16-14z"/>
      <path class="engine-stripe" d="M47 109h6v44l-3 3-3-3z"/>
      <path class="rear-wing" d="M13 153h74l5 15H8z"/>
      <path class="rear-wing-accent" d="M17 158h66l2 5H15z"/>
      <path class="rear-wing-red" d="M40 156h20l2 9H38z"/>

      <ellipse class="cockpit" cx="50" cy="90" rx="12" ry="20"/>
      <path class="helmet" d="M43 87c0-8 3-13 7-13s7 5 7 13l-3 8h-8z"/>
      <path class="visor" d="M44 84c2-4 10-4 12 0l-2 4h-8z"/>
      <path class="halo" d="M39 88c0-14 4-22 11-22s11 8 11 22M50 66v27"/>

      <circle class="number-disc" cx="50" cy="47" r="8"/>
      <text class="car-number" x="50" y="51" text-anchor="middle">26</text>
      <path class="livery-light" d="M42 58 35 74l5 4 10-8 10 8 5-4-7-16-8 5z"/>
      <path class="livery-gold" d="m28 91 12-5-1 8-10 5zm44 0-12-5 1 8 10 5z"/>
      <circle class="livery-blue" cx="31" cy="110" r="5"/><circle class="livery-blue" cx="69" cy="110" r="5"/>
      <circle class="wheel-lock" cx="14" cy="47" r="3"/><circle class="wheel-lock" cx="86" cy="47" r="3"/>
      <circle class="wheel-lock" cx="13" cy="133" r="3"/><circle class="wheel-lock" cx="87" cy="133" r="3"/>
    </g>
  </svg>`;

export function mountSiteCar(container, { onStateChange = () => {} } = {}) {
  const effects = document.createElement("canvas");
  effects.className = "drive-effects";
  effects.setAttribute("aria-hidden", "true");
  const car = document.createElement("button");
  car.type = "button";
  car.className = "drive-car";
  car.setAttribute("aria-label", "Scarlet Formula racing car. Use W A S D or the arrow keys to drive around the website. Press Escape to park.");
  car.innerHTML = CAR_SVG;
  container.append(effects, car);

  const effectContext = effects.getContext("2d", { alpha: true });
  const particles = [];
  let previousTrack = null;

  const keys = new Set();
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  function carDimensions() { return innerWidth < 520 ? { width: 72, height: 130 } : { width: 92, height: 166 }; }
  function safeBounds() {
    const { width, height } = carDimensions();
    const horizontal = innerWidth < 520 ? 46 : 68;
    const vertical = innerWidth < 520 ? 58 : 72;
    return { left: horizontal, right: Math.max(horizontal, innerWidth - width - horizontal), top: vertical, bottom: Math.max(vertical, innerHeight - height - vertical) };
  }
  const initialBounds = safeBounds();
  const state = {
    x: initialBounds.right,
    y: clamp(innerHeight * (innerWidth < 520 ? .65 : .57), initialBounds.top, initialBounds.bottom),
    angle: 0,
    speed: 0,
    steering: 0,
    throttle: 0,
    steerInput: 0,
    driving: true,
    dragging: null,
    lastTime: performance.now()
  };

  function emitState() {
    document.body.classList.toggle("car-driving", state.driving);
    onStateChange({ driving: state.driving });
  }

  function setDriving(driving) {
    state.driving = driving;
    state.speed = driving ? state.speed : 0;
    state.steering = 0;
    keys.clear();
    car.classList.toggle("is-parked", !driving);
    emitState();
  }

  function reset() {
    const bounds = safeBounds();
    state.x = bounds.right;
    state.y = clamp(innerHeight * (innerWidth < 520 ? .65 : .57), bounds.top, bounds.bottom);
    state.angle = 0;
    state.speed = 0;
    state.steering = 0;
    state.throttle = 0;
    state.steerInput = 0;
    previousTrack = null;
    setDriving(true);
    car.focus({ preventScroll: true });
  }

  function editableTarget(target) {
    return target instanceof Element && Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
  }

  const driveKeys = new Set(["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d"]);
  addEventListener("keydown", event => {
    const key = event.key.toLowerCase();
    if (key === "escape" && state.driving) {
      event.preventDefault();
      setDriving(false);
      return;
    }
    if (!state.driving || !driveKeys.has(key) || editableTarget(event.target)) return;
    const focusedControl = document.activeElement?.closest?.("button, a, [role='tab']");
    if (focusedControl && focusedControl !== car) return;
    event.preventDefault();
    keys.add(key);
    if (!event.repeat) {
      if (key === "arrowup" || key === "w") state.speed = clamp(state.speed + 72, -360, 760);
      if (key === "arrowdown" || key === "s") state.speed = clamp(state.speed - 58, -360, 760);
      if (key === "arrowleft" || key === "a") state.angle -= .045;
      if (key === "arrowright" || key === "d") state.angle += .045;
    }
    car.classList.add("is-moving");
  });
  addEventListener("keyup", event => {
    keys.delete(event.key.toLowerCase());
    if (![...keys].some(key => driveKeys.has(key))) car.classList.remove("is-moving");
  });
  addEventListener("blur", () => { keys.clear(); car.classList.remove("is-moving"); });

  car.addEventListener("click", () => {
    if (!state.dragging && !state.driving) setDriving(true);
  });
  car.addEventListener("pointerdown", event => {
    event.preventDefault();
    car.setPointerCapture(event.pointerId);
    car.focus({ preventScroll: true });
    state.dragging = { id: event.pointerId, startX: event.clientX, startY: event.clientY, carX: state.x, carY: state.y, moved: false };
    state.speed = 0;
    car.classList.add("is-grabbed");
  });
  car.addEventListener("pointermove", event => {
    const drag = state.dragging;
    if (!drag || drag.id !== event.pointerId) return;
    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    drag.moved ||= Math.hypot(dx, dy) > 5;
    state.x = clamp(drag.carX + dx, 18, innerWidth - 104);
    state.y = clamp(drag.carY + dy, 18, innerHeight - 166);
    if (Math.hypot(dx, dy) > 4) state.angle = Math.atan2(dx, -dy);
  });
  function releasePointer(event) {
    if (!state.dragging || state.dragging.id !== event.pointerId) return;
    const wasMoved = state.dragging.moved;
    if (car.hasPointerCapture(event.pointerId)) car.releasePointerCapture(event.pointerId);
    state.dragging = wasMoved ? { moved: true } : null;
    car.classList.remove("is-grabbed");
    requestAnimationFrame(() => { state.dragging = null; });
  }
  car.addEventListener("pointerup", releasePointer);
  car.addEventListener("pointercancel", releasePointer);

  function edgeScroll(dt, verticalVelocity) {
    const verticalMargin = Math.min(150, innerHeight * .22);
    let scrollVelocity = 0;
    if (state.y < verticalMargin && scrollY > 0 && verticalVelocity < -20) {
      scrollVelocity = -clamp(Math.min((verticalMargin - state.y) * 5.2, Math.abs(verticalVelocity)), 0, 620);
      state.y = Math.max(state.y, 44);
    } else if (state.y > innerHeight - verticalMargin - 150 && scrollY + innerHeight < document.documentElement.scrollHeight - 1 && verticalVelocity > 20) {
      scrollVelocity = clamp(Math.min((state.y - (innerHeight - verticalMargin - 150)) * 5.2, Math.abs(verticalVelocity)), 0, 620);
      state.y = Math.min(state.y, innerHeight - 72);
    }
    if (scrollVelocity) window.scrollTo(window.scrollX, window.scrollY + scrollVelocity * dt);
  }

  function frame(now) {
    const dt = Math.min((now - state.lastTime) / 1000, .035);
    state.lastTime = now;
    if (state.driving && !state.dragging) {
      const throttle = keys.has("arrowup") || keys.has("w");
      const reverse = keys.has("arrowdown") || keys.has("s");
      const left = keys.has("arrowleft") || keys.has("a");
      const right = keys.has("arrowright") || keys.has("d");
      const input = (throttle ? 1 : 0) - (reverse ? 1 : 0);
      const turn = (right ? 1 : 0) - (left ? 1 : 0);

      state.speed += input * (input < 0 ? 920 : 1350) * dt;
      state.speed *= Math.exp(-(input ? 1.15 : 3.4) * dt);
      state.speed = clamp(state.speed, -360, 760);
      const steeringTarget = turn * (2.5 + Math.min(Math.abs(state.speed) / 260, 1.7));
      state.steering += (steeringTarget - state.steering) * Math.min(1, dt * 12);
      if (Math.abs(state.speed) > 5 || turn) state.angle += state.steering * dt * (state.speed < -5 ? -1 : 1);

      const horizontalVelocity = Math.sin(state.angle) * state.speed;
      const verticalVelocity = -Math.cos(state.angle) * state.speed;
      state.x += horizontalVelocity * dt;
      state.y += verticalVelocity * dt;
      edgeScroll(dt, verticalVelocity);

      const sidePadding = innerWidth < 520 ? 10 : 24;
      const maxX = innerWidth - (innerWidth < 520 ? 42 : 54);
      if (state.x < sidePadding) { state.x = sidePadding; state.speed *= .6; }
      if (state.x > maxX) { state.x = maxX; state.speed *= .6; }
      state.y = clamp(state.y, 10, innerHeight - 164);
      car.classList.toggle("is-moving", Math.abs(state.speed) > 28);
      car.style.setProperty("--car-speed", Math.min(Math.abs(state.speed) / 760, 1).toFixed(3));
    }

    car.style.transform = `translate3d(${state.x.toFixed(2)}px, ${state.y.toFixed(2)}px, 0) rotate(${state.angle.toFixed(4)}rad)`;
    requestAnimationFrame(frame);
  }

  addEventListener("resize", () => {
    state.x = clamp(state.x, 12, innerWidth - (innerWidth < 520 ? 42 : 54));
    state.y = clamp(state.y, 12, innerHeight - 168);
  });
  reducedMotion.addEventListener("change", () => car.classList.toggle("reduce-motion", reducedMotion.matches));
  car.classList.toggle("reduce-motion", reducedMotion.matches);
  emitState();
  requestAnimationFrame(frame);

  return { reset, toggle: () => setDriving(!state.driving), setDriving };
}
