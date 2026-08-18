/**
 * Procedural F1 car, built from primitives in three.js.
 *
 * Built from geometry rather than drawn, because what makes a car read as a car
 * is lighting and proportion, not linework. No external model file, so nothing
 * to license, host or lose.
 *
 * Degrades to nothing if WebGL is unavailable, and stops rendering entirely when
 * scrolled out of view or the tab is hidden — it is decoration and must never
 * cost a phone its battery.
 */
import * as THREE from "./vendor/three.module.min.js";

const RED = 0xd8102a;
const CARBON = 0x0b0d10;
const CHROME = 0xc8ccd4;

export function mountCar(container) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    return null; // no WebGL: the page is fine without this
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);

  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";

  // ---- materials -----------------------------------------------------------
  const bodyMat = new THREE.MeshStandardMaterial({
    color: RED, metalness: 0.25, roughness: 0.28,
  });
  const carbonMat = new THREE.MeshStandardMaterial({
    color: CARBON, metalness: 0.6, roughness: 0.45,
  });
  const rubberMat = new THREE.MeshStandardMaterial({
    color: 0x121418, metalness: 0.1, roughness: 0.9,
  });
  const rimMat = new THREE.MeshStandardMaterial({
    color: CHROME, metalness: 0.95, roughness: 0.22,
  });
  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xe8ebf0, metalness: 0.3, roughness: 0.4,
  });

  const car = new THREE.Group();
  const box = (w, h, d, mat, x, y, z) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    m.castShadow = true;
    car.add(m);
    return m;
  };

  // ---- body ----------------------------------------------------------------
  // Floor / plank: the long flat reference surface everything sits on.
  box(5.3, 0.08, 1.5, carbonMat, 0.1, 0.20, 0);

  // Monocoque, tapering forward via three stacked segments.
  box(1.5, 0.30, 0.86, bodyMat, -0.55, 0.40, 0);
  box(1.3, 0.24, 0.58, bodyMat, -1.42, 0.38, 0);
  box(1.0, 0.18, 0.34, bodyMat, -2.10, 0.36, 0);

  // Nose cone, thin and low.
  const nose = box(1.05, 0.14, 0.20, bodyMat, -2.80, 0.33, 0);
  nose.rotation.z = -0.05;

  // Sidepods: bulky at the front, tapering into the coke-bottle.
  [-1, 1].forEach((side) => {
    box(1.55, 0.34, 0.40, bodyMat, 0.30, 0.41, side * 0.52);
    box(0.95, 0.22, 0.24, bodyMat, 1.25, 0.38, side * 0.36);
    // inlet
    box(0.10, 0.30, 0.34, carbonMat, -0.50, 0.47, side * 0.53);
  });

  // Engine cover + airbox above the driver.
  box(1.9, 0.28, 0.36, bodyMat, 0.85, 0.63, 0);
  box(0.46, 0.26, 0.28, carbonMat, -0.04, 0.74, 0);

  // Cockpit opening + helmet.
  box(0.62, 0.10, 0.52, carbonMat, -0.55, 0.66, 0);
  const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.17, 24, 18), carbonMat);
  helmet.position.set(-0.55, 0.76, 0);
  helmet.castShadow = true;
  car.add(helmet);

  // Halo: a torus arc over the cockpit, plus its forward strut.
  const halo = new THREE.Mesh(
    new THREE.TorusGeometry(0.42, 0.045, 12, 28, Math.PI),
    carbonMat
  );
  halo.position.set(-0.55, 0.80, 0);
  halo.rotation.set(Math.PI / 2, 0, 0);
  halo.castShadow = true;
  car.add(halo);
  box(0.06, 0.22, 0.06, carbonMat, -0.97, 0.75, 0);

  // ---- wings ---------------------------------------------------------------
  // Front wing: main plane, upper flap, endplates.
  box(0.70, 0.030, 2.00, wingMat, -3.15, 0.13, 0);
  box(0.46, 0.028, 1.92, bodyMat, -3.00, 0.23, 0);
  [-1, 1].forEach((s) => box(0.78, 0.30, 0.04, bodyMat, -3.10, 0.24, s * 1.0));

  // Rear wing: raised main plane on twin pillars, with endplates.
  box(0.62, 0.035, 1.02, wingMat, 2.58, 0.92, 0);
  box(0.42, 0.030, 1.00, bodyMat, 2.74, 0.79, 0);
  [-1, 1].forEach((s) => box(0.74, 0.36, 0.035, bodyMat, 2.63, 0.79, s * 0.50));
  box(0.08, 0.52, 0.08, carbonMat, 2.48, 0.58, 0);
  // Beam wing + diffuser hint
  box(0.30, 0.05, 0.90, carbonMat, 2.62, 0.42, 0);
  box(0.45, 0.26, 1.20, carbonMat, 2.30, 0.28, 0);

  // ---- wheels --------------------------------------------------------------
  const wheels = [];
  const makeWheel = (x, z, radius, width) => {
    const g = new THREE.Group();
    const tyre = new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, width, 28),
      rubberMat
    );
    tyre.rotation.x = Math.PI / 2;
    tyre.castShadow = true;
    g.add(tyre);

    const rim = new THREE.Mesh(
      new THREE.CylinderGeometry(radius * 0.56, radius * 0.56, width * 1.02, 20),
      rimMat
    );
    rim.rotation.x = Math.PI / 2;
    g.add(rim);

    // Spokes, so rotation is legible rather than a featureless disc.
    for (let i = 0; i < 5; i++) {
      const spoke = new THREE.Mesh(
        new THREE.BoxGeometry(radius * 1.05, 0.035, width * 0.34),
        rimMat
      );
      spoke.rotation.z = (i * Math.PI) / 5;
      g.add(spoke);
    }
    // Coloured sidewall band
    const band = new THREE.Mesh(
      new THREE.TorusGeometry(radius * 0.82, 0.022, 8, 30),
      new THREE.MeshStandardMaterial({ color: RED, metalness: 0.3, roughness: 0.5 })
    );
    band.position.z = width / 2;
    g.add(band);

    g.position.set(x, radius, z);
    car.add(g);
    wheels.push(g);
    return g;
  };
  [-1, 1].forEach((s) => {
    makeWheel(-1.85, s * 0.86, 0.36, 0.34); // front
    makeWheel(1.75, s * 0.90, 0.40, 0.46); // rear
  });

  car.position.y = -0.25;
  scene.add(car);

  // ---- lighting: key, rim, fill --------------------------------------------
  scene.add(new THREE.HemisphereLight(0x8ea6d8, 0x08090c, 0.85));

  // Key light does the describing: strong, white, high and to the front so the
  // bodywork's curvature is legible rather than flat.
  const key = new THREE.DirectionalLight(0xffffff, 4.2);
  key.position.set(5, 7, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 24;
  key.shadow.camera.left = -6;
  key.shadow.camera.right = 6;
  key.shadow.camera.top = 6;
  key.shadow.camera.bottom = -6;
  scene.add(key);

  // Rim light picks out the silhouette. Kept low: at high intensity it stops
  // being an edge and becomes a red wash over the whole car.
  const rim = new THREE.DirectionalLight(0xff4059, 1.1);
  rim.position.set(-7, 3, -6);
  scene.add(rim);

  // Cool fill lifts the shadow side so the underbody is not a black hole.
  const fill = new THREE.DirectionalLight(0x7d9bff, 1.0);
  fill.position.set(-4, 2, 7);
  scene.add(fill);

  // Small warm kicker along the flank, for a highlight to travel down.
  const kicker = new THREE.DirectionalLight(0xffd9a8, 0.7);
  kicker.position.set(2, 1.2, 8);
  scene.add(kicker);

  // Ground: catches the shadow so the car sits in the scene rather than floating.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.ShadowMaterial({ opacity: 0.55 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.25;
  ground.receiveShadow = true;
  scene.add(ground);

  // ---- interaction ---------------------------------------------------------
  let pointer = 0, targetPointer = 0, steer = 0, targetSteer = 0;
  const reduced = matchMedia("(prefers-reduced-motion:reduce)").matches;

  container.addEventListener("pointermove", (e) => {
    const r = container.getBoundingClientRect();
    targetPointer = ((e.clientX - r.left) / r.width - 0.5) * 2;
  });
  container.addEventListener("pointerleave", () => { targetPointer = 0; });
  addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { targetSteer = -1; e.preventDefault(); }
    if (e.key === "ArrowRight") { targetSteer = 1; e.preventDefault(); }
  });
  addEventListener("keyup", (e) => {
    if (e.key.startsWith("Arrow")) targetSteer = 0;
  });

  function resize() {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  // Only render while on screen and the tab is visible.
  let visible = true;
  new IntersectionObserver(
    ([entry]) => { visible = entry.isIntersecting; },
    { threshold: 0.01 }
  ).observe(container);

  let t = 0;
  function frame() {
    requestAnimationFrame(frame);
    if (!visible || document.hidden) return;

    t += 0.016;
    pointer += (targetPointer - pointer) * 0.06;
    steer += (targetSteer - steer) * 0.08;
    const lead = pointer + steer;

    // Slow turntable, nudged by the cursor rather than driven by it.
    car.rotation.y = 0.30 + Math.sin(t * 0.18) * 0.14 + lead * 0.38;
    car.rotation.z = -lead * 0.045;          // roll into the direction of travel
    car.position.y = -0.25 + (reduced ? 0 : Math.sin(t * 3.1) * 0.012); // suspension

    if (!reduced) {
      const spin = 0.42 + Math.abs(lead) * 0.25;
      wheels.forEach((w) => { w.rotation.z -= spin; });
    }

    // Camera drifts a little so the shot never feels frozen.
    camera.position.set(
      -5.9 + lead * 0.5,
      1.30 + Math.sin(t * 0.24) * 0.10,
      5.4 - lead * 0.7
    );
    camera.lookAt(-0.15, 0.34, 0);
    renderer.render(scene, camera);
  }
  frame();

  return { renderer, scene, car };
}
