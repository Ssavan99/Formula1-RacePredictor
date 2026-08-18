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
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);

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
    color: RED, metalness: 0.45, roughness: 0.32,
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
  box(1.5, 0.42, 0.95, bodyMat, -0.55, 0.44, 0);
  box(1.3, 0.34, 0.66, bodyMat, -1.42, 0.42, 0);
  box(1.0, 0.26, 0.40, bodyMat, -2.10, 0.42, 0);

  // Nose cone, thin and low.
  const nose = box(0.9, 0.20, 0.26, bodyMat, -2.72, 0.38, 0);
  nose.rotation.z = -0.05;

  // Sidepods: bulky at the front, tapering into the coke-bottle.
  [-1, 1].forEach((side) => {
    box(1.55, 0.44, 0.42, bodyMat, 0.30, 0.45, side * 0.52);
    box(0.95, 0.30, 0.26, bodyMat, 1.25, 0.42, side * 0.36);
    // inlet
    box(0.10, 0.30, 0.34, carbonMat, -0.50, 0.47, side * 0.53);
  });

  // Engine cover + airbox above the driver.
  box(1.9, 0.40, 0.44, bodyMat, 0.85, 0.72, 0);
  box(0.55, 0.34, 0.34, carbonMat, -0.02, 0.86, 0);

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
  box(0.62, 0.045, 2.00, wingMat, -3.05, 0.16, 0);
  box(0.42, 0.040, 1.90, bodyMat, -2.92, 0.27, 0);
  [-1, 1].forEach((s) => box(0.70, 0.34, 0.05, bodyMat, -3.02, 0.28, s * 1.0));

  // Rear wing: raised main plane on twin pillars, with endplates.
  box(0.66, 0.05, 1.05, wingMat, 2.55, 1.02, 0);
  box(0.46, 0.045, 1.02, bodyMat, 2.72, 0.86, 0);
  [-1, 1].forEach((s) => box(0.80, 0.46, 0.05, bodyMat, 2.60, 0.86, s * 0.52));
  box(0.10, 0.62, 0.10, carbonMat, 2.45, 0.62, 0);
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
  scene.add(new THREE.HemisphereLight(0x9fb4ff, 0x0a0c10, 0.55));

  const key = new THREE.DirectionalLight(0xffffff, 2.6);
  key.position.set(4, 6, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 24;
  key.shadow.camera.left = -6;
  key.shadow.camera.right = 6;
  key.shadow.camera.top = 6;
  key.shadow.camera.bottom = -6;
  scene.add(key);

  // Red rim light from behind — this is what gives the silhouette its edge.
  const rim = new THREE.DirectionalLight(0xff2a44, 3.0);
  rim.position.set(-6, 2.5, -5);
  scene.add(rim);

  const fill = new THREE.DirectionalLight(0x6f8cff, 0.9);
  fill.position.set(-3, 1.5, 6);
  scene.add(fill);

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
    car.rotation.y = -0.55 + Math.sin(t * 0.18) * 0.16 + lead * 0.42;
    car.rotation.z = -lead * 0.045;          // roll into the direction of travel
    car.position.y = -0.25 + (reduced ? 0 : Math.sin(t * 3.1) * 0.012); // suspension

    if (!reduced) {
      const spin = 0.42 + Math.abs(lead) * 0.25;
      wheels.forEach((w) => { w.rotation.z -= spin; });
    }

    // Camera drifts a little so the shot never feels frozen.
    camera.position.set(
      6.4 + lead * 0.5,
      2.5 + Math.sin(t * 0.24) * 0.22,
      6.0 - lead * 0.9
    );
    camera.lookAt(0, 0.25, 0);
    renderer.render(scene, camera);
  }
  frame();

  return { renderer, scene, car };
}
