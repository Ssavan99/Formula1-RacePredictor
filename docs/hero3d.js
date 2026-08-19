/**
 * Scroll-driven 3D hero: McLaren MP4/5.
 *
 * Modelled on the car-commercial scroll references. The stage is PINNED and
 * scroll progress drives a camera orbit plus an exploded view -- the car never
 * travels over the page, which is the failure mode every drivable-car attempt
 * ran into.
 *
 * The car sits BEHIND the page content, dimmed, and the whole site scrolls over
 * it. Scroll drives a continuous orbit; the exploded view rises and falls again
 * so the car is whole at both ends and only comes apart in the middle:
 *
 * Motion is driven by scroll DISTANCE, not by progress through the page. Tying
 * it to a 0-1 fraction of total height meant a short page (empty tables, a race
 * with no results yet) compressed the whole cycle into a few hundred pixels, so
 * the car never visibly reassembled -- it was already past the end of the curve.
 * A fixed rate per pixel turns at the same speed however long the page is.
 *
 *   rotation   continuous, ANGLE_PER_PX radians per pixel scrolled
 *   explode    one bell curve per CYCLE_PX, so it always comes back together
 *
 * The model is CC Attribution by dark_igorek. Draco-compressed and texture-
 * reduced from 105.8MB to 2.5MB, with all 16 meshes kept separate so the
 * exploded phase has something to explode.
 */
import * as THREE from "./vendor/three.module.min.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";
import { DRACOLoader } from "./vendor/DRACOLoader.js";

const MODEL_URL = "./model/mp45.glb";

export async function mountHero(canvas, { onProgress = () => {} } = {}) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch {
    return null;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 200);

  // --- lighting: a studio, not a sunny day -------------------------------
  scene.add(new THREE.HemisphereLight(0x8fa6c8, 0x0a0b0d, 0.7));

  const key = new THREE.DirectionalLight(0xffffff, 3.4);
  key.position.set(5, 8, 6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { near: 1, far: 40, left: -8, right: 8, top: 8, bottom: -8 });
  scene.add(key);

  const rim = new THREE.DirectionalLight(0xff3b2f, 2.2);   // papaya-ish edge
  rim.position.set(-7, 3, -6);
  scene.add(rim);

  const fill = new THREE.DirectionalLight(0x9ab6ff, 0.9);
  fill.position.set(-4, 2, 7);
  scene.add(fill);

  // Shadow catcher so the car is grounded rather than floating in space.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.ShadowMaterial({ opacity: 0.5 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  // --- load ---------------------------------------------------------------
  const draco = new DRACOLoader();
  draco.setDecoderPath("./vendor/draco/");
  const loader = new GLTFLoader();
  loader.setDRACOLoader(draco);

  const gltf = await new Promise((resolve, reject) =>
    loader.load(MODEL_URL, resolve, e => {
      if (e.total) onProgress(e.loaded / e.total);
    }, reject)
  );

  const car = gltf.scene;
  car.traverse(node => {
    if (node.isMesh) { node.castShadow = true; node.receiveShadow = true; }
  });

  // Normalise: centre on the origin, sit on the floor, scale to a known size.
  const box = new THREE.Box3().setFromObject(car);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const scale = 4.6 / Math.max(size.x, size.y, size.z);
  car.scale.setScalar(scale);
  car.position.set(-centre.x * scale, -box.min.y * scale, -centre.z * scale);

  const pivot = new THREE.Group();
  pivot.add(car);
  scene.add(pivot);
  ground.position.y = 0;

  // Record each part's outward direction so the exploded view opens the car up
  // rather than drifting every piece the same way.
  //
  // The displacement is applied to node.position, which is in the PARENT's local
  // space, while the bounding boxes are in world space and the car is uniformly
  // scaled. Dividing by that scale converts a world-space distance into the
  // local units the position actually uses -- without it the parts separate by
  // a fraction of the intended amount and the effect is invisible.
  const parts = [];
  const carCentre = new THREE.Box3().setFromObject(car).getCenter(new THREE.Vector3());
  car.traverse(node => {
    if (!node.isMesh) return;
    const partCentre = new THREE.Box3().setFromObject(node).getCenter(new THREE.Vector3());
    const direction = partCentre.clone().sub(carCentre);
    direction.y *= 0.30;                 // damp vertical travel; keep its sign
    if (direction.lengthSq() < 1e-6) direction.set(0, 1, 0);
    parts.push({ node, home: node.position.clone(), dir: direction.normalize().divideScalar(scale) });
  });

  // --- scroll-driven state ------------------------------------------------
  //: Pixels of scrolling per full explode-and-reassemble cycle. Set from the
  //: page's actual scrollable distance so the car is always whole again by the
  //: bottom: on a short page (empty tables, no results yet) a fixed 2400px
  //: cycle peaked at the last reachable pixel and never reassembled.
  let cyclePx = 2400;
  //: Radians of orbit per pixel scrolled. A full turn takes ~3600px.
  const ANGLE_PER_PX = (Math.PI * 2) / 3600;

  let scrolled = 0, shown = 0;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }, { threshold: 0.01 })
    .observe(canvas);

  function frame() {
    requestAnimationFrame(frame);
    if (!visible || document.hidden) return;

    // Ease toward the scroll target so flicks feel weighted rather than jumpy.
    shown += (scrolled - shown) * (reduced ? 1 : 0.09);

    // Orbit continuously: never snaps back, and turns at the same rate whether
    // the page is one screen long or ten.
    const angle = -0.55 + shown * ANGLE_PER_PX;
    // Radius and height breathe gently over the cycle instead of ramping to a
    // fixed end state, which only made sense when there was an "end".
    const swing = Math.sin((shown / cyclePx) * Math.PI * 2);
    const radius = 8.4 - swing * 0.7;
    const height = 1.85 + swing * 0.55;
    camera.position.set(Math.sin(angle) * radius, Math.max(height, 0.75), Math.cos(angle) * radius);
    camera.lookAt(0, 0.76, 0);

    // Explode on a bell curve, once per CYCLE_PX: apart in the middle of each
    // cycle, whole at both ends, repeating for as long as the page scrolls.
    const cycle = ((shown % cyclePx) + cyclePx) % cyclePx / cyclePx;
    const bell = Math.sin(Math.PI * cycle);
    const eased = bell * bell * (3 - 2 * bell) / 2 + bell / 2;  // fuller peak
    for (const part of parts) {
      part.node.position.copy(part.home).addScaledVector(part.dir, eased * 1.75);
    }
    // A slow counter-rotation while apart makes the separation read as
    // deliberate rather than as the model falling over.
    pivot.rotation.y = eased * 0.35;

    renderer.render(scene, camera);
  }
  frame();

  return {
    /** Drive the animation from absolute scroll position, in pixels. */
    setScroll(px) { scrolled = Math.max(0, px || 0); },
    /** Back-compat for callers that only know page progress (0-1). */
    setProgress(value) { scrolled = Math.min(Math.max(value, 0), 1) * cyclePx; },
    /**
     * Tell the car how far the page can actually scroll, so one cycle fits.
     * Clamped: too short and it spins frantically, too long and nothing moves.
     */
    setExtent(px) { cyclePx = Math.min(Math.max(px || 0, 700), 3200); },
    parts: parts.length,
  };
}
