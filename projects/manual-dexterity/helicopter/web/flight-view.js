import * as THREE from '/vendor/three.module.js';

export function createFlightView(container) {
const el = id => id === 'scene' ? container : document.getElementById(id);
const put = (id, value) => { const node = el(id); if (node) node.textContent = value; };
let state = null;
let lastSuccess = 0;
let scene, camera, renderer, aircraft, mainRotor, tailRotor, shadow, targetMarker, targetStem, flightLine;
let renderAvailable = false;
let lastFrame = performance.now();
const orbit = { yaw: -0.98, pitch: 0.54, distance: 13, focus: new THREE.Vector3(1.15, 1.15, 1.2) };
const desiredPosition = new THREE.Vector3();
const desiredQuaternion = new THREE.Quaternion();
const material = (color, roughness = 0.7, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });

function mesh(geometry, surface, parent, position = [0, 0, 0]) {
  const item = new THREE.Mesh(geometry, surface);
  item.position.set(...position);
  item.castShadow = true;
  item.receiveShadow = true;
  parent.add(item);
  return item;
}

function rod(a, b, radius, surface, parent, segments = 10) {
  const start = new THREE.Vector3(...a);
  const end = new THREE.Vector3(...b);
  const delta = end.clone().sub(start);
  const item = mesh(new THREE.CylinderGeometry(radius, radius, delta.length(), segments), surface, parent);
  item.position.copy(start).add(end).multiplyScalar(0.5);
  item.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
  return item;
}

function makeAircraft() {
  const group = new THREE.Group();
  const shell = material(0xe28a51, 0.5);
  const white = material(0xf4edd7, 0.6);
  const black = material(0x263f3b, 0.55, 0.15);
  const glass = material(0x416766, 0.2, 0.35);
  const silver = material(0x82988c, 0.5, 0.4);
  const body = mesh(new THREE.SphereGeometry(0.62, 32, 20), shell, group);
  body.scale.set(1.25, 0.73, 0.8);
  const cockpit = mesh(new THREE.SphereGeometry(0.48, 32, 20), glass, group, [0.47, 0, 0.09]);
  cockpit.scale.set(1.14, 0.84, 0.81);
  const chin = mesh(new THREE.SphereGeometry(0.44, 24, 16), white, group, [0.38, 0, -0.24]);
  chin.scale.set(1.17, 0.87, 0.45);
  rod([0.58, 0, 0.47], [0.94, 0, -0.15], 0.021, white, group);
  rod([-0.25, 0, -0.05], [-2.05, 0, 0.23], 0.07, shell, group);
  rod([-0.3, 0, -0.04], [-1.9, 0, 0.21], 0.035, white, group);
  const fin = mesh(new THREE.BoxGeometry(0.36, 0.045, 0.73), shell, group, [-1.89, 0, 0.49]);
  fin.rotation.y = -0.28;
  mesh(new THREE.BoxGeometry(0.33, 0.76, 0.05), white, group, [-1.61, 0, 0.22]);
  const engine = mesh(new THREE.SphereGeometry(0.34, 20, 12), white, group, [-0.24, 0, 0.38]);
  engine.scale.set(1.24, 0.78, 0.48);
  rod([-0.12, 0, 0.4], [-0.12, 0, 0.79], 0.049, black, group);
  for (const side of [-1, 1]) {
    rod([-0.8, side * 0.47, -0.64], [0.76, side * 0.47, -0.64], 0.032, black, group);
    rod([0.76, side * 0.47, -0.64], [0.93, side * 0.47, -0.52], 0.032, black, group);
    rod([0.31, side * 0.29, -0.3], [0.38, side * 0.47, -0.61], 0.027, silver, group);
    rod([-0.43, side * 0.28, -0.31], [-0.46, side * 0.47, -0.61], 0.027, silver, group);
    mesh(new THREE.BoxGeometry(0.38, 0.01, 0.047), white, group, [-0.25, side * 0.446, -0.03]);
    mesh(new THREE.SphereGeometry(0.035, 10, 6), material(side > 0 ? 0x97ce88 : 0xcb5735), group, [-0.2, side * 0.45, 0.04]);
  }
  mainRotor = new THREE.Group();
  mainRotor.position.set(-0.12, 0, 0.79);
  group.add(mainRotor);
  for (let i = 0; i < 4; i++) {
    const blade = new THREE.Group();
    blade.rotation.z = i * Math.PI / 2;
    mainRotor.add(blade);
    mesh(new THREE.BoxGeometry(1.6, 0.095, 0.012), black, blade, [0.92, 0.02, 0]);
    mesh(new THREE.BoxGeometry(0.13, 0.099, 0.015), white, blade, [1.64, 0.02, 0]);
  }
  mesh(new THREE.SphereGeometry(0.09, 16, 10), silver, mainRotor);
  const rotorDisc = mesh(new THREE.CircleGeometry(1.71, 64), new THREE.MeshBasicMaterial({ color: 0x829080, opacity: 0.075, transparent: true, side: THREE.DoubleSide, depthWrite: false }), group, [-0.12, 0, 0.789]);
  rotorDisc.castShadow = false;
  tailRotor = new THREE.Group();
  tailRotor.position.set(-1.97, -0.1, 0.35);
  group.add(tailRotor);
  mesh(new THREE.BoxGeometry(0.52, 0.025, 0.045), black, tailRotor);
  mesh(new THREE.BoxGeometry(0.045, 0.025, 0.52), black, tailRotor);
  mesh(new THREE.SphereGeometry(0.047, 10, 6), white, tailRotor);
  // The pose origin is the physical centre of mass; skids touch z=0 at z=0.20 m.
  group.scale.set(0.43, 0.43, 0.3125);
  return group;
}

function labelTexture(text, color = '#687a60', size = 64) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 128;
  const context = canvas.getContext('2d');
  context.fillStyle = color;
  context.textAlign = 'center'; context.textBaseline = 'middle';
  context.font = `600 ${size}px sans-serif`;
  context.fillText(text, 128, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function groundLabel(text, x, y, scale, color) {
  const label = mesh(new THREE.PlaneGeometry(scale * 2, scale), new THREE.MeshBasicMaterial({ map: labelTexture(text, color), transparent: true, depthWrite: false }), scene, [x, y, 0.012]);
  label.castShadow = false;
  return label;
}

function initScene() {
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe5e7d9);
  scene.fog = new THREE.Fog(0xe5e7d9, 26, 65);
  camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
  camera.up.set(0, 0, 1);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;
  renderer.domElement.setAttribute('aria-label', 'Interactive 3D flight view. Drag to orbit, scroll to zoom.');
  renderer.domElement.setAttribute('role', 'img');
  el('scene').append(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xfff9dd, 0x8b9e89, 2.8));
  const sun = new THREE.DirectionalLight(0xfff7d5, 3.8);
  sun.position.set(5, -7, 17);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -15, right: 15, top: 15, bottom: -15, near: 1, far: 45 });
  sun.shadow.bias = -0.001;
  sun.shadow.normalBias = 0.04;
  scene.add(sun);
  const ground = mesh(new THREE.PlaneGeometry(200, 200), material(0xe0e4d4), scene, [0, 0, -0.021]);
  ground.castShadow = false;
  const grid = [];
  for (let n = -25; n <= 25; n++) {
    grid.push(-25, n, 0.001, 25, n, 0.001, n, -25, 0.001, n, 25, 0.001);
  }
  scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(grid, 3)), new THREE.LineBasicMaterial({ color: 0xabbba0, transparent: true, opacity: 0.28 })));
  const pad = mesh(new THREE.CircleGeometry(1.55, 80), material(0xc4cfb9), scene, [0, 0, 0.004]);
  pad.castShadow = false;
  mesh(new THREE.RingGeometry(1.35, 1.39, 80), new THREE.MeshBasicMaterial({ color: 0xf3f4e7, side: THREE.DoubleSide }), scene, [0, 0, 0.009]).castShadow = false;
  groundLabel('H', 0, 0, 1.0, '#f1f2df');
  groundLabel('01', -2.1, 0, 0.44, '#8fa080');
  groundLabel('HOME', 0, -2, 0.42, '#8b9c7e');
  for (const [x, y] of [[3, 0], [3, 3]]) {
    mesh(new THREE.RingGeometry(0.21, 0.23, 32), new THREE.MeshBasicMaterial({ color: 0xaab99c, transparent: true, opacity: 0.5 }), scene, [x, y, 0.015]).castShadow = false;
  }
  const grass = material(0x9fb391);
  const curb = material(0xc1ceb3);
  for (const [x, y, width, length] of [[-10, 0, 0.08, 18], [10, 0, 0.08, 18], [0, 9, 20, 0.08], [0, -9, 20, 0.08]]) {
    mesh(new THREE.BoxGeometry(width, length, 0.035), curb, scene, [x, y, 0]);
  }
  for (const [x, y] of [[-9, 8], [9, 8], [-9, -8], [9, -8]]) {
    const marker = mesh(new THREE.CylinderGeometry(0.09, 0.14, 0.45, 12), grass, scene, [x, y, 0.225]);
    marker.rotation.x = Math.PI / 2;
    const top = mesh(new THREE.SphereGeometry(0.09, 10, 6), material(0xf3f4d9), scene, [x, y, 0.46]);
    top.scale.z = 0.3;
  }
  const compassColor = new THREE.LineBasicMaterial({ color: 0x9aad8d });
  const compass = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(7, -5, 0.02), new THREE.Vector3(9, -5, 0.02), new THREE.Vector3(8.7, -4.85, 0.02), new THREE.Vector3(9, -5, 0.02), new THREE.Vector3(8.7, -5.15, 0.02)]);
  scene.add(new THREE.Line(compass, compassColor));
  groundLabel('+X', 9.6, -5, 0.36, '#95a588');
  aircraft = makeAircraft();
  aircraft.position.z = 0.65;
  scene.add(aircraft);
  shadow = mesh(new THREE.CircleGeometry(0.85, 40), new THREE.MeshBasicMaterial({ color: 0x536d49, transparent: true, opacity: 0.12, depthWrite: false }), scene, [0, 0, 0.018]);
  shadow.castShadow = false;
  targetMarker = new THREE.Group();
  const targetMaterial = new THREE.LineBasicMaterial({ color: 0x599381, transparent: true, opacity: 0.8 });
  const targetPoints = [];
  for (let i = 0; i <= 64; i++) targetPoints.push(new THREE.Vector3(0.32 * Math.cos(i * Math.PI / 32), 0.32 * Math.sin(i * Math.PI / 32), 0));
  targetMarker.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(targetPoints), targetMaterial));
  const cross = [-0.44, 0, 0, -0.2, 0, 0, 0.2, 0, 0, 0.44, 0, 0, 0, -0.44, 0, 0, -0.2, 0, 0, 0.2, 0, 0, 0.44, 0];
  targetMarker.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(cross, 3)), targetMaterial));
  scene.add(targetMarker);
  targetStem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineDashedMaterial({ color: 0x79a18e, dashSize: 0.09, gapSize: 0.09, transparent: true, opacity: 0.55 }));
  scene.add(targetStem);
  flightLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xd48152, transparent: true, opacity: 0.75 }));
  scene.add(flightLine);
  const resize = () => {
    const box = el('scene').getBoundingClientRect();
    renderer.setSize(box.width, box.height, false);
    camera.aspect = box.width / Math.max(box.height, 1);
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el('scene'));
  resize();
  let drag = null;
  renderer.domElement.addEventListener('pointerdown', event => {
    drag = { x: event.clientX, y: event.clientY };
    renderer.domElement.setPointerCapture(event.pointerId);
  });
  renderer.domElement.addEventListener('pointermove', event => {
    if (!drag) return;
    orbit.yaw -= (event.clientX - drag.x) * 0.006;
    orbit.pitch = Math.max(0.12, Math.min(1.42, orbit.pitch + (event.clientY - drag.y) * 0.004));
    drag = { x: event.clientX, y: event.clientY };
  });
  renderer.domElement.addEventListener('pointerup', () => { drag = null; });
  renderer.domElement.addEventListener('pointercancel', () => { drag = null; });
  renderer.domElement.addEventListener('wheel', event => {
    event.preventDefault();
    orbit.distance = Math.max(6, Math.min(42, orbit.distance * Math.exp(event.deltaY * 0.001)));
  }, { passive: false });
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    renderAvailable = false;
    showRenderError('The browser lost its 3D graphics context. Telemetry is still live. Reload the page to restore the flight view.');
  });
  renderAvailable = true;
  requestAnimationFrame(animate);
}

function showRenderError(message) {
  put('render-error', message);
  el('render-error').hidden = false;
}

function animate(now) {
  requestAnimationFrame(animate);
  if (!renderAvailable) return;
  const dt = Math.min(0.08, (now - lastFrame) / 1000);
  lastFrame = now;
  if (state) {
    aircraft.position.copy(desiredPosition);
    aircraft.quaternion.copy(desiredQuaternion);
    const active = state.running && !state.crashed && !state.completed && performance.now() - lastSuccess < 2500;
    if (active) {
      mainRotor.rotation.z += dt * 49;
      tailRotor.rotation.y += dt * 77;
    }
    shadow.position.set(state.position[0], state.position[1], 0.018);
    const altitude = Math.max(0, state.position[2]);
    shadow.scale.setScalar(1 + altitude * 0.07);
    shadow.material.opacity = Math.max(0.025, 0.14 - altitude * 0.018);
  }
  const horizontal = orbit.distance * Math.cos(orbit.pitch);
  camera.position.set(orbit.focus.x + horizontal * Math.cos(orbit.yaw), orbit.focus.y + horizontal * Math.sin(orbit.yaw), orbit.focus.z + orbit.distance * Math.sin(orbit.pitch));
  camera.lookAt(orbit.focus);
  renderer.render(scene, camera);
}

try { initScene(); } catch (error) {
  console.error('3D flight view initialization failed:', error);
  showRenderError('WebGL is unavailable. Telemetry and simulation controls remain available.');
}
return {
  update(data) {
    state = data;
    lastSuccess = performance.now();
    desiredPosition.set(...data.position);
    desiredQuaternion.set(data.quaternion[1], data.quaternion[2], data.quaternion[3], data.quaternion[0]).normalize();
    if (!renderAvailable) return;
    targetMarker.position.set(...data.target);
    const points = targetStem.geometry.attributes.position;
    points.setXYZ(0, data.target[0], data.target[1], 0.02);
    points.setXYZ(1, ...data.target);
    points.needsUpdate = true;
    targetStem.computeLineDistances();
    targetStem.geometry.computeBoundingSphere();
  },
  setTrail(points) {
    if (!renderAvailable) return;
    flightLine.geometry.dispose();
    flightLine.geometry = new THREE.BufferGeometry().setFromPoints(points.map(point => new THREE.Vector3(...point)));
  },
  resetCamera() {
    orbit.yaw = -0.98; orbit.pitch = 0.54; orbit.distance = 13;
  }
};
}
