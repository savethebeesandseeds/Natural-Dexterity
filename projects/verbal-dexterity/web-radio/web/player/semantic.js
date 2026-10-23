/* A local audition uses its own media clock; it never joins the station rundown. */
"use strict";
(() => {
  const slug = /^[a-z0-9][a-z0-9-]{0,127}$/;
  const hash = /^[a-f0-9]{64}$/;
  function validPreview(data, id, audioHash) {
    if (!data || data.preview_id !== id || data.audio_sha256 !== audioHash) return false;
    const {preview_id, audio_sha256, ...timeline} = data;
    return RadioPerformance.validDocument({...timeline, release_id:id, package_sha256:audioHash}, id);
  }
  function validSemantic(data, id, audioHash, duration, cast) {
    if (!data || data.schema_version !== 1 || data.preview_id !== id || data.audio_sha256 !== audioHash ||
        data.duration_ms !== duration || data.model !== "gemini-embedding-2" || !hash.test(data.space_id || "") ||
        !data.projection || data.projection.method !== "pca" || ![3,4,5].includes(data.projection.dimensions) ||
        data.projection.fitted_count !== data.windows?.length || !hash.test(data.projection.basis_sha256 || "") ||
        !Array.isArray(data.windows) || !data.windows.length || data.windows.length > 2000) return false;
    const ids = new Set(cast.map(item=>item.id));
    const seen = new Set();
    let previous = -1;
    return data.windows.every(window => {
      const good = window && typeof window.id === "string" && !seen.has(window.id) && ids.has(window.character_id) && Number.isInteger(window.start_ms) &&
        Number.isInteger(window.end_ms) && window.start_ms >= previous && window.start_ms >= 0 &&
        window.end_ms > window.start_ms && window.end_ms <= duration &&
        Array.isArray(window.position) && window.position.length === 3 &&
        window.position.every(value=>Number.isFinite(value) && Math.abs(value)<=1) &&
        (data.projection.dimensions >= 4
          ? Number.isFinite(window.rotation) && Math.abs(window.rotation)<=1
          : window.rotation == null) &&
        (data.projection.dimensions === 5
          ? Number.isFinite(window.depth_rotation) && Math.abs(window.depth_rotation)<=1
          : window.depth_rotation == null);
      if (good) { previous = window.start_ms; seen.add(window.id); }
      return good;
    });
  }
  function positionsAt(windows, offset) {
    const positions = {};
    if (!Number.isFinite(offset)) return positions;
    for (const window of windows) {
      if (window.start_ms > offset) break;
      positions[window.character_id] = window.position;
    }
    return positions;
  }
  function rotationsAt(windows, offset) {
    const rotations = {};
    if (!Number.isFinite(offset)) return rotations;
    for (const window of windows) {
      if (window.start_ms > offset) break;
      if (Number.isFinite(window.rotation)) rotations[window.character_id] = window.rotation;
    }
    return rotations;
  }
  function depthRotationsAt(windows, offset) {
    const rotations = {};
    if (!Number.isFinite(offset)) return rotations;
    for (const window of windows) {
      if (window.start_ms > offset) break;
      if (Number.isFinite(window.depth_rotation)) rotations[window.character_id] = window.depth_rotation;
    }
    return rotations;
  }
  class Recording {
    constructor(scene, fetchJSON) {
      this.scene=scene; this.fetchJSON=fetchJSON; this.metadata=null;
      this.document=null; this.semantic=null; this.pending=false;
    }
    async load(id, {library=false, staticSite=false} = {}) {
      if (!slug.test(id) || (staticSite && (!library ||
          !/^(?:creation|project)-[0-9a-f]{24}$/.test(id)))) throw new Error("Invalid recording");
      this.pending=true;
      try {
        const metadataPath=staticSite ? "/data/recordings/"+id+".json" :
          (library ? "/api/creation-library/" : "/api/previews/")+id;
        const metadata=await this.fetchJSON(metadataPath);
        // Project aliases resolve once; every asset remains pinned to that concrete version.
        const projectAlias=library && /^project-[0-9a-f]{24}$/.test(id);
        const resolvedId=projectAlias ? metadata?.preview_id : id;
        if (projectAlias && (typeof resolvedId!=="string" ||
            !/^creation-[0-9a-f]{24}$/.test(resolvedId))) throw new Error("Invalid recording");
        const base=(library ? "/creation-library/" : "/preview/")+resolvedId+"/";
        if (!metadata || metadata.preview_id!==resolvedId || metadata.local_preview!==true ||
            !metadata.audio || metadata.audio.url!==base+"audio.wav" || !hash.test(metadata.audio.sha256||"") ||
            !metadata.performance || metadata.performance.url!==base+"performance.json") throw new Error("Invalid recording");
        const document=await this.fetchJSON(metadata.performance.url);
        if (!validPreview(document,resolvedId,metadata.audio.sha256) || document.duration_ms!==metadata.duration_ms) throw new Error("Invalid recording");
        let semantic=null;
        if (metadata.semantic) {
          if(metadata.semantic.url!==base+"semantic.json")throw new Error("Invalid semantic source");
          semantic=await this.fetchJSON(metadata.semantic.url);
          if(!validSemantic(semantic,resolvedId,metadata.audio.sha256,metadata.duration_ms,document.cast))throw new Error("Invalid semantic timeline");
        }
        this.metadata=metadata; this.document=document; this.semantic=semantic;
        this.scene.setCast(document.cast);
        this.scene.setPositions(semantic ? positionsAt(semantic.windows,0) : null);
        this.scene.setRotations(semantic ? rotationsAt(semantic.windows,0) : null);
        this.scene.setDepthRotations(semantic ? depthRotationsAt(semantic.windows,0) : null);
        return metadata;
      } finally { this.pending=false; }
    }
    tick(offset,mixed,audible) {
      if(!this.document)return false;
      this.scene.setFeatures(audible ? RadioPerformance.sample(this.document,offset,mixed) : {});
      if(this.semantic) {
        this.scene.setPositions(positionsAt(this.semantic.windows,offset));
        this.scene.setRotations(rotationsAt(this.semantic.windows,offset));
        this.scene.setDepthRotations(depthRotationsAt(this.semantic.windows,offset));
      }
      return false;
    }
  }
  window.RadioSemantic=Object.freeze({Recording,Preview:Recording,positionsAt,rotationsAt,depthRotationsAt,validRecording:validPreview,validPreview,validSemantic});
})();
