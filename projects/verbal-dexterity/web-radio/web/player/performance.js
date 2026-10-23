/* Read-only, playback-timed visual performance. No speaker inference. */
"use strict";
(() => {
  const EMPTY = Object.freeze({});
  const clamp = value => Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

  function activeSpeakers(performance, offset) {
    if (!Number.isFinite(offset) || offset < 0 || offset >= performance.duration_ms) return [];
    const active = new Set();
    for (const cue of performance.cues) {
      if (cue.start_ms > offset) break;
      if (cue.start_ms <= offset && offset < cue.end_ms) active.add(cue.speaker_id);
    }
    return [...active];
  }

  function sampleTrack(track, hop, offset) {
    const index = Math.min(track.energy.length - 1, Math.max(0, offset / hop));
    const left = Math.floor(index);
    const right = Math.min(left + 1, track.energy.length - 1);
    const fraction = index - left;
    const value = band => clamp(track[band][left] + (track[band][right] - track[band][left]) * fraction);
    return {energy: value("energy"), bands: [value("low"), value("mid"), value("high")]};
  }

  function ownsAudioWindow(performance, id, first, last) {
    return performance.cues.some(cue => cue.speaker_id === id &&
      cue.start_ms <= first && cue.end_ms >= last) &&
      !performance.cues.some(cue => cue.speaker_id !== id &&
        cue.start_ms < last && cue.end_ms > first);
  }

  function sample(performance, offset, mixed = null) {
    const active = activeSpeakers(performance, offset);
    const result = {};
    for (const id of active) {
      const tracks = performance.features && performance.features.tracks;
      const track = tracks && Object.hasOwn(tracks, id) ? tracks[id] : null;
      if (track) result[id] = sampleTrack(track, performance.features.hop_ms, offset);
      // A single timed speaker can use the mixed signal. Overlaps require separate tracks.
      else if (active.length === 1 && mixed) {
        const owns = duration => duration === undefined ||
          Number.isFinite(duration) && duration >= 0 &&
          ownsAudioWindow(performance, id, offset - duration, offset);
        result[id] = {
          energy: owns(mixed.energy_window_ms) ? mixed.energy : 0,
          bands: owns(mixed.bands_window_ms) ? mixed.bands : [0, 0, 0]
        };
      }
      // A measured mixed waveform belongs to the sole timed voice only.
      // Overlapping voices need separate prepared waveforms.
      if (active.length === 1 && mixed && Array.isArray(mixed.waveform) && result[id]) {
        const ranges = mixed.waveform_ranges_ms;
        if (ranges !== undefined) {
          // Each contour point is owned only when its complete RMS source
          // footprint lies inside this character's cue and no other voice.
          const validRanges = Array.isArray(ranges) && ranges.length === mixed.waveform.length;
          const mask = mixed.waveform.map((_value, bin) => {
            const range = validRanges && ranges[bin];
            if (!Array.isArray(range) || range.length !== 2 ||
                !range.every(Number.isFinite) || range[0] >= range[1] || range[1] > 0) return false;
            const first = offset + range[0], last = offset + range[1];
            return ownsAudioWindow(performance, id, first, last);
          });
          result[id] = {...result[id],
            waveform: mixed.waveform.map((value, bin) => mask[bin] ? value : 0),
            waveform_mask: mask};
        } else {
          // Compatibility for older callers with only a whole-window extent.
          const windowMs = Number.isFinite(mixed.waveform_window_ms)
            ? Math.max(0, mixed.waveform_window_ms) : 0;
          const containsOtherVoice = performance.cues.some(cue =>
            cue.speaker_id !== id && cue.start_ms < offset && cue.end_ms > offset - windowMs);
          if (!containsOtherVoice) result[id] = {...result[id], waveform:mixed.waveform};
        }
      }
    }
    return result;
  }

  function validDocument(data, releaseId) {
    if (!data || data.schema_version !== 1 || data.release_id !== releaseId ||
        !/^[a-f0-9]{64}$/.test(data.package_sha256 || "") ||
        !Number.isInteger(data.duration_ms) || data.duration_ms < 1 || data.duration_ms > 1800000 ||
        !Array.isArray(data.cast) || !data.cast.length || data.cast.length > 12 ||
        !Array.isArray(data.cues) || data.cues.length > 100000) return false;
    const ids = new Set();
    for (const member of data.cast) {
      if (!member || !/^[a-z0-9][a-z0-9-]{0,127}$/.test(member.id || "") ||
          !/^#[a-fA-F0-9]{6}$/.test(member.color || "") || ids.has(member.id)) return false;
      ids.add(member.id);
    }
    let previous = -1;
    for (const cue of data.cues) {
      if (!cue || !ids.has(cue.speaker_id) || !Number.isInteger(cue.start_ms) ||
          !Number.isInteger(cue.end_ms) || cue.start_ms < previous || cue.start_ms < 0 ||
          cue.end_ms <= cue.start_ms || cue.end_ms > data.duration_ms) return false;
      previous = cue.start_ms;
    }
    if (data.features != null) {
      const features = data.features;
      if (!Number.isInteger(features.hop_ms) || features.hop_ms < 20 || features.hop_ms > 1000 ||
          !features.tracks || typeof features.tracks !== "object" || Array.isArray(features.tracks)) return false;
      const length = Math.ceil(data.duration_ms / features.hop_ms) + 1;
      let values = 0;
      for (const [id, track] of Object.entries(features.tracks)) {
        if (!ids.has(id) || !track) return false;
        for (const key of ["energy", "low", "mid", "high"]) {
          if (!Array.isArray(track[key]) || track[key].length !== length) return false;
          values += track[key].length;
          if (values > 500000 || track[key].some(value => !Number.isFinite(value) || value < 0 || value > 1)) return false;
        }
      }
    }
    return true;
  }

  class Timeline {
    constructor(scene, fetchJSON) {
      this.scene = scene;
      this.fetchJSON = fetchJSON;
      this.airing = null;
      this.document = null;
      this.releaseId = null;
      this.requestToken = 0;
      this.retryAfter = 0;
      this.pending = false;
    }

    async setAiring(airing) {
      const releaseId = airing && airing.release_id;
      this.airing = airing || null;
      if (releaseId === this.releaseId && (this.document || this.pending || Date.now() < this.retryAfter)) return;
      if (releaseId !== this.releaseId) {
        this.requestToken++;
        this.document = null;
        this.pending = false;
        this.retryAfter = 0;
        this.scene.setCast([]);
        this.scene.setFeatures(EMPTY);
      }
      this.releaseId = releaseId || null;
      if (!releaseId || !/^[a-z0-9][a-z0-9-]{0,127}$/.test(releaseId)) return;
      const token = ++this.requestToken;
      this.pending = true;
      try {
        const data = await this.fetchJSON("/api/performance/" + encodeURIComponent(releaseId));
        if (token !== this.requestToken) return;
        if (!validDocument(data, releaseId)) throw new Error("Invalid performance");
        this.document = data;
        this.scene.setCast(data.cast);
      } catch {
        if (token === this.requestToken) {
          this.document = null;
          this.retryAfter = Date.now() + 30000;
          this.scene.setCast([]);
          this.scene.setFeatures(EMPTY);
        }
      } finally {
        if (token === this.requestToken) this.pending = false;
      }
    }

    tick(timestamp, mixed, audible) {
      if (!audible) {
        this.scene.setFeatures(EMPTY);
        return false;
      }
      if (!this.document) {
        // One unassigned body may follow the mixed audio, without implying a character.
        this.scene.setFeatures(mixed ? {"__unassigned__": mixed} : EMPTY);
        return false;
      }
      const airing = this.airing;
      const start = airing && Date.parse(airing.stream_start);
      if (!airing || !airing.timing_calibrated || !Number.isFinite(timestamp) || !Number.isFinite(start)) {
        this.scene.setFeatures(EMPTY);
        return false;
      }
      const offset = timestamp - start;
      const end = airing.stream_end && Date.parse(airing.stream_end);
      if (offset < 0 || offset >= this.document.duration_ms || (Number.isFinite(end) && timestamp >= end)) {
        this.scene.setFeatures(EMPTY);
        return true;
      }
      this.scene.setFeatures(sample(this.document, offset, mixed));
      return false;
    }
  }

  window.RadioPerformance = {Timeline, activeSpeakers, sampleTrack, sample, validDocument};
})();
