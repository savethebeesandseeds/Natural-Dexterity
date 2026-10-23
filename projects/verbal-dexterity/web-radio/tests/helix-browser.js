"use strict";
(async () => {
  const tests = [];
  const assert = (value, message) => { if (!value) throw new Error(message || "Assertion failed"); };
  const test = async (name, run) => {
    const item = document.createElement("li");
    try { await run(); item.textContent = "PASS · " + name; item.dataset.result = "pass"; tests.push(true); }
    catch (error) { item.textContent = "FAIL · " + name + ": " + error.message; item.dataset.result = "fail"; tests.push(false); }
    document.getElementById("results").append(item);
  };
  const base = () => ({
    schema_version:1, release_id:"fixture-one", package_sha256:"0".repeat(64), duration_ms:1000,
    cast:[{id:"a",color:"#518e89"},{id:"b",color:"#b08c50"}],
    cues:[{speaker_id:"a",start_ms:0,end_ms:500},{speaker_id:"b",start_ms:500,end_ms:1000}]
  });
  const featureTrack = energy => ({energy:[energy,energy],low:[.1,.1],mid:[.2,.2],high:[.3,.3]});
  const mixed = {energy:.6,bands:[.3,.4,.2]};
  const {sample, activeSpeakers, validDocument, Timeline} = RadioPerformance;
  await test("Continuous helix has depth and a long axis", () => {
    const points = RadioSpirits.projectHelix({radius:20,length:150,points:150,waveform:Array(64).fill(.5),renderer:{baseAngleDegrees:0,axialTiltDegrees:17,yawDegrees:7,lengthScale:1,tailFraction:0}});
    assert(points.length===151);
    assert(points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&Number.isFinite(p.z)));
    const span = key => Math.max(...points.map(p=>p[key]))-Math.min(...points.map(p=>p[key]));
    assert(span("y")>span("x")*2 && span("z")>20);
  });
  await test("Zero waveform is a straight axis with no resting coil", () => {
    for (const radius of [0,20]) {
      const points=RadioSpirits.projectHelix({
        radius,length:180,energy:.8,bands:[.6,.8,.4],
        waveform:Array(64).fill(0),tilt:.9,pitch:.3,yaw:.12
      });
      const a=points[0],b=points[points.length-1];
      const dx=b.x-a.x,dy=b.y-a.y;
      assert(Math.hypot(dx,dy)>100);
      for(const point of points) {
        assert(Number.isFinite(point.x)&&Number.isFinite(point.y)&&Number.isFinite(point.z));
        assert(Math.abs((point.x-a.x)*dy-(point.y-a.y)*dx)<1e-8);
        assert(point.front===.5);
      }
    }
    const dot=RadioSpirits.projectHelix({radius:0,length:0});
    assert(dot.every(point=>point.x===0&&point.y===0&&point.z===0));
  });
  await test("Renderer options are bounded, immutable and independently reusable", () => {
    const defaults=RadioSpirits.DEFAULT_OPTIONS;
    const normalized=RadioSpirits.normalizeOptions({
      baseAngleDegrees:NaN,tailFraction:-1,turns:500,lengthScale:"2"
    });
    assert(Object.isFrozen(defaults)&&Object.isFrozen(normalized));
    assert(normalized.baseAngleDegrees===90 && normalized.tailFraction===0 && normalized.turns===10);
    assert(normalized.lengthScale===defaults.lengthScale && defaults.tailFraction===.12);
    const source={length:180,waveform:Array(64).fill(0)};
    const a=RadioSpirits.projectHelix({...source,renderer:{baseAngleDegrees:0}});
    const b=RadioSpirits.projectHelix({...source,renderer:{baseAngleDegrees:90}});
    const axis=points=>[points.at(-1).x-points[0].x,points.at(-1).y-points[0].y];
    const [ax,ay]=axis(a),[bx,by]=axis(b);
    assert(Math.abs((ax*bx+ay*by)/(Math.hypot(ax,ay)*Math.hypot(bx,by)))<1e-10);
  });
  await test("Envelope smoothing is frame-rate independent", () => {
    let thirty=0,sixty=0;
    for(let i=0;i<30;i++)thirty=RadioSpirits.smoothEnvelope(thirty,1,1/30);
    for(let i=0;i<60;i++)sixty=RadioSpirits.smoothEnvelope(sixty,1,1/60);
    assert(Math.abs(thirty-sixty)<1e-8);
  });
  await test("Speaker cues use half-open boundaries", () => {
    assert(activeSpeakers(base(),499).join()==="a");
    assert(activeSpeakers(base(),500).join()==="b");
    assert(activeSpeakers(base(),1000).length===0);
  });
  await test("Overlapping voices do not borrow the mixed signal", () => {
    const data=base();data.cues[0].end_ms=700;
    assert(Object.keys(sample(data,600,mixed)).length===0);
  });
  await test("Prepared tracks deform overlapping voices independently", () => {
    const data=base();data.cues[0].end_ms=700;
    data.features={hop_ms:1000,tracks:{a:featureTrack(.2),b:featureTrack(.8)}};
    assert(validDocument(data,"fixture-one"));
    const value=sample(data,600,mixed);assert(Math.abs(value.a.energy-.2)<1e-8 && Math.abs(value.b.energy-.8)<1e-8);
  });
  await test("RMS footprints mask prior voices without blanking the new voice", () => {
    const source={...mixed,waveform:Array(64).fill(.4),
      waveform_ranges_ms:Array.from({length:64},(_,i)=> i<32?[-150,-50]:[-30,-10])};
    const value=sample(base(),550,source).b;
    assert(value.waveform.slice(0,32).every(v=>v===0));
    assert(value.waveform.slice(32).every(v=>v===.4));
    assert(value.waveform_mask.slice(0,32).every(v=>v===false));
    assert(value.waveform_mask.slice(32).every(v=>v===true));
    const overlap=base();overlap.cues[0].end_ms=540;
    assert(sample(overlap,550,source).b.waveform.every(v=>v===0));
    source.waveform_ranges_ms[63]=[0,10];
    assert(sample(base(),550,source).b.waveform[63]===0);
  });
  await test("Unknown character references are rejected", () => {
    const data=base();data.cues[0].speaker_id="unknown";assert(!validDocument(data,"fixture-one"));
  });
  await test("Constructor character ID does not inherit a feature track", () => {
    const data=base();data.cast[1].id="constructor";data.cues[1].speaker_id="constructor";
    data.features={hop_ms:1000,tracks:{a:featureTrack(.2)}};
    assert(validDocument(data,"fixture-one"));
    assert(sample(data,600,mixed).constructor.energy===mixed.energy);
  });
  await test("No exact clock means quiet characters; pause stays quiet", async () => {
    let latest=null;const scene={setCast(){},setFeatures(value){latest=value;}};
    const timeline=new Timeline(scene,async()=>base());
    const start=Date.parse("2026-10-22T12:00:00Z");
    await timeline.setAiring({release_id:"fixture-one",stream_start:new Date(start).toISOString(),timing_calibrated:true});
    timeline.tick(null,mixed,true);assert(Object.keys(latest).length===0);
    timeline.tick(start+600,mixed,true);assert(latest.b.energy===mixed.energy);
    timeline.tick(start+600,mixed,false);assert(Object.keys(latest).length===0);
    assert(timeline.tick(start+1000,mixed,true)===true && Object.keys(latest).length===0);
  });
  await test("Late requests cannot replace the new programme cast", async () => {
    const resolves={};let cast=[];
    const scene={setCast(value){cast=value;},setFeatures(){}};
    const timeline=new Timeline(scene,path=>new Promise(resolve=>{resolves[path.split("/").pop()]=resolve;}));
    const first=timeline.setAiring({release_id:"fixture-one"});
    const second=timeline.setAiring({release_id:"fixture-two"});
    const next=base();next.release_id="fixture-two";next.cast=[{id:"c",color:"#9f7289"}];next.cues=[];
    resolves["fixture-two"](next);await second;
    resolves["fixture-one"](base());await first;
    assert(cast.length===1 && cast[0].id==="c");
  });
  await test("Cast count follows supplied identities", () => {
    const scene=new RadioSpirits.Scene(document.getElementById("fixture"));
    scene.setCast([{id:"fixture-teal",color:"#518e89"},{id:"fixture-gold",color:"#b08c50"},{id:"fixture-rose",color:"#a4778e"}]);
    assert(scene.cast.length===3);
    scene.setCast([{id:"fixture-teal",color:"#518e89"}]);assert(scene.cast.length===1);
    scene.setCast([{id:"fixture-teal",color:"#518e89"},{id:"fixture-gold",color:"#b08c50"},{id:"fixture-rose",color:"#a4778e"}]);
  });
  const projectAlias = "project-" + "1".repeat(24);
  const creationOne = "creation-" + "2".repeat(24);
  const creationTwo = "creation-" + "3".repeat(24);
  const recordingFixture = (id, {resolved=id, library=true, staticSite=false} = {}) => {
    const audioHash="a".repeat(64);
    const source=base();
    const {release_id,package_sha256,...timeline}=source;
    const performance={...timeline,preview_id:resolved,audio_sha256:audioHash};
    const semantic={
      schema_version:1,preview_id:resolved,audio_sha256:audioHash,duration_ms:1000,
      model:"gemini-embedding-2",space_id:"b".repeat(64),
      projection:{method:"pca",dimensions:5,fitted_count:2,basis_sha256:"c".repeat(64)},
      windows:[
        {id:"first",character_id:"a",start_ms:0,end_ms:500,position:[0,.1,.2],rotation:.3,depth_rotation:.4},
        {id:"second",character_id:"b",start_ms:500,end_ms:1000,position:[.2,.3,.4],rotation:.5,depth_rotation:.6}
      ]
    };
    const prefix=library?"/creation-library/":"/preview/";
    const assetBase=prefix+resolved+"/";
    const metadata={
      preview_id:resolved,local_preview:true,duration_ms:1000,
      audio:{url:assetBase+"audio.wav",sha256:audioHash},
      performance:{url:assetBase+"performance.json"},
      semantic:{url:assetBase+"semantic.json"}
    };
    const requested=staticSite ? "/data/recordings/"+id+".json" :
      (library?"/api/creation-library/":"/api/previews/")+id;
    const documents=new Map([
      [requested,metadata],[metadata.performance.url,performance],[metadata.semantic.url,semantic]
    ]);
    const calls=[],updates=[];
    const scene={};
    for(const method of ["setCast","setPositions","setRotations","setDepthRotations"]) {
      scene[method]=value=>updates.push({method,value});
    }
    const recording=new RadioSemantic.Recording(scene,async path=>{
      calls.push(path);
      if(!documents.has(path))throw new Error("Unexpected asset path: "+path);
      return documents.get(path);
    });
    return {recording,metadata,performance,semantic,calls,updates,assetBase,requested,id,library,staticSite};
  };
  const rejectedRecording = async fixture => {
    let rejected=false;
    try { await fixture.recording.load(fixture.id,{library:fixture.library,staticSite:fixture.staticSite}); }
    catch { rejected=true; }
    assert(rejected,"Mismatched recording was accepted");
    assert(fixture.recording.pending===false,"Pending state was not cleared");
    assert(fixture.recording.document===null && fixture.recording.semantic===null &&
      fixture.updates.length===0,"Rejected payload changed the scene");
  };
  await test("Project aliases pin performance and five-dimensional semantics to one concrete creation", async () => {
    const fixture=recordingFixture(projectAlias,{resolved:creationOne});
    const metadata=await fixture.recording.load(projectAlias,{library:true});
    assert(metadata.preview_id===creationOne);
    assert(fixture.recording.document.preview_id===creationOne);
    assert(fixture.recording.semantic.preview_id===creationOne);
    assert(fixture.recording.semantic.projection.dimensions===5);
    assert(fixture.calls.join("|")===[
      fixture.requested,fixture.assetBase+"performance.json",fixture.assetBase+"semantic.json"
    ].join("|"),"Assets were not pinned to the concrete version");
    assert(fixture.updates.map(item=>item.method).join()==="setCast,setPositions,setRotations,setDepthRotations");
  });
  await test("Exact creation and preview identities keep their existing loader contract", async () => {
    for(const [id,library] of [[creationOne,true],["fixture-preview",false]]) {
      const fixture=recordingFixture(id,{library});
      await fixture.recording.load(id,{library});
      assert(fixture.recording.document.preview_id===id && fixture.recording.semantic.preview_id===id);
      assert(fixture.calls[0]===(library?"/api/creation-library/":"/api/previews/")+id);
    }
  });
  await test("Alias resolution is limited to library project IDs and concrete creation IDs", async () => {
    await rejectedRecording(recordingFixture(projectAlias,{resolved:creationOne,library:false}));
    await rejectedRecording(recordingFixture(creationOne,{resolved:creationTwo}));
    for(const resolved of ["fixture-preview",projectAlias,"creation-not-a-concrete-id"]) {
      await rejectedRecording(recordingFixture(projectAlias,{resolved}));
    }
  });
  await test("An alias cannot leave audio or sidecars on a moving project asset URL", async () => {
    for(const asset of ["audio","performance","semantic"]) {
      const fixture=recordingFixture(projectAlias,{resolved:creationOne});
      fixture.metadata[asset].url="/creation-library/"+projectAlias+"/"+asset+
        (asset==="audio"?".wav":".json");
      await rejectedRecording(fixture);
    }
  });
  await test("Concrete performance and semantic identity, audio hash and timing checks survive alias resolution", async () => {
    const mutations=[
      fixture=>fixture.performance.preview_id=creationTwo,
      fixture=>fixture.semantic.preview_id=creationTwo,
      fixture=>fixture.performance.audio_sha256="f".repeat(64),
      fixture=>fixture.semantic.audio_sha256="f".repeat(64),
      fixture=>fixture.performance.duration_ms=2000,
      fixture=>fixture.semantic.duration_ms=2000,
      fixture=>fixture.performance.cues[1].end_ms=1001,
      fixture=>fixture.semantic.windows[1].end_ms=1001
    ];
    for(const mutate of mutations) {
      const fixture=recordingFixture(projectAlias,{resolved:creationOne});
      mutate(fixture);
      await rejectedRecording(fixture);
    }
  });
  await test("Static project and creation links load exported JSON without live API requests", async () => {
    for(const id of [projectAlias,creationOne]) {
      const fixture=recordingFixture(id,{resolved:creationOne,staticSite:true});
      await fixture.recording.load(id,{library:true,staticSite:true});
      assert(fixture.calls.join("|")===[
        "/data/recordings/"+id+".json",fixture.assetBase+"performance.json",fixture.assetBase+"semantic.json"
      ].join("|"));
      assert(fixture.recording.metadata.preview_id===creationOne);
      assert(fixture.recording.semantic.projection.dimensions===5);
      assert(fixture.calls.every(path=>!path.startsWith("/api/")));
    }
  });
  await test("Static playback refuses private preview identities before making a request", async () => {
    for(const [id,library] of [["fixture-preview",false],["fixture-preview",true],[projectAlias,false]]) {
      const fixture=recordingFixture(id,{library,staticSite:true});
      await rejectedRecording(fixture);
      assert(fixture.calls.length===0);
    }
  });
  await test("Static aliases retain audio, semantic and concrete URL validation", async () => {
    const mutations=[
      fixture=>fixture.metadata.preview_id=projectAlias,
      fixture=>fixture.metadata.audio.url="https://other.example/audio.wav",
      fixture=>fixture.metadata.performance.url="/creation-library/"+projectAlias+"/performance.json",
      fixture=>fixture.metadata.semantic.url="/creation-library/"+projectAlias+"/semantic.json",
      fixture=>fixture.performance.audio_sha256="f".repeat(64),
      fixture=>fixture.semantic.preview_id=creationTwo,
      fixture=>fixture.semantic.duration_ms=2000
    ];
    for(const mutate of mutations) {
      const fixture=recordingFixture(projectAlias,{resolved:creationOne,staticSite:true});
      mutate(fixture);
      await rejectedRecording(fixture);
    }
  });
  const pass=tests.filter(Boolean).length;
  const summary=document.getElementById("summary");
  summary.textContent=pass+" / "+tests.length+" browser tests passed";
  summary.dataset.result=pass===tests.length?"pass":"fail";
})();
