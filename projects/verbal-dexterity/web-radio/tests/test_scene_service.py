"""Scene workflow fixtures use mock provider audio; never real credentials."""
import base64
import copy
import io
import json
import wave
from pathlib import Path

import httpx
import pytest
from web_radio import gemini
from web_radio.characters import CharacterStore
from web_radio.scene_service import SceneService,ScenePolicy


class Clock:
    def __init__(self):self.value=2_000_000_000.0;self.waits=[]
    def __call__(self):return self.value
    def sleep(self,seconds):self.waits.append(seconds);self.value+=seconds
    def advance(self,seconds):self.value+=seconds


def wav(level=1000):
    result=io.BytesIO()
    with wave.open(result,"wb") as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
        stream.writeframes(int(level).to_bytes(2,"little",signed=True)*2400)
    return result.getvalue()


def reply(request,level=1000,usage=True,extra=None):
    model=json.loads(request.content)["model"]
    data={"id":"fixture-interaction","status":"completed","model":model,
          "steps":[{"type":"model_output","content":[{"type":"audio","mime_type":"audio/wav",
                     "data":base64.b64encode(wav(level)).decode()}]}]}
    if usage:data["usage"]={"total_input_tokens":20,"total_output_tokens":3}
    if extra:data.update(extra)
    return httpx.Response(200,json=data)


def quota(request,kind="minute",delay="33s"):
    period="minute" if kind=="minute" else "day"
    number=3 if kind=="minute" else 10
    return httpx.Response(429,json={"error":{"code":"too_many_requests",
        "message":f"Rate limit exceeded for model {json.loads(request.content)['model']} (limit: {number} requests per {period} on Free Tier). Please retry in {delay} or upgrade your tier."}})


@pytest.fixture
def scene(tmp_path):
    store=CharacterStore(tmp_path)
    for identity,name,voice in (("alpha","Alpha","Charon"),("beta","Beta","Kore")):
        store.create({"character_id":identity,"name":name,"personality":"Original fictional fixture character"})
        store.add_prebuilt(identity,voice,"gemini-3.8-flash-tts")
    script={"programme_id":"scene-fixture","series_id":"scene-fixtures","kind":"programme","title":"ISOLATED FIXTURE",
            "language":"en","speakers":[{"name":"Alpha","character_id":"alpha"},{"name":"Beta","character_id":"beta"}],
            "turns":[{"speaker":"Alpha","text":"The rain is coming.","style":"warm"},
                     {"speaker":"Beta","text":"Then come inside.","style":"quiet"},
                     {"speaker":"Alpha","text":"Thank you.","style":"relieved"}]}
    clock=Clock()
    service=SceneService(tmp_path,key_provider=lambda:"fixture-not-a-real-key",clock=clock,sleep=clock.sleep)
    return service,script,clock


def test_small_scene_exports_sources_transcript_and_measured_stems(scene):
    service,script,clock=scene
    plan=service.plan(script,budget_usd="0.25")
    assert plan["request_count"]==3 and plan["budget_fits"] is True
    assert not plan["dispatched"] and not plan["reserved"]
    job=service.create(script,"0.25")
    calls=[]
    def perform(request):
        calls.append(json.loads(request.content));return reply(request,level=1000+len(calls)*200)
    result=service.run(job["id"],transport=httpx.MockTransport(perform))
    assert result["status"]=="ready",result.get("error")
    assert len(calls)==3 and result["counts"]=={"completed":3}
    assert all(call["generation_config"]["max_output_tokens"]==1024 for call in calls)
    receipt=result["outputs"]
    assert receipt["total_frames"]==14400 and receipt["duration_ms"]==300
    assert [(line["start_frame"],line["end_frame"]) for line in receipt["lines"]]==[(0,4800),(4800,9600),(9600,14400)]
    assert {line["character_id"] for line in receipt["lines"]}=={"alpha","beta"}
    assert {item["path"] for item in receipt["files"]}>={"scene.wav","scene.mp3","transcript.txt"}
    assert result["accounting"]["verified_invoice_microdollars"]==0
    assert result["budget"]["reserved_microdollars"]==0
    for line in result["lines"]:
        assert Path(line["source"]["source"]).read_bytes()==wav(1000+(line["line_index"]+1)*200)
    again=service.run(job["id"],transport=httpx.MockTransport(lambda r:pytest.fail("Ready scene repeated")))
    assert again["outputs"]==receipt


def test_paced_continuation_retains_byte_identical_completed_lines(scene):
    service,script,clock=scene
    policy={"quota_limits":{"gemini-3.8-flash-tts":{"requests_per_minute":3,"observed_at":"2026-10-23T00:00:00+04:00",
                                                  "evidence":"Observed only for this fixture credential/model."}},
            "max_wait_seconds":0.0}
    job=service.create(script,"0.25",policy)
    calls=[]
    def perform(request):calls.append(request);return reply(request,len(calls)*500)
    transport=httpx.MockTransport(perform)
    first=service.run(job["id"],transport=transport)
    assert first["status"]=="waiting_quota" and len(calls)==1
    original=Path(first["lines"][0]["source"]["source"]).read_bytes()
    clock.advance(22)
    second=service.run(job["id"],resume=True,transport=transport)
    assert second["status"]=="waiting_quota" and len(calls)==2
    clock.advance(22)
    third=service.run(job["id"],resume=True,transport=transport)
    assert third["status"]=="ready" and len(calls)==3
    assert Path(third["lines"][0]["source"]["source"]).read_bytes()==original


def test_minute_throttle_retries_within_policy_and_keeps_rejected_hold(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    calls=[]
    def perform(request):
        calls.append(request)
        return quota(request) if len(calls)==2 else reply(request)
    result=service.run(job["id"],transport=httpx.MockTransport(perform))
    assert result["status"]=="ready",result.get("error")
    assert len(calls)==4 and clock.waits[0]>=33 and clock.waits[1]>=22
    assert [item["state"] for item in result["lines"][1]["attempts"]]==["quota_rejected","completed"]
    assert result["budget"]["reserved_microdollars"]>0
    assert result["accounting"]["verified_invoice_microdollars"]==0


def test_daily_exhaustion_waits_without_repeated_calls(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    calls=[]
    def perform(request):calls.append(request);return quota(request,"daily","10h53m25s")
    result=service.run(job["id"],transport=httpx.MockTransport(perform))
    assert result["status"]=="waiting_daily" and len(calls)==1
    assert result["wait"]["retry_at_unix"]>clock()+36000
    result=service.run(job["id"],resume=True,transport=httpx.MockTransport(perform))
    assert result["status"]=="waiting_daily" and len(calls)==1


def test_declared_daily_model_fallback_records_actual_model_and_same_voice(scene):
    service,script,clock=scene
    policy={"models":list(("gemini-3.8-flash-tts","gemini-3.8-flash-lite-tts")),"fallback_on_daily":True}
    job=service.create(script,"0.25",policy)
    calls=[]
    def perform(request):
        calls.append(json.loads(request.content))
        return quota(request,"daily","10h") if len(calls)==1 else reply(request)
    result=service.run(job["id"],transport=httpx.MockTransport(perform))
    assert result["status"]=="ready"
    assert len(calls)==4 and all(item["model"]=="gemini-3.8-flash-lite-tts" for item in calls[1:])
    assert calls[0]["generation_config"]["speech_config"]==calls[1]["generation_config"]["speech_config"]
    assert all(line["source"]["actual_model"]=="gemini-3.8-flash-lite-tts" for line in result["lines"])
    assert result["lines"][0]["attempts"][0]["state"]=="quota_rejected"


def test_uncertain_dispatch_never_redispatches_or_creates_unreviewed_revision(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    calls=[]
    def timeout(request):calls.append(request);raise httpx.ReadTimeout("fixture",request=request)
    result=service.run(job["id"],transport=httpx.MockTransport(timeout))
    assert result["status"]=="uncertain" and len(calls)==1
    assert service.recover(job["id"])["status"]=="uncertain"
    assert service.run(job["id"],resume=True,transport=httpx.MockTransport(timeout))["status"]=="uncertain"
    with pytest.raises(ValueError,match="uncertain"):
        service.revise(job["id"],0,budget_usd="0.25")
    assert len(calls)==1 and result["budget"]["reserved_microdollars"]>0


def test_saved_response_recovers_without_key_or_provider(scene,monkeypatch):
    service,script,clock=scene
    job=service.create(script,"0.25")
    original=service._complete
    monkeypatch.setattr(service,"_complete",lambda *args:(_ for _ in ()).throw(OSError("fixture interruption")))
    first=service.run(job["id"],transport=httpx.MockTransport(reply))
    assert first["status"]=="uncertain"
    monkeypatch.setattr(service,"_complete",original)
    service.key_provider=lambda:pytest.fail("Local recovery read a credential")
    recovered=service.recover(job["id"])
    assert recovered["lines"][0]["state"]=="completed"
    assert recovered["status"]=="queued"
    service.key_provider=lambda:"fixture-not-a-real-key"
    calls=[]
    def perform(request):calls.append(request);return reply(request)
    final=service.run(job["id"],resume=True,transport=httpx.MockTransport(perform))
    assert final["status"]=="ready" and len(calls)==2


def test_selected_line_revision_and_alternative_recipe_preserve_prior_takes(scene):
    service,script,clock=scene
    first=service.create(script,"0.25")
    ready=service.run(first["id"],transport=httpx.MockTransport(reply))
    before=[line["source"]["source_sha256"] for line in ready["lines"]]
    alternative=service.characters.add_prebuilt("beta","Puck","gemini-3.8-flash-tts")
    revision=service.revise(first["id"],1,text="Stay by the fire.",style="a warm invitation",
                            recipe_id=alternative["recipe_id"],budget_usd="0.25")
    assert revision["parent_job"]==first["id"] and revision["counts"]=={"completed":2,"pending":1}
    calls=[]
    def perform(request):calls.append(json.loads(request.content));return reply(request,2200)
    final=service.run(revision["id"],transport=httpx.MockTransport(perform))
    assert final["status"]=="ready",final.get("error")
    assert len(calls)==1 and calls[0]["generation_config"]["speech_config"][0]["voice"]=="Puck"
    assert final["lines"][0]["source"]["source_sha256"]==before[0]
    assert final["lines"][2]["source"]["source_sha256"]==before[2]
    assert service.status(first["id"])["script"]==ready["script"]
    assert final["lines"][1]["data"]["voice_recipe"]["origin"]=="prebuilt"


def test_output_ceiling_needs_review_instead_of_silent_partial_export(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    def limit(request):
        base=json.loads(reply(request).content);base["usage"]["total_output_tokens"]=1024
        return httpx.Response(200,json=base)
    result=service.run(job["id"],transport=httpx.MockTransport(limit))
    assert result["status"]=="review_required" and result["outputs"] is None
    assert result["lines"][0]["source"]["source_attempt"]["output_limit_reached"] is True
    assert Path(result["lines"][0]["source"]["source"]).is_file()


def test_explicit_finish_reason_without_usage_is_not_assembled(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    result=service.run(job["id"],transport=httpx.MockTransport(
        lambda request:reply(request,usage=False,extra={"finish_reason":"MAX_TOKENS"})))
    assert result["status"]=="review_required" and result["outputs"] is None


def test_budget_shortfall_rolls_back_all_scene_attempts(scene):
    from web_radio.budget import BudgetError
    service,script,clock=scene
    with pytest.raises(BudgetError):service.create(script,"0.001")
    with service.budget.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM scene_jobs").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM scene_attempts").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0]==0


def test_reconciliation_keeps_evidence_and_never_assumes_missing_audio(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    def timeout(request):raise httpx.ReadTimeout("fixture",request=request)
    result=service.run(job["id"],transport=httpx.MockTransport(timeout))
    attempt=result["lines"][0]["attempts"][0]["id"]
    service.reconcile(job["id"],attempt,"0","Verified fixture invoice shows no billed charge.")
    current=service.status(job["id"])
    assert current["lines"][0]["state"]=="blocked"
    assert current["lines"][0]["source"] is None
    assert current["accounting"]["verified_invoice_microdollars"]==0
    new=service.revise(job["id"],0,budget_usd="0.25")
    assert new["status"]=="queued"


def test_no_provider_model_substitution_is_accepted(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    result=service.run(job["id"],transport=httpx.MockTransport(
        lambda request:reply(request,extra={"model":"models/gemini-3.8-flash-lite-tts"})))
    assert result["status"]=="uncertain" and result["outputs"] is None
    assert result["lines"][0]["attempts"][0]["state"]=="uncertain"


def test_actual_cli_plan_create_and_show_are_keyless(scene,capsys):
    from web_radio.scene_cli import main
    service,script,clock=scene
    path=service.root/"scene.json";path.write_text(json.dumps(script))
    prefix=["--root",str(service.root)]
    assert main(prefix+["plan","--file",str(path),"--budget","0.25"])==0
    assert json.loads(capsys.readouterr().out)["result"]["request_count"]==3
    assert main(prefix+["create","--file",str(path),"--budget","0.25"])==0
    job=json.loads(capsys.readouterr().out)["result"]
    assert main(prefix+["show","--job",job["id"]])==0
    assert json.loads(capsys.readouterr().out)["result"]["status"]=="queued"


def legacy_parts(scene):
    from web_radio.studio import Studio,BudgetSettings,GenerationRequest
    service,script,clock=scene
    script=copy.deepcopy(script)
    service.characters.create({"character_id":"gamma","name":"Gamma","personality":"Fictional fixture guest."})
    service.characters.add_prebuilt("gamma","Puck","gemini-3.8-flash-tts")
    script["speakers"].append({"name":"Gamma","character_id":"gamma"})
    script["turns"][2]["speaker"]="Gamma"
    content=service.root/"content/series/first-light";content.mkdir(parents=True)
    for kind in ("programme","ident"):
        data=copy.deepcopy(script);data["kind"]=kind
        (content/(kind+".json")).write_text(json.dumps(data))
    studio=Studio(service.root);studio.key=lambda:"fixture-not-real"
    studio.configure(BudgetSettings(episode_limit_usd="10",series_limit_usd="10"))
    revision=studio.save_script(gemini.Script.model_validate(script))
    first=studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts",script_revision_id=revision["id"]))
    calls=[]
    def partial(request):
        calls.append(request);return reply(request) if len(calls)==1 else quota(request)
    studio.run_one(httpx.MockTransport(partial),take_id=first["id"])
    continuation=copy.deepcopy(script);continuation["turns"]=continuation["turns"][1:]
    revision=studio.save_script(gemini.Script.model_validate(continuation))
    second=studio.enqueue(GenerationRequest(model="gemini-3.8-flash-lite-tts",script_revision_id=revision["id"]))
    studio.run_one(httpx.MockTransport(reply),take_id=second["id"])
    mapping=[{"turn_index":0,"take_id":first["id"],"request_index":0},
             {"turn_index":1,"take_id":second["id"],"request_index":0},
             {"turn_index":2,"take_id":second["id"],"request_index":1}]
    return service,script,mapping,studio


def test_completed_partial_takes_compose_truthfully_without_provider(scene):
    service,script,mapping,studio=legacy_parts(scene)
    before={entry["take_id"]:studio.take(entry["take_id"]) for entry in mapping}
    service.key_provider=lambda:pytest.fail("Source import read a credential")
    result=service.import_sources(script,mapping,"0.25")
    assert result["status"]=="ready" and result["counts"]=={"completed":3}
    assert result["budget"]["spent_microdollars"]==0 and result["budget"]["reserved_microdollars"]==0
    assert [line["source"]["actual_model"] for line in result["lines"]]==[
        "gemini-3.8-flash-tts","gemini-3.8-flash-lite-tts","gemini-3.8-flash-lite-tts"]
    assert result["lines"][0]["source"]["source_take"]==mapping[0]["take_id"]
    for take_id,value in before.items():assert studio.take(take_id)==value
    assert all(line["source"]["source_attempt"]["accounting"].startswith("Prior legacy ledger") for line in result["lines"])


def test_adoption_rejects_a_valid_wav_swapped_after_completed_legacy_step(scene):
    service,script,mapping,studio=legacy_parts(scene)
    first=mapping[0]
    path=studio.state/"takes"/first["take_id"]/"requests/0000/raw.wav"
    path.write_bytes(wav(2200))
    with pytest.raises(ValueError,match="authoritative"):
        service.import_sources(script,mapping,"0.25")


def test_source_receipt_cannot_silently_change_cast_or_text(scene):
    service,script,mapping,studio=legacy_parts(scene)
    mapping[0]["text"]="Words that were never recorded."
    with pytest.raises(ValueError,match="receipt"):
        service.import_sources(script,mapping,"0.25")


def test_scene_lease_blocks_backup_without_touching_recordings(scene):
    from web_radio.backup import backup_runtime,BackupError
    service,script,clock=scene
    job=service.create(script,"0.25")
    with service._worker():
        with pytest.raises(BackupError):
            backup_runtime(service.root/".runtime",service.root/"blocked.tar.gz")
    assert not (service.root/"blocked.tar.gz").exists()


def test_accounting_separates_planned_reservations_from_rejected_holds(scene):
    service,script,clock=scene
    queued=service.create(script,"0.25")
    assert queued["accounting"]["conservative_planned_reservations_microdollars"]>0
    assert queued["accounting"]["unresolved_held_microdollars"]==0
    def rejected(request):return httpx.Response(400,json={"error":{"code":"invalid_argument","message":"The supplied field was rejected."}})
    blocked=service.run(queued["id"],transport=httpx.MockTransport(rejected))
    assert blocked["status"]=="blocked"
    assert blocked["accounting"]["unresolved_held_microdollars"]>0
    assert blocked["accounting"]["conservative_planned_reservations_microdollars"]>0
    assert blocked["accounting"]["verified_invoice_microdollars"]==0
    assert blocked["lines"][0]["attempts"][0]["diagnostic"]["code"]=="invalid_argument"


def test_ready_scene_has_concise_playable_summary(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    ready=service.run(job["id"],transport=httpx.MockTransport(reply))
    descriptor=ready["outputs"]["scene_summary"]
    summary=json.loads((service.root/descriptor["path"]).read_bytes())
    assert summary["line_count"]==3 and summary["new_dispatches"]==3
    assert summary["publication"]=="Not published or scheduled."
    assert (service.root/summary["playable_mp3"]).is_file()
    assert service.recover(job["id"])["outputs"]["scene_summary"]==descriptor


def test_forecast_uses_only_known_same_credential_observations_without_probe(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    calls=[]
    def perform(request):
        calls.append(request);return quota(request) if len(calls)==1 else reply(request)
    assert service.run(job["id"],transport=httpx.MockTransport(perform))["status"]=="ready"
    before=len(calls)
    assert service.plan(script)["forecast"]["known_requests_per_minute"]==3
    assert len(calls)==before
    service.key_provider=lambda:"a-different-fixture-credential"
    assert service.plan(script)["forecast"]["known_requests_per_minute"] is None


def test_unknown_quota_without_retry_hint_pauses_instead_of_hammering(scene):
    service,script,clock=scene
    job=service.create(script,"0.25")
    calls=[]
    def perform(request):
        calls.append(request);return httpx.Response(429,json={"error":{"message":"Quota temporarily unavailable."}})
    result=service.run(job["id"],transport=httpx.MockTransport(perform))
    assert result["status"]=="waiting_quota" and len(calls)==1
    assert result["wait"]["reason"]=="quota_without_retry_hint"


def test_explicit_revision_policy_can_raise_output_cap_for_only_new_line(scene):
    service,script,clock=scene
    first=service.create(script,"0.25")
    ready=service.run(first["id"],transport=httpx.MockTransport(reply))
    policy=copy.deepcopy(ready["policy"]);policy["max_output_tokens"]=2048
    new=service.revise(first["id"],1,budget_usd="0.25",policy=policy)
    calls=[]
    def perform(request):calls.append(json.loads(request.content));return reply(request)
    result=service.run(new["id"],transport=httpx.MockTransport(perform))
    assert result["status"]=="ready" and len(calls)==1
    assert calls[0]["generation_config"]["max_output_tokens"]==2048
    assert result["lines"][0]["source"]==ready["lines"][0]["source"]

def test_partial_revision_policy_preserves_parent_models_and_pacing(scene):
    service,script,clock=scene
    policy={"models":["gemini-3.8-flash-lite-tts"],"min_interval_seconds":22.0}
    first=service.create(script,"0.25",policy)
    ready=service.run(first["id"],transport=httpx.MockTransport(reply))
    assert ready["status"]=="ready"
    new=service.revise(first["id"],1,budget_usd="0.25",policy={"max_output_tokens":2048})
    assert new["policy"]["models"]==policy["models"]
    assert new["policy"]["min_interval_seconds"]==22.0
    assert new["policy"]["max_output_tokens"]==2048
    calls=[]
    def perform(request):calls.append(json.loads(request.content));return reply(request)
    result=service.run(new["id"],transport=httpx.MockTransport(perform))
    assert result["status"]=="ready" and len(calls)==1
    assert calls[0]["model"]=="gemini-3.8-flash-lite-tts"
    assert result["lines"][0]["source"]==ready["lines"][0]["source"]

def test_revision_refuses_active_worker_between_dispatches(scene):
    service,script,clock=scene
    parent=service.create(script,"0.25")
    with service._worker():
        with pytest.raises(ValueError,match="busy|progress|active|lock|running"):
            service.revise(parent["id"],1,budget_usd="0.25")
    with service.budget.connect() as db:
        assert db.execute("SELECT count(*) FROM scene_jobs").fetchone()[0]==1
