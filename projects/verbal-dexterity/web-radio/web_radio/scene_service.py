"""Selected scene jobs with retained lines, explicit continuation and revision."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from . import gemini
from .budget import Budget, BudgetError, dollars_to_microdollars, usage_cost
from .characters import CharacterStore
from .scene_storage import encode, digest, ordinary, read, write, lease

MODELS=("gemini-3.8-flash-tts","gemini-3.8-flash-lite-tts")
MAX_JSON=8*1024*1024
IDENTIFIER=r"^[a-f0-9]{32}$"


def now():
    return datetime.now(timezone.utc).isoformat()


class QuotaLimit(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True,allow_inf_nan=False)
    requests_per_minute:int|None=Field(default=None,ge=1,le=1000000)
    requests_per_day:int|None=Field(default=None,ge=1,le=10000000)
    observed_at:str=Field(min_length=10,max_length=64)
    evidence:str=Field(min_length=12,max_length=2000)

    @model_validator(mode="after")
    def timestamp(self):
        stamp=datetime.fromisoformat(self.observed_at.replace("Z","+00:00"))
        if stamp.tzinfo is None:raise ValueError("Quota evidence needs a timezone-aware timestamp.")
        if self.requests_per_day is None and self.requests_per_minute is None:
            raise ValueError("Record at least one observed quota limit.")
        return self


class ScenePolicy(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True,allow_inf_nan=False)
    schema_version:Literal[1]=1
    models:list[Literal["gemini-3.8-flash-tts","gemini-3.8-flash-lite-tts"]]=Field(default_factory=lambda:[MODELS[0]],min_length=1,max_length=2)
    fallback_on_daily:bool=False
    max_output_tokens:int=Field(default=1024,ge=128,le=16384)
    max_quota_retries:int=Field(default=2,ge=0,le=10)
    max_wait_seconds:float=Field(default=60.0,ge=0,le=60)
    min_interval_seconds:float=Field(default=0.0,ge=0,le=3600)
    quota_limits:dict[str,QuotaLimit]=Field(default_factory=dict,max_length=2)
    stems:bool=True

    @model_validator(mode="before")
    @classmethod
    def literal_version(cls,value):
        if isinstance(value,dict) and "schema_version" in value and type(value["schema_version"]) is not int:
            raise ValueError("Scene policy version must be integer one.")
        return value

    @model_validator(mode="after")
    def valid_policy(self):
        if type(self.schema_version) is not int or len(set(self.models))!=len(self.models):
            raise ValueError("Use an integer policy version and unique ordered models.")
        if self.fallback_on_daily and len(self.models)<2:
            raise ValueError("Daily fallback needs a declared second model.")
        if any(model not in self.models for model in self.quota_limits):
            raise ValueError("Quota evidence must name an allowed model.")
        return self


class SceneService:
    def __init__(self,root:Path,*,key_provider:Callable|None=None,clock:Callable|None=None,sleep:Callable|None=None):
        self.root=ordinary(Path(os.path.abspath(root)))
        if not self.root.is_dir():raise ValueError("An existing project directory is required.")
        self.state=self.root/".runtime/studio/scenes"
        self.database=self.root/".runtime/studio/studio.sqlite3"
        self.clock=clock or time.time;self.sleep=sleep or time.sleep
        self.key_provider=key_provider or self._stored_key
        with self._operation():
            self.state.mkdir(parents=True,exist_ok=True)
            self.budget=Budget(self.database)
            self.characters=CharacterStore(self.root)
            with self.budget.connect() as db:
                db.executescript("""
                CREATE TABLE IF NOT EXISTS scene_jobs(
                id TEXT PRIMARY KEY,parent_job TEXT,status TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
                script_json TEXT NOT NULL,cast_json TEXT NOT NULL,policy_json TEXT NOT NULL,total_limit_micro INTEGER NOT NULL,
                wait_json TEXT,outputs_json TEXT,error TEXT);
                CREATE TABLE IF NOT EXISTS scene_lines(
                job_id TEXT NOT NULL,line_index INTEGER NOT NULL,state TEXT NOT NULL,data_json TEXT NOT NULL,
                source_json TEXT,PRIMARY KEY(job_id,line_index));
                CREATE TABLE IF NOT EXISTS scene_attempts(
                id TEXT PRIMARY KEY,job_id TEXT NOT NULL,line_index INTEGER NOT NULL,ordinal INTEGER NOT NULL,
                model TEXT NOT NULL,state TEXT NOT NULL,request_json TEXT NOT NULL,request_sha256 TEXT NOT NULL,
                allocation_micro INTEGER NOT NULL,response_sha256 TEXT,audio_sha256 TEXT,metadata_json TEXT,
                diagnostic_json TEXT,scope_id TEXT,dispatched_at REAL,charged_micro INTEGER,accounting TEXT);
                CREATE TABLE IF NOT EXISTS scene_credential_scopes(
                fingerprint TEXT PRIMARY KEY,id TEXT NOT NULL UNIQUE);
                CREATE TABLE IF NOT EXISTS scene_quotas(
                scope_id TEXT NOT NULL,model TEXT NOT NULL,rpm INTEGER,rpd INTEGER,next_at REAL NOT NULL DEFAULT 0,
                daily_exhausted INTEGER NOT NULL DEFAULT 0,daily_until REAL,evidence_json TEXT,
                PRIMARY KEY(scope_id,model));
                CREATE TABLE IF NOT EXISTS scene_reconciliations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,attempt_id TEXT NOT NULL,charged_micro INTEGER NOT NULL,
                evidence TEXT NOT NULL,recorded_at TEXT NOT NULL);
                """)

    @contextmanager
    def _operation(self):
        ordinary(self.state)
        with lease(self.root/".runtime/studio/generation-worker.lock"):
            yield

    @contextmanager
    def _worker(self):
        with self._operation(),lease(self.state/"worker.lock",exclusive=True):
            yield

    def _directory(self,job_id):
        if not isinstance(job_id,str) or not re.fullmatch(IDENTIFIER,job_id):raise ValueError("Unknown scene job.")
        return ordinary(self.state/job_id)

    def _stored_key(self):
        path=ordinary(self.root/".runtime/secrets/gemini-api-key")
        return (read(path,4096).decode().strip() if path.exists() else os.environ.get("GEMINI_API_KEY","").strip()) or None

    def _scope(self,key):
        # A private salted HMAC identifies credential changes without publishing a
        # key digest. Neither fingerprint nor scope ID is returned by status().
        salt_path=self.state/"credential-salt.bin"
        if not salt_path.exists():write(salt_path,os.urandom(32))
        fingerprint=hmac.new(read(salt_path,32),key.encode(),hashlib.sha256).hexdigest()
        with self.budget.transaction() as db:
            row=db.execute("SELECT id FROM scene_credential_scopes WHERE fingerprint=?",(fingerprint,)).fetchone()
            if row:return row[0]
            identifier=uuid4().hex
            db.execute("INSERT INTO scene_credential_scopes VALUES(?,?)",(fingerprint,identifier))
            return identifier

    def _prepare(self,script,policy=None,overrides=None,reuse=None,frozen_cast=None):
        from .production_pricing import request_allocation,expected_estimate
        script=gemini.Script.model_validate(script.model_dump() if isinstance(script,gemini.Script) else script)
        policy=ScenePolicy.model_validate(policy.model_dump() if isinstance(policy,ScenePolicy) else (policy or {}))
        bound,cast=(script,frozen_cast) if frozen_cast is not None else self.characters.bind_cast(script)
        if any(not speaker.character_id for speaker in bound.speakers):
            raise ValueError("Scene speakers need reusable character identities; use character load/prebuilt/design first.")
        snapshots={item["speaker"]:item for item in cast["speakers"]}
        speakers={item.name:item for item in bound.speakers}
        lines=[]
        for index,turn in enumerate(bound.turns):
            speaker=speakers[turn.speaker]
            snapshot=snapshots[turn.speaker]
            if overrides and index in overrides:
                variant=overrides[index]
                speaker=gemini.SpeakerRecipe.model_validate(variant["speaker"])
                snapshot={"character":variant["character"],"voice_recipe":variant["voice_recipe"],"speaker":speaker.name}
            unary=bound.model_copy(update={"speakers":[speaker],"turns":[turn]},deep=True)
            if reuse and index in reuse:
                lines.append({"index":index,"turn":turn.model_dump(),"speaker":speaker.model_dump(),
                              "character":snapshot["character"],"voice_recipe":snapshot["voice_recipe"],
                              "requests":{},"allocations":{},"expected_estimate":None})
                continue
            verified,_=self.characters.bind_cast(unary)
            if verified.model_dump()!=unary.model_dump():
                raise ValueError("The effective unfinished line differs from its frozen recipe.")
            requests={}
            allocations={}
            for model in policy.models:
                request=gemini.build_request(unary,model,approved_designed_voices=self.characters.approved_designed_voices(),
                                             max_output_tokens=policy.max_output_tokens)
                requests[model]=request
                allocations[model]=request_allocation(request)
            estimate_seconds=max(0.4,len(turn.text.split())/2.5+0.2)
            estimate=expected_estimate(policy.models[0],input_tokens=max(1,math.ceil(len(encode(requests[policy.models[0]]))/4)),
                                       audio_seconds=estimate_seconds)
            lines.append({"index":index,"turn":turn.model_dump(),"speaker":speaker.model_dump(),
                          "character":snapshot["character"],"voice_recipe":snapshot["voice_recipe"],
                          "requests":requests,"allocations":allocations,"expected_estimate":estimate})
        return bound,cast,policy,lines

    def _forecast(self,policy,count):
        model=policy.models[0];observed=policy.quota_limits.get(model)
        rpm=observed.requests_per_minute if observed else None
        rpd=observed.requests_per_day if observed else None
        retained=None
        salt_path=self.state/"credential-salt.bin"
        if salt_path.exists():
            key=self.key_provider()
            if key:
                fingerprint=hmac.new(read(salt_path,32),key.encode(),hashlib.sha256).hexdigest()
                with self.budget.connect() as db:
                    scope=db.execute("SELECT id FROM scene_credential_scopes WHERE fingerprint=?",(fingerprint,)).fetchone()
                    row=db.execute("SELECT * FROM scene_quotas WHERE scope_id=? AND model=?",(scope[0],model)).fetchone() if scope else None
                if row:
                    retained=dict(row)
                    if row["rpm"] is not None:rpm=min(rpm,row["rpm"]) if rpm else row["rpm"]
                    if row["rpd"] is not None:rpd=min(rpd,row["rpd"]) if rpd else row["rpd"]
                key=None
        interval=max(policy.min_interval_seconds,60/rpm+2 if rpm else 0)
        return {"model":model,"requests":count,"known_requests_per_minute":rpm,"known_requests_per_day":rpd,
                "known_daily_exhaustion":bool(retained and retained["daily_exhausted"] and
                    (retained["daily_until"] is None or retained["daily_until"]>self.clock())),
                "observed_retry_at_unix":retained["daily_until"] if retained and retained["daily_exhausted"] else None,
                "likely_minimum_pacing_seconds":round(max(0,count-1)*interval,3) if interval else None,
                "may_exceed_one_day_quota":count>rpd if rpd else None,
                "basis":"Authored key/model observations only; external usage and actual quota resets are unknown.",
                "evidence":observed.model_dump() if observed else None}

    def plan(self,script,policy=None,budget_usd=None):
        with self._operation():
            bound,cast,policy,lines=self._prepare(script,policy)
            reserved=sum(max(item["reservation_microdollars"] for item in line["allocations"].values()) for line in lines)
            limit=dollars_to_microdollars(budget_usd) if budget_usd is not None else None
            return {"title":bound.title,"lines":len(lines),"cast":cast,"policy":policy.model_dump(),
                    "request_count":len(lines),"reservation_microdollars":reserved,
                    "reservation_usd":f"{reserved/1_000_000:.6f}","budget_fits":None if limit is None else reserved<=limit,
                    "expected_estimates":[line["expected_estimate"] for line in lines],
                    "forecast":self._forecast(policy,len(lines)),"dispatched":False,"reserved":False}

    def create(self,script,budget_usd,policy=None):
        with self._operation():return self._create(script,budget_usd,policy)

    def _create(self,script,budget_usd,policy=None,*,parent=None,overrides=None,reuse=None,frozen_cast=None):
        bound,cast,policy,lines=self._prepare(script,policy,overrides,reuse,frozen_cast)
        reuse=reuse or {}
        identifier=uuid4().hex;directory=self._directory(identifier);limit=dollars_to_microdollars(budget_usd)
        scope="scene:"+identifier
        self.budget.configure(scope,scope,limit,limit)
        stamp=now()
        with self.budget.transaction() as db:
            for line in lines:
                index=line["index"];source=reuse.get(index)
                if source:
                    self._verify_source(source)
                state="completed" if source else "pending"
                db.execute("INSERT INTO scene_lines VALUES(?,?,?,?,?)",
                           (identifier,index,state,encode(line).decode(),encode(source).decode() if source else None))
                if not source:
                    model=policy.models[0];request=line["requests"][model]
                    amount=max(item["reservation_microdollars"] for item in line["allocations"].values())
                    attempt=uuid4().hex
                    self.budget.reserve(attempt,scope,scope,amount,connection=db)
                    db.execute("""INSERT INTO scene_attempts
                    (id,job_id,line_index,ordinal,model,state,request_json,request_sha256,allocation_micro)
                    VALUES(?,?,?,0,?,'pending',?,?,?)""",
                    (attempt,identifier,index,model,encode(request).decode(),digest(encode(request)),amount))
            db.execute("""INSERT INTO scene_jobs
            (id,parent_job,status,created_at,updated_at,script_json,cast_json,policy_json,total_limit_micro)
            VALUES(?,?,'queued',?,?,?,?,?,?)""",
            (identifier,parent,stamp,stamp,encode(bound.model_dump()).decode(),encode(cast).decode(),encode(policy.model_dump()).decode(),limit))
            write(directory/"script.json",encode(bound.model_dump()))
            write(directory/"cast.json",encode(cast));write(directory/"policy.json",encode(policy.model_dump()))
        return self.status(identifier)

    def status(self,job_id):
        self._directory(job_id)
        with self._operation(),self.budget.connect() as db:
            row=db.execute("SELECT * FROM scene_jobs WHERE id=?",(job_id,)).fetchone()
            if row is None:raise ValueError("Unknown scene job.")
            result=dict(row)
            for name in ("script","cast","policy","wait","outputs"):
                raw=result.pop(name+"_json");result[name]=json.loads(raw) if raw else None
            result["lines"]=[]
            for item in db.execute("SELECT * FROM scene_lines WHERE job_id=? ORDER BY line_index",(job_id,)):
                item=dict(item);item["data"]=json.loads(item.pop("data_json"));raw=item.pop("source_json")
                item["source"]=json.loads(raw) if raw else None
                attempts=[]
                for attempt in db.execute("""SELECT id,ordinal,model,state,request_sha256,response_sha256,audio_sha256,
                    diagnostic_json,charged_micro,accounting FROM scene_attempts
                    WHERE job_id=? AND line_index=? ORDER BY ordinal""",(job_id,item["line_index"])):
                    value=dict(attempt);raw=value.pop("diagnostic_json")
                    value["diagnostic"]=json.loads(raw) if raw else None;attempts.append(value)
                item["attempts"]=attempts;result["lines"].append(item)
            result["counts"]={state:sum(line["state"]==state for line in result["lines"])
                              for state in sorted({line["state"] for line in result["lines"]})}
            result["budget"]=self.budget.summary("scene:"+job_id,"scene:"+job_id)["episode"]
            verified=db.execute("""SELECT COALESCE(SUM(charged_micro),0) FROM (
                SELECT a.id,r.charged_micro FROM scene_attempts a JOIN reservations r ON r.attempt_id=a.id
                WHERE a.job_id=? AND r.accounting='verified-scene-billing')""",(job_id,)).fetchone()[0]
            held={row["state"]:row["amount"] for row in db.execute(
                """SELECT a.state,COALESCE(SUM(r.reserved_micro),0) amount FROM scene_attempts a
                   JOIN reservations r ON r.attempt_id=a.id WHERE a.job_id=? AND r.state='reserved'
                   GROUP BY a.state""",(job_id,))}
            result["accounting"]={"verified_invoice_microdollars":verified,
                "token_estimates_or_allocations_consumed_microdollars":result["budget"]["spent_microdollars"]-verified,
                "conservative_planned_reservations_microdollars":held.get("pending",0),
                "in_flight_reserved_microdollars":held.get("running",0),
                "unresolved_held_microdollars":sum(value for state,value in held.items() if state not in ("pending","running")),
                "spent_field_includes_token_estimates_or_conservative_allocations":True}
            auditions={}
            for line in result["lines"]:
                data=line["data"];recipe=data["voice_recipe"]
                auditions[(data["character"]["character_id"],recipe["recipe_id"])]={
                    "character_id":data["character"]["character_id"],"recipe_id":recipe["recipe_id"],
                    "origin":recipe["origin"],"audition_path":recipe.get("audition_path")}
            result["auditions"]=list(auditions.values())
            prior_takes={line["source"].get("source_take") for line in result["lines"] if line["source"] and line["source"].get("source_take")}
            result["legacy_source_accounting"]=[]
            for take_id in sorted(prior_takes):
                values=db.execute("""SELECT COALESCE(SUM(CASE WHEN r.state='reserved' THEN r.reserved_micro ELSE 0 END),0) held,
                    COALESCE(SUM(CASE WHEN r.state='spent' THEN r.charged_micro ELSE 0 END),0) estimated_or_verified
                    FROM reservations r JOIN generation_requests g ON g.attempt_id=r.attempt_id WHERE g.take_id=?""",(take_id,)).fetchone()
                result["legacy_source_accounting"].append({"take_id":take_id,"unresolved_held_microdollars":values["held"],
                    "prior_estimates_or_charges_microdollars":values["estimated_or_verified"],
                    "notice":"Prior ledger only; no invoice inferred, no holds released, excluded from this new job budget."})
            return result

    def _verify_source(self,source):
        data=read(Path(source["source"]),gemini.MAX_AUDIO_BYTES)
        if digest(data)!=source["source_sha256"]:raise ValueError("Completed line audio failed its retained hash.")
        gemini.parse_audio_block({"mime_type":"audio/wav","data":__import__("base64").b64encode(data).decode()})
        return data

    def _set_job(self,job_id,state,*,wait=None,error=None):
        with self.budget.connect() as db:
            db.execute("UPDATE scene_jobs SET status=?,wait_json=?,error=?,updated_at=? WHERE id=?",
                       (state,encode(wait).decode() if wait else None,error,now(),job_id))

    def _attempts(self,job_id,index):
        with self.budget.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM scene_attempts WHERE job_id=? AND line_index=? ORDER BY ordinal",
                                                    (job_id,index))]

    def _quota(self,scope,model,policy):
        authored=policy.quota_limits.get(model)
        data={"rpm":authored.requests_per_minute if authored else None,
              "rpd":authored.requests_per_day if authored else None,"next_at":0.0,
              "daily_exhausted":False,"daily_until":None}
        with self.budget.connect() as db:
            row=db.execute("SELECT * FROM scene_quotas WHERE scope_id=? AND model=?",(scope,model)).fetchone()
        if row:
            for key in ("rpm","rpd","next_at","daily_until"):
                if row[key] is not None:
                    data[key]=min(data[key],row[key]) if key in ("rpm","rpd") and data[key] is not None else row[key]
            data["daily_exhausted"]=bool(row["daily_exhausted"])
        if data["daily_exhausted"] and data["daily_until"] is not None and self.clock()>=data["daily_until"]:
            with self.budget.connect() as db:
                db.execute("UPDATE scene_quotas SET daily_exhausted=0,daily_until=NULL WHERE scope_id=? AND model=?",(scope,model))
            data["daily_exhausted"]=False;data["daily_until"]=None
        return data

    def _pace_after_response(self,scope,model,policy,diagnostic=None):
        from .provider_diagnostics import retry_after_seconds,quota_kind,next_daily_reset
        value=self._quota(scope,model,policy)
        for observation in (diagnostic or {}).get("quota_observations",[]):
            if observation.get("unit")!="requests" or type(observation.get("limit")) is not int or observation["limit"]<=0:
                continue
            dimensions=observation.get("dimensions",{})
            named=dimensions.get("model")
            if named and named not in (model,"models/"+model):continue
            if observation.get("kind")=="minute":value["rpm"]=min(value["rpm"],observation["limit"]) if value["rpm"] else observation["limit"]
            if observation.get("kind")=="daily":value["rpd"]=min(value["rpd"],observation["limit"]) if value["rpd"] else observation["limit"]
        interval=max(policy.min_interval_seconds,60/value["rpm"]+2 if value["rpm"] else 0)
        value["next_at"]=self.clock()+interval
        delay=retry_after_seconds(diagnostic or {})
        kind=quota_kind(diagnostic or {})
        if delay is not None:value["next_at"]=max(value["next_at"],self.clock()+delay)
        if kind=="daily":
            value["daily_exhausted"]=True
            value["daily_until"]=self.clock()+delay if delay is not None else next_daily_reset(self.clock())
        with self.budget.connect() as db:
            db.execute("""INSERT INTO scene_quotas(scope_id,model,rpm,rpd,next_at,daily_exhausted,daily_until,evidence_json)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(scope_id,model) DO UPDATE SET
                rpm=excluded.rpm,rpd=excluded.rpd,next_at=excluded.next_at,daily_exhausted=excluded.daily_exhausted,
                daily_until=excluded.daily_until,evidence_json=COALESCE(excluded.evidence_json,scene_quotas.evidence_json)""",
                (scope,model,value["rpm"],value["rpd"],value["next_at"],int(value["daily_exhausted"]),value["daily_until"],
                 encode(diagnostic).decode() if diagnostic else None))
        return value

    def _choose_model(self,scope,policy):
        first=policy.models[0]
        quota=self._quota(scope,first,policy)
        if not quota["daily_exhausted"]:return first,quota
        if policy.fallback_on_daily:
            for model in policy.models[1:]:
                alternative=self._quota(scope,model,policy)
                if not alternative["daily_exhausted"]:return model,alternative
        return None,quota

    def _new_attempt(self,job,line,model):
        from .production_pricing import request_allocation
        data=line["data"];payload=data["requests"][model]
        amount=request_allocation(payload)["reservation_microdollars"]
        attempts=self._attempts(job["id"],line["line_index"])
        identifier=uuid4().hex
        with self.budget.transaction() as db:
            self.budget.reserve(identifier,"scene:"+job["id"],"scene:"+job["id"],amount,connection=db)
            db.execute("""INSERT INTO scene_attempts
            (id,job_id,line_index,ordinal,model,state,request_json,request_sha256,allocation_micro)
            VALUES(?,?,?,?,?,'pending',?,?,?)""",
            (identifier,job["id"],line["line_index"],len(attempts),model,encode(payload).decode(),digest(encode(payload)),amount))
        return self._attempts(job["id"],line["line_index"])[-1]

    def _complete(self,job,line,attempt):
        from .production_pricing import provider_usage_estimate
        directory=self._directory(job["id"])/"attempts"/attempt["id"]
        audio=read(directory/"raw.wav",gemini.MAX_AUDIO_BYTES)
        response=read(directory/"response.json",MAX_JSON)
        if digest(audio)!=attempt["audio_sha256"] or digest(response)!=attempt["response_sha256"]:
            raise ValueError("Retained scene response/audio failed authoritative verification.")
        gemini.parse_audio_block({"mime_type":"audio/wav","data":__import__("base64").b64encode(audio).decode()})
        metadata=json.loads(response)
        output_limit=json.loads(attempt["request_json"])["generation_config"]["max_output_tokens"]
        usage=metadata.get("usage") or {}
        at_limit=type(usage.get("total_output_tokens")) is int and usage["total_output_tokens"]>=output_limit
        if metadata.get("finish_reason") in ("MAX_TOKENS","max_output_tokens","MAX_OUTPUT_TOKENS"):at_limit=True
        if metadata.get("model") not in (None,attempt["model"],"models/"+attempt["model"]):
            raise ValueError("Provider returned an unexpected model; no cast/model substitution is accepted.")
        try:estimated=provider_usage_estimate(attempt["model"],usage)
        except BudgetError:estimated=None  # Local recovery never requires fresh dispatch prices.
        if estimated and estimated["estimated_microdollars"]>attempt["allocation_micro"]:
            raise BudgetError("Reported usage exceeds the frozen allocation; retain the hold for billing inspection.")
        charged=estimated["estimated_microdollars"] if estimated else attempt["allocation_micro"]
        accounting="provider-token-estimate" if estimated else "conservative-scene-allocation"
        data=line["data"]
        recipe=dict(data["voice_recipe"])
        source={"index":line["line_index"],"speaker":data["turn"]["speaker"],
                "character_id":data["character"]["character_id"],"source":str(directory/"raw.wav"),
                "source_sha256":attempt["audio_sha256"],"text":data["turn"]["text"],"style":data["turn"]["style"],
                "actual_model":attempt["model"],"recipe_id":recipe["recipe_id"],
                "character_revision":data["character"]["revision"],"voice_recipe_snapshot":recipe,
                "source_request_sha256":attempt["request_sha256"],"source_response_sha256":attempt["response_sha256"],
                "source_attempt":{"id":attempt["id"],"usage":usage,"pricing_estimate":estimated,
                                  "interaction_id":metadata.get("interaction_id"),"output_limit_reached":at_limit},
                "source_job":job["id"]}
        with self.budget.transaction() as db:
            reservation=db.execute("SELECT state FROM reservations WHERE attempt_id=?",(attempt["id"],)).fetchone()
            if reservation is None:raise BudgetError("No retained scene allocation exists.")
            if reservation["state"]=="reserved":self.budget.settle(attempt["id"],charged,accounting,connection=db)
            elif reservation["state"]!="spent":raise BudgetError("Dispatched scene allocation is unavailable.")
            else:
                prior=db.execute("SELECT charged_micro,accounting FROM reservations WHERE attempt_id=?",(attempt["id"],)).fetchone()
                charged,accounting=prior["charged_micro"],prior["accounting"]
            db.execute("""UPDATE scene_attempts SET state='completed',metadata_json=?,charged_micro=?,accounting=?
                       WHERE id=?""",(encode(metadata).decode(),charged,accounting,attempt["id"]))
            db.execute("UPDATE scene_lines SET state=?,source_json=? WHERE job_id=? AND line_index=?",
                       ("review_required" if at_limit else "completed",encode(source).decode(),job["id"],line["line_index"]))
        return not at_limit

    def _finalize(self,job_id):
        from .scene_audio import assemble_scene
        job=self.status(job_id)
        if not all(line["state"]=="completed" for line in job["lines"]):return False
        sources=[line["source"] for line in job["lines"]]
        for source in sources:self._verify_source(source)
        directory=self._directory(job_id)
        receipt=assemble_scene(directory/"output",job["script"],sources,stems=job["policy"]["stems"])
        counts={}
        for source in sources:counts[source["actual_model"]]=counts.get(source["actual_model"],0)+1
        summary={"schema_version":1,"kind":"scene_summary","job_id":job_id,"parent_job":job["parent_job"],
                 "title":job["script"]["title"],"line_count":len(sources),"duration_ms":receipt["duration_ms"],
                 "audio_sha256":receipt["audio_sha256"],"models_by_line_count":counts,
                 "total_job_budget_microdollars":job["total_limit_micro"],
                 "new_dispatches":sum(attempt["state"]!="pending" for line in job["lines"] for attempt in line["attempts"]),
                 "reused_source_lines":sum(source.get("source_job")!=job_id for source in sources),
                 "receipt_path":str((directory/"output/assembly.json").relative_to(self.root)),
                 "receipt_sha256":digest(read(directory/"output/assembly.json",MAX_JSON)),
                 "playable_wav":str((directory/"output/scene.wav").relative_to(self.root)),
                 "playable_mp3":str((directory/"output/scene.mp3").relative_to(self.root)),
                 "transcript":str((directory/"output/transcript.txt").relative_to(self.root)),
                 "auditions":job["auditions"],
                 "accounting_notice":"Live status separates estimates, planned reservations, unresolved holds, and invoice evidence.",
                 "publication":"Not published or scheduled."}
        summary_bytes=encode(summary);write(directory/"scene-summary.json",summary_bytes)
        receipt=receipt|{"scene_summary":{"path":str((directory/"scene-summary.json").relative_to(self.root)),
                                        "sha256":digest(summary_bytes)}}
        with self.budget.connect() as db:
            db.execute("UPDATE scene_jobs SET status='ready',outputs_json=?,wait_json=NULL,error=NULL,updated_at=? WHERE id=?",
                       (encode(receipt).decode(),now(),job_id))
        return True

    def _recover_owned(self,job_id):
        job=self.status(job_id)
        for line in job["lines"]:
            for attempt in self._attempts(job_id,line["line_index"]):
                if attempt["state"] not in ("running","uncertain","reconciled"):continue
                if attempt["audio_sha256"] and attempt["response_sha256"]:
                    try:self._complete(job,line,attempt);continue
                    except (OSError,ValueError,KeyError,TypeError):pass
                if attempt["state"]!="reconciled":
                    with self.budget.connect() as db:
                        db.execute("UPDATE scene_attempts SET state='uncertain' WHERE id=?",(attempt["id"],))
                        db.execute("UPDATE scene_lines SET state='uncertain' WHERE job_id=? AND line_index=?",(job_id,line["line_index"]))
        current=self.status(job_id)
        if any(line["state"]=="uncertain" for line in current["lines"]):
            self._set_job(job_id,"uncertain",error="A dispatched line has no verified durable result. Recovery never redispatches it.")
        elif any(line["state"]=="review_required" for line in current["lines"]):
            self._set_job(job_id,"review_required",error="A line reached its output ceiling; retained audio needs deliberate revision/review.")
        elif all(line["state"]=="completed" for line in current["lines"]):
            try:self._finalize(job_id)
            except (ValueError,OSError,KeyError):self._set_job(job_id,"assembly_failed",error="Completed sources are retained; local assembly needs inspection.")
        elif current["status"] in ("running","uncertain","assembly_failed"):
            self._set_job(job_id,"queued",error="Durable results recovered; only unfinished work remains.")
        return self.status(job_id)

    def recover(self,job_id):
        self._directory(job_id)
        with self._worker():return self._recover_owned(job_id)

    def run(self,job_id,resume=False,transport=None,progress=None):
        self._directory(job_id)
        with self._worker():
            try:return self._run_owned(job_id,resume=resume,transport=transport,progress=progress)
            except Exception:
                current=self.status(job_id)
                running=any(attempt["state"]=="running" for line in current["lines"] for attempt in line["attempts"])
                if running:
                    with self.budget.connect() as db:
                        indices=[row[0] for row in db.execute("SELECT line_index FROM scene_attempts WHERE job_id=? AND state='running'",(job_id,))]
                        db.execute("UPDATE scene_attempts SET state='uncertain' WHERE job_id=? AND state='running'",(job_id,))
                        for index in indices:db.execute("UPDATE scene_lines SET state='uncertain' WHERE job_id=? AND line_index=?",(job_id,index))
                self._set_job(job_id,"uncertain" if running else "blocked",
                              error="Scene preflight or local persistence could not complete; retained inputs and allocations are preserved.")
                return self.status(job_id)

    def _run_owned(self,job_id,resume=False,transport=None,progress=None):
        job=self._recover_owned(job_id)
        if job["status"]=="ready":return job
        if job["status"]!="queued" and not resume:return job
        if any(line["state"] in ("uncertain","review_required","blocked") for line in job["lines"]):return job
        policy=ScenePolicy.model_validate(job["policy"])
        key=self.key_provider()
        if not key:
            self._set_job(job_id,"waiting_key",error="No private Gemini credential is configured; nothing dispatched.")
            return self.status(job_id)
        scope=self._scope(key)
        self._set_job(job_id,"running")
        for initial in job["lines"]:
            if initial["state"]=="completed":continue
            index=initial["line_index"]
            while True:
                current=self.status(job_id);line=current["lines"][index]
                attempts=self._attempts(job_id,index)
                if any(item["state"] in ("running","uncertain","reconciled") for item in attempts):
                    self._set_job(job_id,"uncertain",error="An earlier dispatched result needs explicit recovery or a new reviewed revision.")
                    return self.status(job_id)
                if attempts and attempts[-1]["state"]=="rejected":
                    self._set_job(job_id,"blocked",error="Provider rejected this line; inspect diagnostics and revise deliberately.")
                    return self.status(job_id)
                model,quota=self._choose_model(scope,policy)
                if model is None:
                    self._set_job(job_id,"waiting_daily",wait={"reason":"daily_quota","line_index":index,
                                  "retry_at_unix":quota["daily_until"],"model":policy.models[0],
                                  "notice":"Known daily exhaustion; pacing cannot resolve it. No allowed fallback is available."})
                    return self.status(job_id)
                delay=max(0,quota["next_at"]-self.clock())
                if delay>policy.max_wait_seconds:
                    self._set_job(job_id,"waiting_quota",wait={"reason":"quota_delay","line_index":index,
                                  "retry_at_unix":quota["next_at"],"model":model})
                    return self.status(job_id)
                if delay:
                    self._set_job(job_id,"waiting_quota",wait={"reason":"paced_dispatch","line_index":index,
                                  "retry_at_unix":quota["next_at"],"model":model})
                    if progress:progress({"event":"waiting","job_id":job_id,"line_index":index,"seconds":round(delay,3)})
                    self.sleep(delay)
                    if self.clock()+0.001<quota["next_at"]:return self.status(job_id)
                    self._set_job(job_id,"running")
                quota_rejections=sum(item["state"]=="quota_rejected" for item in attempts)
                attempt=attempts[-1] if attempts and attempts[-1]["state"]=="pending" else None
                if attempt is None:
                    if quota_rejections>policy.max_quota_retries:
                        self._set_job(job_id,"waiting_policy",wait={"reason":"quota_retry_policy_exhausted","line_index":index,"model":model})
                        return self.status(job_id)
                    try:attempt=self._new_attempt(current,line,model)
                    except BudgetError:
                        self._set_job(job_id,"waiting_budget",wait={"reason":"retry_allocation_not_available","line_index":index},
                                      error="Successful estimates and unresolved holds remain inside the original total job budget.")
                        return self.status(job_id)
                payload=line["data"]["requests"][model]
                from .production_pricing import request_allocation
                needed=request_allocation(payload)["reservation_microdollars"]
                if needed>attempt["allocation_micro"]:
                    self._set_job(job_id,"waiting_budget",error="Current request pricing exceeds its frozen allocation; revise explicitly.")
                    return self.status(job_id)
                # Revalidate only the unfinished line's pinned recipe. Completed originals are never regenerated.
                unary=gemini.Script.model_validate(current["script"]).model_copy(update={
                    "speakers":[gemini.SpeakerRecipe.model_validate(line["data"]["speaker"])],
                    "turns":[gemini.Turn.model_validate(line["data"]["turn"])]},deep=True)
                verified,_=self.characters.bind_cast(unary)
                if verified.model_dump()!=unary.model_dump():raise ValueError("Pinned scene recipe no longer matches retained inputs.")
                request_sha=digest(encode(payload))
                with self.budget.connect() as db:
                    unresolved=db.execute("""SELECT 1 FROM scene_attempts WHERE scope_id=? AND request_sha256=?
                                            AND state IN ('running','uncertain') AND id<>? LIMIT 1""",(scope,request_sha,attempt["id"])).fetchone()
                if unresolved:
                    self._set_job(job_id,"blocked",error="An equivalent dispatched result is unresolved; no duplicate call was sent.")
                    return self.status(job_id)
                directory=self._directory(job_id)/"attempts"/attempt["id"]
                write(directory/"request.json",encode(payload))
                with self.budget.connect() as db:
                    db.execute("""UPDATE scene_attempts SET state='running',model=?,request_json=?,request_sha256=?,
                               scope_id=?,dispatched_at=? WHERE id=?""",
                               (model,encode(payload).decode(),request_sha,scope,self.clock(),attempt["id"]))
                    db.execute("UPDATE scene_lines SET state='running' WHERE job_id=? AND line_index=?",(job_id,index))
                attempt=dict(attempt)|{"model":model,"request_json":encode(payload).decode(),"request_sha256":request_sha}
                try:
                    audio,metadata=gemini.generate(payload,key,transport=transport)
                    response=encode(metadata)
                    with self.budget.connect() as db:
                        db.execute("UPDATE scene_attempts SET audio_sha256=?,response_sha256=? WHERE id=?",
                                   (digest(audio),digest(response),attempt["id"]))
                    write(directory/"raw.wav",audio);write(directory/"response.json",response)
                    attempt.update(audio_sha256=digest(audio),response_sha256=digest(response))
                    self._pace_after_response(scope,model,policy)
                    complete=self._complete(current,line,attempt)
                    if not complete:
                        self._set_job(job_id,"review_required",error="Output-token ceiling reached; audio retained without silently accepting truncation.")
                        return self.status(job_id)
                    if progress:progress({"event":"line_completed","job_id":job_id,"line_index":index,"model":model})
                    break
                except gemini.GeminiFailure as exc:
                    diagnostic=getattr(exc,"diagnostic",None) or {"classification":"dispatched_uncertain","message":str(exc)}
                    classification=getattr(exc,"classification","dispatched_uncertain")
                    quota_rejected=classification=="quota-rejected"
                    state="quota_rejected" if quota_rejected else "uncertain" if exc.ambiguous else "rejected"
                    write(directory/"diagnostic.json",encode(diagnostic))
                    with self.budget.connect() as db:
                        db.execute("UPDATE scene_attempts SET state=?,diagnostic_json=? WHERE id=?",
                                   (state,encode(diagnostic).decode(),attempt["id"]))
                        db.execute("UPDATE scene_lines SET state=? WHERE job_id=? AND line_index=?",
                                   ("pending_retry" if quota_rejected else "uncertain" if exc.ambiguous else "blocked",job_id,index))
                    self._pace_after_response(scope,model,policy,diagnostic)
                    if not quota_rejected:
                        self._set_job(job_id,"uncertain" if exc.ambiguous else "blocked",error=str(exc))
                        return self.status(job_id)
                    # Explicitly rejected quota calls retain their allocation.
                    # A policy retry is a new attempt under the same total cap.
                    from .provider_diagnostics import quota_kind,retry_after_seconds
                    if quota_kind(diagnostic)=="unknown" and retry_after_seconds(diagnostic) is None:
                        self._set_job(job_id,"waiting_quota",wait={"reason":"quota_without_retry_hint","line_index":index,"model":model})
                        return self.status(job_id)
                    continue
                except Exception:
                    with self.budget.connect() as db:
                        db.execute("UPDATE scene_attempts SET state='uncertain' WHERE id=?",(attempt["id"],))
                        db.execute("UPDATE scene_lines SET state='uncertain' WHERE job_id=? AND line_index=?",(job_id,index))
                    self._set_job(job_id,"uncertain",error="Dispatched work was interrupted; retained results can be recovered without redispatch.")
                    return self.status(job_id)
        key=None
        try:self._finalize(job_id)
        except (ValueError,OSError,KeyError):
            self._set_job(job_id,"assembly_failed",error="All line originals are retained; local assembly needs inspection.")
        return self.status(job_id)

    def revise(self,job_id,line_index,text=None,style=None,recipe_id=None,budget_usd=None,policy=None):
        if budget_usd is None:raise BudgetError("Choose an explicit total revision budget.")
        with self._worker():
            parent=self.status(job_id)
            if type(line_index) is not int or not 0<=line_index<len(parent["lines"]):raise ValueError("Choose an existing scene line.")
            for line in parent["lines"]:
                if any(item["state"] in ("running","uncertain") for item in line["attempts"]):
                    raise ValueError("Reconcile/recover uncertain dispatched work before creating a replacement revision.")
                if line["line_index"]!=line_index and line["state"]!="completed" and any(
                        item["state"] in ("rejected","reconciled") for item in line["attempts"]):
                    raise ValueError("Another rejected line needs its own deliberate revision.")
            script=gemini.Script.model_validate(parent["script"])
            turn=script.turns[line_index].model_dump()
            if text is not None:turn["text"]=text
            if style is not None:turn["style"]=style
            script.turns[line_index]=gemini.Turn.model_validate(turn)
            overrides={line["line_index"]:{key:line["data"][key] for key in ("speaker","character","voice_recipe")}
                       for line in parent["lines"]}
            if recipe_id is not None:
                data=overrides[line_index]
                recipe=self.characters.get_recipe(data["character"]["character_id"],recipe_id)
                self.characters._verify_recipe(recipe)
                speaker=gemini.SpeakerRecipe.model_validate(data["speaker"]).model_copy(update={"voice_recipe_id":recipe_id})
                unary=script.model_copy(update={"speakers":[speaker],"turns":[script.turns[line_index]]},deep=True)
                verified,cast=self.characters.bind_cast(unary)
                selected=dict(cast["speakers"][0]["voice_recipe"])
                baseline=next(item for item in script.speakers if item.name==turn["speaker"])
                if recipe_id!=baseline.voice_recipe_id:
                    selected["_scene_override"]={"baseline_recipe_id":baseline.voice_recipe_id,
                                                 "parent_job":job_id,"line_index":line_index}
                overrides[line_index]={"speaker":verified.speakers[0].model_dump(),
                                      "character":cast["speakers"][0]["character"],"voice_recipe":selected}
            reuse={line["line_index"]:line["source"] for line in parent["lines"]
                   if line["line_index"]!=line_index and line["state"]=="completed"}
            explicit_policy=policy.model_dump(exclude_unset=True) if isinstance(policy,ScenePolicy) else policy
            if explicit_policy is not None and not isinstance(explicit_policy,dict):
                raise ValueError("Revision policy must be an object.")
            revision_policy=parent["policy"] | (explicit_policy or {})
            return self._create(script,budget_usd,revision_policy,parent=job_id,overrides=overrides,reuse=reuse,frozen_cast=parent["cast"])

    def audition(self,character_id,text,style="",recipe_id=None,budget_usd=None):
        if budget_usd is None:raise BudgetError("Choose an explicit total audition budget.")
        with self._operation():
            record=self.characters.get(character_id)
            speaker={"name":record["name"],"character_id":character_id}
            if recipe_id is not None:speaker["voice_recipe_id"]=recipe_id
            script={"programme_id":"character-audition","series_id":"character-auditions","kind":"programme",
                    "title":"Voice audition: "+record["name"],"language":"en",
                    "speakers":[speaker],"turns":[{"speaker":record["name"],"text":text,"style":style}]}
            return self._create(script,budget_usd)

    def reconcile(self,job_id,attempt_id,charged_usd,evidence):
        from .production_pricing import verified_invoice_charge
        self._directory(job_id)
        if not isinstance(attempt_id,str) or not re.fullmatch(IDENTIFIER,attempt_id):raise ValueError("Choose one retained attempt.")
        charge=verified_invoice_charge(charged_usd,evidence,confirmed=True)
        charged=charge["charged_microdollars"]
        with self._worker(),self.budget.transaction() as db:
            attempt=db.execute("SELECT * FROM scene_attempts WHERE id=? AND job_id=?",(attempt_id,job_id)).fetchone()
            if attempt is None or attempt["state"] not in ("completed","quota_rejected","rejected","uncertain","reconciled"):
                raise ValueError("Only dispatched attempts can be reconciled from billing evidence.")
            reservation=db.execute("SELECT * FROM reservations WHERE attempt_id=?",(attempt_id,)).fetchone()
            if charged>reservation["reserved_micro"]:raise BudgetError("Verified charge exceeds the recorded allocation; inspect billing before changing limits.")
            if reservation["state"]=="reserved":self.budget.settle(attempt_id,charged,"verified-scene-billing",connection=db)
            elif reservation["state"]=="spent":
                db.execute("UPDATE reservations SET charged_micro=?,accounting='verified-scene-billing' WHERE attempt_id=?",(charged,attempt_id))
            else:raise BudgetError("No dispatched allocation remains.")
            db.execute("INSERT INTO scene_reconciliations(attempt_id,charged_micro,evidence,recorded_at) VALUES(?,?,?,?)",
                       (attempt_id,charged,evidence,now()))
            # A billing reconciliation never claims missing audio exists.
            if attempt["state"]=="uncertain":
                db.execute("UPDATE scene_attempts SET state='reconciled',charged_micro=?,accounting='verified-scene-billing' WHERE id=?",(charged,attempt_id))
                db.execute("UPDATE scene_lines SET state='blocked' WHERE job_id=? AND line_index=?",(job_id,attempt["line_index"]))
            else:
                db.execute("UPDATE scene_attempts SET charged_micro=?,accounting='verified-scene-billing' WHERE id=?",(charged,attempt_id))
        return self.status(job_id)

    def import_sources(self,script,mapping,budget_usd,policy=None):
        """Adopt verified unary legacy sources; never rewrite or reclassify old takes."""
        if not isinstance(mapping,list) or not mapping or len(mapping)>200:raise ValueError("Supply a bounded source map.")
        with self._worker():
            authored=gemini.Script.model_validate(script.model_dump() if isinstance(script,gemini.Script) else script)
            active_policy=ScenePolicy.model_validate(policy or {})
            speakers={item.name:item for item in authored.speakers}
            baseline={};snapshots={};reuse={};overrides={}
            for entry in mapping:
                if not isinstance(entry,dict):raise ValueError("Each source mapping must be an object.")
                index=entry.get("line_index",entry.get("turn_index"))
                take_id=entry.get("take_id");request_index=entry.get("request_index")
                if (type(index) is not int or not 0<=index<len(authored.turns) or index in reuse
                        or not isinstance(take_id,str) or not re.fullmatch(IDENTIFIER,take_id)
                        or type(request_index) is not int or not 0<=request_index<200):
                    raise ValueError("Each mapping needs one unique line and exact legacy take/request identity.")
                with self.budget.connect() as db:
                    take=db.execute("SELECT * FROM takes WHERE id=?",(take_id,)).fetchone()
                    step=db.execute("SELECT * FROM generation_requests WHERE take_id=? AND request_index=?",(take_id,request_index)).fetchone()
                    step_count=db.execute("SELECT COUNT(*) FROM generation_requests WHERE take_id=?",(take_id,)).fetchone()[0]
                if take is None or step is None or step["state"]!="completed":
                    raise ValueError("Only durably completed legacy request audio can be adopted.")
                old_script=gemini.Script.model_validate_json(take["script_json"])
                old_plan=json.loads(take["plan_json"]) if take["plan_json"] else None
                indices=old_plan["requests"][request_index]["turn_indices"] if old_plan else ([0] if len(old_script.turns)==1 else [])
                if len(indices)!=1:raise ValueError("A mixed/native dialogue request does not establish isolated line timing.")
                old_turn=old_script.turns[indices[0]];expected=authored.turns[index]
                if old_turn.model_dump()!=expected.model_dump():
                    raise ValueError("Source text, speaker and delivery must exactly match the target line.")
                request=json.loads(step["request_json"])
                if len(request.get("input",[{}])[0].get("content",[]))!=1:
                    raise ValueError("The source request must contain one known generated line.")
                old_speaker=next(item for item in old_script.speakers if item.name==old_turn.speaker)
                wanted=speakers[old_turn.speaker]
                if (old_speaker.character_id!=wanted.character_id or old_speaker.style!=wanted.style
                        or wanted.character_revision is not None and old_speaker.character_revision!=wanted.character_revision
                        or wanted.voice_recipe_id is not None and old_speaker.voice_recipe_id!=wanted.voice_recipe_id):
                    raise ValueError("Source character pins and baseline delivery differ from the authored scene.")
                old_cast=json.loads(take["cast_snapshot_json"])
                snapshot=next(item for item in old_cast["speakers"] if item["speaker"]==old_turn.speaker)
                recipe=dict(snapshot["voice_recipe"])
                if self.characters.get_recipe(old_speaker.character_id,recipe["recipe_id"])!=recipe:
                    raise ValueError("Source recipe snapshot conflicts with immutable registry history.")
                # Existing audio is usable after a provider voice expires: verify
                # retained registry identity, never invoke active-ID validation.
                if old_turn.speaker not in baseline:
                    baseline[old_turn.speaker]=old_speaker
                    snapshots[old_turn.speaker]=snapshot
                baseline_speaker=baseline[old_turn.speaker]
                if old_speaker.character_revision!=baseline_speaker.character_revision:
                    raise ValueError("One imported character needs coherent frozen identity revisions.")
                if recipe["recipe_id"]!=baseline_speaker.voice_recipe_id:
                    recipe["_scene_override"]={"baseline_recipe_id":baseline_speaker.voice_recipe_id,
                                               "parent_job":take_id,"line_index":index}
                overrides[index]={"speaker":old_speaker.model_dump(),"character":snapshot["character"],"voice_recipe":recipe}
                directory=self.root/".runtime/studio/takes"/take_id
                part=directory if step_count==1 else directory/"requests"/f"{request_index:04d}"
                raw=read(part/"raw.wav",gemini.MAX_AUDIO_BYTES);response=read(part/"response.json",MAX_JSON)
                metadata=json.loads(response)
                if json.loads(step["metadata_json"])!=metadata:raise ValueError("Legacy response metadata conflicts with its ledger.")
                if metadata.get("audio_sha256")!=digest(raw) or metadata.get("audio_bytes")!=len(raw):
                    raise ValueError("Legacy audio does not match its authoritative provider-result identity.")
                model=take["model"]
                if request.get("model")!=model or metadata.get("model") not in (None,model,"models/"+model):
                    raise ValueError("Legacy source model identity is inconsistent.")
                if entry.get("audio_sha256") not in (None,digest(raw)):raise ValueError("Source receipt audio hash does not match.")
                for name,value in (("text",expected.text),("speaker",expected.speaker),("model",model)):
                    if name in entry and entry[name]!=value:raise ValueError("Source receipt differs from verified legacy records.")
                if "usage" in entry and entry["usage"]!=metadata.get("usage",{}):raise ValueError("Source receipt usage differs from retained metadata.")
                if "interaction_id" in entry and entry["interaction_id"]!=metadata.get("interaction_id"):
                    raise ValueError("Source receipt response identity differs from retained metadata.")
                if "audio_path" in entry and entry["audio_path"]!=(part/"raw.wav").relative_to(self.root).as_posix():
                    raise ValueError("Source receipt path does not identify the verified legacy recording.")
                source={"index":index,"speaker":expected.speaker,"character_id":old_speaker.character_id,
                        "source":str(part/"raw.wav"),"source_sha256":digest(raw),"text":expected.text,"style":expected.style,
                        "actual_model":model,"recipe_id":recipe["recipe_id"],"character_revision":old_speaker.character_revision,
                        "voice_recipe_snapshot":recipe,"source_request_sha256":digest(encode(request)),
                        "source_response_sha256":digest(response),"source_job":None,"source_take":take_id,
                        "source_request_index":request_index,
                        "source_attempt":{"legacy_attempt_id":step["attempt_id"],"usage":metadata.get("usage",{}),
                                          "interaction_id":metadata.get("interaction_id"),
                                          "accounting":"Prior legacy ledger; no new dispatch, charge or reconciliation implied."}}
                self._verify_source(source);reuse[index]=source
            # Resolve characters with unfinished material normally. Covered cast
            # members use their actual old snapshots, not today's preferred voice.
            for name,speaker in speakers.items():
                if name not in baseline:
                    turn=next(turn for turn in authored.turns if turn.speaker==name)
                    unary=authored.model_copy(update={"speakers":[speaker],"turns":[turn]},deep=True)
                    resolved,cast=self.characters.bind_cast(unary)
                    baseline[name]=resolved.speakers[0];snapshots[name]=cast["speakers"][0]
            bound=authored.model_copy(update={"speakers":[baseline[item.name] for item in authored.speakers]},deep=True)
            frozen_cast={"schema_version":1,"speakers":[snapshots[item.name] for item in authored.speakers]}
            result=self._create(bound,budget_usd,active_policy,overrides=overrides,reuse=reuse,frozen_cast=frozen_cast)
            if len(reuse)==len(authored.turns):
                self._finalize(result["id"]);result=self.status(result["id"])
            return result
