"""
CrowdVision Demo v3 — Zone-based Multi-Camera Event Safety Intelligence

Each zone gets its own camera image. Per-zone analysis:
  - Crowd density (local model + OpenAI enhancement)
  - Anomaly detection (OpenAI Vision)
  - Area mapping (OpenAI)
Combined: zone risk matrix, dispatch decisions, full AI report, PDF/JSON export.
"""
import base64, io, json, os, sys, traceback, re, tempfile
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
from dotenv import load_dotenv

DEMO_DIR = Path(__file__).parent
REPO_ROOT = DEMO_DIR.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))
CKPT_ROOT = REPO_ROOT / "checkpoints"

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI(title="CrowdVision", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
samples_dir = DEMO_DIR / "samples"
if samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")

# OpenAI from .env
_oc: Optional[OpenAI] = None
def oai() -> Optional[OpenAI]:
    global _oc
    if _oc is None:
        k = os.environ.get("OPENAI_API_KEY", "")
        if k: _oc = OpenAI(api_key=k)
    return _oc

# PyTorch density model
import torch
from torchvision import transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_mcache = {}

def density_model():
    if "dm" in _mcache: return _mcache["dm"]
    try:
        for cls_name, mod_path, ckpt in [
            ("AdaptiveCSRNet","src.models.density.adaptive_csrnet","adaptive_csrnet_shaA"),
            ("CSRNet","src.models.density.csrnet","csrnet_shaA"),
        ]:
            cp = CKPT_ROOT / ckpt / "best.pt"
            if cp.exists():
                mod = __import__(mod_path, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                m = cls(load_weights=False).to(DEVICE) if "Adaptive" in cls_name else cls().to(DEVICE)
                sd = torch.load(cp, map_location=DEVICE, weights_only=False)
                m.load_state_dict(sd.get("model", sd)); m.eval()
                _mcache["dm"] = m; return m
    except Exception as e: print(f"[W] density: {e}")
    return None

dtf = transforms.Compose([transforms.Resize((576,768)),transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def pil_b64(img, fmt="PNG"):
    b=io.BytesIO(); img.save(b,format=fmt); return base64.b64encode(b.getvalue()).decode()

def heatmap_b64(arr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,1,figsize=(arr.shape[1]/80,arr.shape[0]/80),dpi=120)
    ax.imshow(arr,cmap="jet",interpolation="bilinear"); ax.axis("off")
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    b=io.BytesIO(); fig.savefig(b,format="png",bbox_inches="tight",pad_inches=0,dpi=120)
    plt.close(fig); b.seek(0); return base64.b64encode(b.read()).decode()

def overlay_b64(orig, arr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.cm as cm
    o=np.array(orig.resize((arr.shape[1],arr.shape[0])))
    n=(arr-arr.min())/(arr.max()-arr.min()+1e-8)
    h=(cm.get_cmap("jet")(n)[:,:,:3]*255).astype(np.uint8)
    bl=(0.45*o.astype(float)+0.55*h.astype(float)).astype(np.uint8)
    return pil_b64(Image.fromarray(bl))

# Chat history per session
chat_histories: Dict[str, list] = {}

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(str(DEMO_DIR/"index.html"), media_type="text/html")

@app.get("/api/health")
async def health():
    return {"status":"ok","device":DEVICE,"openai":oai() is not None,
            "density_ready":any((CKPT_ROOT/d/"best.pt").exists() for d in ["adaptive_csrnet_shaA","csrnet_shaA"])}

@app.get("/api/samples")
async def list_samples():
    if not samples_dir.exists(): return {"density":[],"anomaly_normal":[],"anomaly_abnormal":[]}
    fs=sorted(samples_dir.iterdir())
    return {"density":[f"/samples/{f.name}" for f in fs if f.name.startswith("density_")],
            "anomaly_normal":[f"/samples/{f.name}" for f in fs if "normal" in f.name and "abnormal" not in f.name],
            "anomaly_abnormal":[f"/samples/{f.name}" for f in fs if "abnormal" in f.name]}

@app.post("/api/analyze_zone")
async def analyze_zone(file: UploadFile=File(...), zone_name: str=Form("Zone-A")):
    """Full per-zone analysis: density + anomaly + area mapping."""
    ib = await file.read()
    img = Image.open(io.BytesIO(ib)).convert("RGB")
    ib64 = pil_b64(img.resize((640,480)))
    result = {"zone": zone_name, "original": ib64}

    # 1. Density (local model)
    dm = density_model()
    if dm:
        inp = dtf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = dm(inp)
            if isinstance(p,(tuple,list)): p=p[0]
        dmap = p[0,0].cpu().numpy()
        count = float(dmap.sum())
        result["density_count"] = round(count,1)
        result["heatmap"] = heatmap_b64(dmap)
        result["overlay"] = overlay_b64(img, dmap)

    # 2. OpenAI enhanced density + anomaly + area mapping (single call)
    client = oai()
    if client:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":[
                {"type":"text","text":f"""Analyze this surveillance camera feed for zone "{zone_name}". You are an expert crowd safety AI.

ANOMALY DETECTION — look specifically for:
- Unusual crowd behavior: stampede, crowd running/panic, counter-flow, crushing
- Dangerous objects: vehicles in pedestrian zones, weapons/guns, knives
- Environmental hazards: smoke, fire, gas, flooding
- Security threats: fights, violence, suspicious packages, abandoned bags
- Crowd issues: overcrowding, blocked exits, bottlenecks, fallen persons
- Any other unusual or dangerous situation

Provide JSON:
{{
  "crowd_estimate": number,
  "density_level": "LOW/MODERATE/HIGH/EXTREME",
  "crowd_description": "brief crowd distribution and behavior",
  "anomaly_score": float 0.0-1.0,
  "is_anomaly": bool,
  "anomaly_type": "none OR specific type (vehicle intrusion/smoke/fire/weapon/stampede/fight/etc)",
  "anomaly_severity": "NONE/LOW/MEDIUM/HIGH/CRITICAL",
  "anomaly_details": "what you see — be specific about threats",
  "recommended_action": "specific security action",
  "area_description": "area type (corridor/plaza/entrance/etc)",
  "area_features": "exits, barriers, stages, etc",
  "capacity_concern": "percentage estimate of capacity usage",
  "flow_pattern": "crowd movement pattern"
}}
JSON only."""},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{ib64}"}}
            ]}],
            max_tokens=500,
        )
        try:
            ai = json.loads(resp.choices[0].message.content.strip().strip("```json").strip("```"))
        except:
            ai = {"crowd_estimate":0,"anomaly_score":0,"anomaly_details":resp.choices[0].message.content}
        result["ai_analysis"] = ai
        # Use OpenAI estimate if no local model or to enhance
        if "density_count" not in result:
            result["density_count"] = ai.get("crowd_estimate", 0)
        else:
            # Average local model + OpenAI for better accuracy
            ai_count = ai.get("crowd_estimate", result["density_count"])
            result["density_count_model"] = result["density_count"]
            result["density_count_ai"] = ai_count
            result["density_count"] = round((result["density_count"] * 0.6 + ai_count * 0.4), 1)

    # Risk score
    d_norm = min(1.0, result.get("density_count",0) / 150)
    a_norm = result.get("ai_analysis",{}).get("anomaly_score", 0)
    result["risk_score"] = round(0.55 * d_norm + 0.45 * a_norm, 4)
    result["risk_level"] = "CRITICAL" if result["risk_score"]>.75 else "HIGH" if result["risk_score"]>.5 else "GUARDED" if result["risk_score"]>.25 else "LOW"

    return result

@app.post("/api/dispatch")
async def dispatch_decision(data: str=Form(...)):
    """OpenAI-powered dispatch decision making (fast model)."""
    client = oai()
    if not client: raise HTTPException(500,"OpenAI not configured")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a crowd safety dispatch commander. Give specific deployment decisions. Respond ONLY with valid JSON."},
            {"role":"user","content":f"""Zone analysis data:\n{data}\n\nJSON response:\n{{"priority_zones":[],"deployments":[{{"zone":"name","units":1,"unit_type":"security/medical/traffic","action":"instruction","urgency":"IMMEDIATE/HIGH/MEDIUM/LOW"}}],"overall_threat_level":"NORMAL/ELEVATED/HIGH/SEVERE/CRITICAL","commander_summary":"2-3 sentences","escalation_triggers":["conditions"]}}"""}
        ],
        max_tokens=700,
    )
    try:
        return json.loads(resp.choices[0].message.content.strip().strip("```json").strip("```"))
    except:
        return {"commander_summary": resp.choices[0].message.content, "deployments": []}


import asyncio

@app.post("/api/dispatch_and_report")
async def dispatch_and_report(data: str=Form(...)):
    """Run dispatch + report generation IN PARALLEL for speed."""
    client = oai()
    if not client: raise HTTPException(500,"OpenAI not configured")
    try: parsed = json.loads(data)
    except: parsed = {"raw": data}
    zones_json = json.dumps(parsed.get("zones", parsed), indent=1)

    async def run_dispatch():
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Crowd safety dispatch commander. JSON only."},
                {"role":"user","content":f"Zone data:\n{zones_json}\n\nJSON: {{\"priority_zones\":[],\"deployments\":[{{\"zone\":\"\",\"units\":1,\"unit_type\":\"security/medical/traffic\",\"action\":\"\",\"urgency\":\"IMMEDIATE/HIGH/MEDIUM/LOW\"}}],\"overall_threat_level\":\"NORMAL/ELEVATED/HIGH/SEVERE/CRITICAL\",\"commander_summary\":\"\",\"escalation_triggers\":[]}}"},
            ], max_tokens=600)
        try: return json.loads(r.choices[0].message.content.strip().strip("```json").strip("```"))
        except: return {"commander_summary": r.choices[0].message.content}

    async def run_report():
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"CrowdVision AI safety analyst. Write detailed markdown safety reports."},
                {"role":"user","content":f"""Multi-zone analysis:\n{json.dumps(parsed,indent=1)}\n\nWrite a professional Event Safety Report with: Executive Summary, Crowd Density Analysis, Anomaly Assessment, Zone Risk Matrix (use 🔴🟠🟡🟢), Dispatch Recommendations, Preventive Measures. Be data-driven and actionable."""}
            ], max_tokens=2000, temperature=0.7)
        return {"report": r.choices[0].message.content}

    dispatch_result, report_result = await asyncio.gather(
        asyncio.to_thread(run_dispatch),
        asyncio.to_thread(run_report),
    )
    return {"dispatch": dispatch_result, "report": report_result}

@app.post("/api/report")
async def generate_report(data: str=Form(...)):
    """Generate comprehensive AI safety report."""
    client = oai()
    if not client: raise HTTPException(500,"OpenAI not configured")
    try: parsed = json.loads(data)
    except: parsed = {"raw": data}

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are CrowdVision AI, an expert crowd safety analysis platform. Generate detailed, professional safety reports."},
            {"role":"user","content":f"""Generate a comprehensive Event Safety Report from this multi-zone analysis:

{json.dumps(parsed, indent=2)}

# 🛡️ CROWDVISION EVENT SAFETY REPORT

Include these sections with detailed analysis:

## Executive Summary
Overall situation assessment (3-4 sentences).

## Crowd Density Analysis
Per-zone density breakdown, total estimated attendance, density patterns.

## Anomaly Assessment
Per-zone anomaly findings, severity levels, types detected.

## Zone Risk Matrix
Table of all zones with risk levels, key metrics, specific concerns.

## Area Mapping
Physical description of each zone area, features, capacity estimates.

## Dispatch Recommendations
Specific unit deployments, priority actions, resource allocation.

## Preventive Measures & Forecast
What to monitor, escalation triggers, 30-60 minute forecast.

## Communication Protocol
Who needs to be informed and when.

Use emojis: 🔴 CRITICAL, 🟠 HIGH, 🟡 GUARDED, 🟢 LOW
Be data-driven, specific, and actionable. Format as clean markdown."""}
        ],
        max_tokens=3000, temperature=0.7,
    )
    return {"report": resp.choices[0].message.content}

@app.post("/api/report/json")
async def report_json(data: str=Form(...)):
    """Download analysis as JSON."""
    try: parsed = json.loads(data)
    except: parsed = {"raw": data}
    content = json.dumps(parsed, indent=2)
    return StreamingResponse(io.BytesIO(content.encode()), media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=crowdvision_report.json"})

@app.post("/api/report/pdf")
async def report_pdf(data: str=Form(...)):
    """Download report as PDF."""
    from fpdf import FPDF
    try: parsed = json.loads(data)
    except: parsed = {}

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica","B",20)
    pdf.cell(0,12,"CrowdVision Event Safety Report",ln=True,align="C")
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,8,f"Generated by CrowdVision AI",ln=True,align="C")
    pdf.ln(8)

    report_text = parsed.get("report","")
    # Clean markdown
    report_text = re.sub(r'[🔴🟠🟡🟢🛡️⚠️📊🎯💬✅❌👥💡🔬🔄]','',report_text)
    for line in report_text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            pdf.set_font("Helvetica","B",16); pdf.ln(4)
            pdf.cell(0,8,line[2:],ln=True)
        elif line.startswith("## "):
            pdf.set_font("Helvetica","B",13); pdf.ln(3)
            pdf.cell(0,7,line[3:],ln=True)
        elif line.startswith("### "):
            pdf.set_font("Helvetica","B",11); pdf.ln(2)
            pdf.cell(0,6,line[4:],ln=True)
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica","",10)
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            pdf.cell(0,5,f"  {clean}",ln=True)
        elif line:
            pdf.set_font("Helvetica","",10)
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            pdf.multi_cell(0,5,clean)
        else:
            pdf.ln(2)

    # Zone data
    zones = parsed.get("zones", [])
    if zones:
        pdf.ln(4); pdf.set_font("Helvetica","B",13)
        pdf.cell(0,8,"Zone Data Summary",ln=True)
        pdf.set_font("Helvetica","",9)
        for z in zones:
            pdf.cell(0,5,f"{z.get('zone','?')}: Risk={z.get('risk_level','?')} ({z.get('risk_score',0):.0%}), Density={z.get('density_count',0):.0f}",ln=True)

    buf = io.BytesIO(); pdf.output(buf); buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition":"attachment; filename=crowdvision_report.pdf"})

@app.post("/api/chat")
async def chat(message: str=Form(...), context: str=Form(""), session_id: str=Form("default")):
    """Conversational chatbot with memory."""
    client = oai()
    if not client: return {"reply":"OpenAI not configured. Add OPENAI_API_KEY to .env"}

    if session_id not in chat_histories:
        chat_histories[session_id] = [{"role":"system","content":
            "You are CrowdVision AI, an intelligent crowd safety assistant. You help event commanders "
            "understand crowd analysis results, make safety decisions, and answer questions about "
            "crowd management. You have access to real-time zone analysis data. Be specific, "
            "professional, and actionable. Remember the conversation context."}]

    hist = chat_histories[session_id]
    if context and len(hist) < 3:
        hist.append({"role":"system","content":f"Current analysis data:\n{context[:3000]}"})

    hist.append({"role":"user","content":message})
    if len(hist) > 20: hist = [hist[0]] + hist[-18:]
    chat_histories[session_id] = hist

    resp = client.chat.completions.create(model="gpt-4o", messages=hist, max_tokens=800, temperature=0.7)
    reply = resp.choices[0].message.content
    hist.append({"role":"assistant","content":reply})
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 CrowdVision v3 — http://localhost:8000")
    print(f"   Device: {DEVICE} | OpenAI: {'✓' if oai() else '✗'}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
