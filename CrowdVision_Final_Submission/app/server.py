"""
CrowdVision v4 — Production Crowd Intelligence Platform

Hybrid anomaly detection: Local ML models (FutureFrameNet + ConvAE) 
combined with OpenAI GPT-4o Vision for best-of-both-worlds accuracy.

Endpoints:
    GET  /              → SPA
    GET  /api/health    → System status
    GET  /api/samples   → Sample images
    POST /api/analyze   → Hybrid zone analysis
    POST /api/dispatch  → AI dispatch decisions  
    POST /api/report    → Safety report + dispatch (parallel)
    POST /api/report/pdf  → PDF export
    POST /api/report/json → JSON export
    POST /api/chat      → Conversational AI
"""

import asyncio
import base64
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Paths
APP_DIR = Path(__file__).parent
REPO_ROOT = APP_DIR.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="CrowdVision", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Mount static files
static_dir = APP_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── OpenAI Client ─────────────────────────────────────────────────────

from openai import OpenAI

_oai_client: Optional[OpenAI] = None

def get_openai() -> Optional[OpenAI]:
    global _oai_client
    if _oai_client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            _oai_client = OpenAI(api_key=key)
    return _oai_client

# ── ML Pipeline ───────────────────────────────────────────────────────

import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.inference.pipeline import CrowdVisionPipeline
        _pipeline = CrowdVisionPipeline(
            checkpoint_dir=str(REPO_ROOT / "checkpoints"),
            device=DEVICE,
            enable_density=True,
            enable_anomaly=True,
            enable_forecasting=True,
        )
    return _pipeline

# Image transforms for anomaly models
_anomaly_tf = transforms.Compose([
    transforms.Resize((128, 192)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

_density_tf = transforms.Compose([
    transforms.Resize((576, 768)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Helpers ───────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def generate_heatmap_b64(density_map: np.ndarray) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(density_map.shape[1]/80, density_map.shape[0]/80), dpi=120)
    ax.imshow(density_map, cmap="inferno", interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_overlay_b64(orig: Image.Image, density_map: np.ndarray) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    orig_arr = np.array(orig.resize((density_map.shape[1], density_map.shape[0])))
    norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-8)
    heatmap = (cm.get_cmap("inferno")(norm)[:, :, :3] * 255).astype(np.uint8)
    blend = (0.4 * orig_arr.astype(float) + 0.6 * heatmap.astype(float)).astype(np.uint8)
    return pil_to_b64(Image.fromarray(blend))


# ── Local ML Inference ────────────────────────────────────────────────

@torch.no_grad()
def run_local_density(img: Image.Image) -> dict:
    """Run AdaptiveCSRNet density estimation."""
    pipe = get_pipeline()
    if "density" not in pipe.models:
        return {"error": "Density model not loaded"}

    t0 = time.time()
    inp = _density_tf(img).unsqueeze(0).to(DEVICE)
    dmap = pipe.models["density"](inp)
    if isinstance(dmap, (tuple, list)):
        dmap = dmap[0]
    dmap_np = dmap[0, 0].cpu().numpy()
    count = float(dmap_np.sum())
    latency = (time.time() - t0) * 1000

    return {
        "count": round(count, 1),
        "heatmap_b64": generate_heatmap_b64(dmap_np),
        "overlay_b64": generate_overlay_b64(img, dmap_np),
        "latency_ms": round(latency, 1),
    }


@torch.no_grad()
def run_local_anomaly(img: Image.Image) -> dict:
    """Run ConvAE + MemAE anomaly detection."""
    pipe = get_pipeline()
    if "anomaly" not in pipe.models:
        return {"score": 0.0, "error": "Anomaly model not loaded"}

    t0 = time.time()
    inp = _anomaly_tf(img).unsqueeze(0).to(DEVICE)
    score = pipe.models["anomaly"].reconstruction_error(inp).item()
    latency = (time.time() - t0) * 1000

    return {
        "score": round(score, 6),
        "latency_ms": round(latency, 1),
    }


# ── OpenAI Vision Analysis ───────────────────────────────────────────

def run_openai_analysis(img_b64: str, zone_name: str) -> dict:
    """Run GPT-4o Vision for semantic anomaly detection."""
    client = get_openai()
    if not client:
        return {"error": "OpenAI not configured"}

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Analyze this surveillance camera feed for zone "{zone_name}". You are an expert crowd safety AI.

ANOMALY DETECTION — check for:
- Crowd behavior: stampede, panic, counter-flow, crushing, crowd collapse
- Dangerous objects: vehicles in pedestrian zones, weapons, knives, suspicious packages
- Environmental hazards: smoke, fire, gas, flooding
- Security threats: fights, violence, trespassing, abandoned bags
- Crowd issues: overcrowding, blocked exits, bottlenecks, fallen persons

Respond with JSON only:
{{
  "crowd_estimate": number,
  "density_level": "LOW|MODERATE|HIGH|EXTREME",
  "crowd_description": "brief crowd distribution",
  "anomaly_score": float 0.0-1.0,
  "is_anomaly": boolean,
  "anomaly_type": "none or specific type",
  "anomaly_severity": "NONE|LOW|MEDIUM|HIGH|CRITICAL",
  "anomaly_details": "specific observations",
  "recommended_action": "security action",
  "area_description": "area type",
  "area_features": "exits, barriers, etc",
  "capacity_usage": "percentage estimate",
  "flow_pattern": "crowd movement"
}}"""},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.strip("```json").strip("```").strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"anomaly_score": 0, "anomaly_details": raw}
    except Exception as e:
        result = {"error": str(e), "anomaly_score": 0}

    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result


# ── Hybrid Scoring ────────────────────────────────────────────────────

def compute_hybrid_score(local_convae: float, openai_score: float) -> dict:
    """
    Combine local + OpenAI anomaly scores for best accuracy.
    
    Weights: 30% local (ConvAE reconstruction error), 70% OpenAI (semantic).
    Local score is sigmoid-normalized to [0,1].
    """
    # Normalize ConvAE score (raw reconstruction error → sigmoid)
    local_norm = 1.0 / (1.0 + np.exp(-(local_convae - 0.02) * 80))

    hybrid = 0.30 * local_norm + 0.70 * openai_score

    return {
        "hybrid_score": float(round(hybrid, 4)),
        "local_score": float(round(local_norm, 4)),
        "openai_score": float(round(openai_score, 4)),
        "is_anomaly": bool(hybrid > 0.45),
        "risk_level": (
            "CRITICAL" if hybrid > 0.75 else
            "HIGH" if hybrid > 0.55 else
            "ELEVATED" if hybrid > 0.35 else
            "LOW"
        ),
    }


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(APP_DIR / "index.html"), media_type="text/html")


@app.get("/api/health")
async def health():
    pipe = get_pipeline()
    return {
        "status": "operational",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "openai": get_openai() is not None,
        "models": {
            "density": "density" in pipe.models,
            "anomaly": "anomaly" in pipe.models,
            "forecasting": "forecasting" in pipe.models,
        },
        "model_info": pipe.get_model_info(),
    }


@app.get("/api/samples")
async def list_samples():
    if not static_dir.exists():
        return {"samples": []}
    files = sorted(static_dir.iterdir())
    return {
        "density": [f"/static/{f.name}" for f in files if f.name.startswith("density_")],
        "anomaly_normal": [f"/static/{f.name}" for f in files if "normal" in f.name and "abnormal" not in f.name],
        "anomaly_abnormal": [f"/static/{f.name}" for f in files if "abnormal" in f.name],
    }


@app.post("/api/analyze")
async def analyze_zone(
    file: UploadFile = File(...),
    zone_name: str = Form("Zone-A"),
):
    """
    Hybrid zone analysis: local ML models + OpenAI Vision.
    
    1. Local density (AdaptiveCSRNet) — instant
    2. Local anomaly (ConvAE) — instant
    3. OpenAI Vision (GPT-4o) — ~2s
    4. Hybrid score fusion
    """
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_small = img.resize((640, 480))
    img_b64 = pil_to_b64(img_small)

    result = {
        "zone": zone_name,
        "timestamp": time.time(),
        "original_b64": img_b64,
    }

    # 1. Local density
    density = run_local_density(img)
    result["density"] = density

    # 2. Local anomaly
    local_anomaly = run_local_anomaly(img)
    result["local_anomaly"] = local_anomaly

    # 3. OpenAI analysis
    ai_analysis = await asyncio.to_thread(run_openai_analysis, img_b64, zone_name)
    result["ai_analysis"] = ai_analysis

    # 4. Hybrid scoring
    local_score = local_anomaly.get("score", 0.0)
    openai_score = ai_analysis.get("anomaly_score", 0.0)
    hybrid = compute_hybrid_score(local_score, openai_score)
    result["anomaly"] = hybrid

    # Combined density count (weighted: 60% local, 40% AI)
    local_count = density.get("count", 0)
    ai_count = ai_analysis.get("crowd_estimate", local_count)
    result["crowd_count"] = round(0.6 * local_count + 0.4 * ai_count, 1)
    result["density_level"] = ai_analysis.get("density_level", "UNKNOWN")

    # Risk score
    d_norm = min(1.0, result["crowd_count"] / 150)
    result["risk_score"] = round(0.50 * d_norm + 0.50 * hybrid["hybrid_score"], 4)
    result["risk_level"] = (
        "CRITICAL" if result["risk_score"] > 0.75 else
        "HIGH" if result["risk_score"] > 0.55 else
        "ELEVATED" if result["risk_score"] > 0.35 else
        "LOW"
    )

    return result


@app.post("/api/dispatch_and_report")
async def dispatch_and_report(data: str = Form(...)):
    """Run dispatch + report generation in parallel."""
    client = get_openai()
    if not client:
        raise HTTPException(500, "OpenAI not configured")

    try:
        parsed = json.loads(data)
    except:
        parsed = {"raw": data}

    zones_json = json.dumps(parsed.get("zones", parsed), indent=1)

    async def run_dispatch():
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Crowd safety dispatch commander. Respond with JSON only."},
                {"role": "user", "content": f"""Zone analysis data:
{zones_json}

Respond with JSON:
{{"priority_zones":[],"deployments":[{{"zone":"","units":1,"unit_type":"security/medical/traffic","action":"","urgency":"IMMEDIATE/HIGH/MEDIUM/LOW"}}],"overall_threat_level":"NORMAL/ELEVATED/HIGH/SEVERE/CRITICAL","commander_summary":"2-3 sentences","escalation_triggers":[]}}"""}
            ],
            max_tokens=600,
        )
        try:
            return json.loads(r.choices[0].message.content.strip().strip("```json").strip("```"))
        except:
            return {"commander_summary": r.choices[0].message.content}

    async def run_report():
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "CrowdVision AI safety analyst. Write professional markdown safety reports."},
                {"role": "user", "content": f"""Multi-zone analysis:
{json.dumps(parsed, indent=1)}

Write a professional Event Safety Report with:
- Executive Summary
- Crowd Density Analysis (per-zone)
- Anomaly Assessment (hybrid ML + AI scores)
- Zone Risk Matrix (use 🔴🟠🟡🟢)
- Dispatch Recommendations
- Preventive Measures & Forecast
Be data-driven and actionable."""}
            ],
            max_tokens=2000,
            temperature=0.7,
        )
        return {"report": r.choices[0].message.content}

    dispatch_result, report_result = await asyncio.gather(
        asyncio.to_thread(run_dispatch),
        asyncio.to_thread(run_report),
    )
    return {"dispatch": dispatch_result, "report": report_result}


@app.post("/api/report/json")
async def report_json(data: str = Form(...)):
    try:
        parsed = json.loads(data)
    except:
        parsed = {"raw": data}
    content = json.dumps(parsed, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=crowdvision_report.json"},
    )


@app.post("/api/report/pdf")
async def report_pdf(data: str = Form(...)):
    from fpdf import FPDF

    try:
        parsed = json.loads(data)
    except:
        parsed = {}

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 14, "CrowdVision Event Safety Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Generated by CrowdVision AI v4.0", ln=True, align="C")
    pdf.ln(10)

    report_text = parsed.get("report", "")
    report_text = re.sub(r'[🔴🟠🟡🟢🛡️⚠️📊🎯💬✅❌👥💡🔬🔄]', '', report_text)

    for line in report_text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.ln(4)
            pdf.cell(0, 8, line[2:], ln=True)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(3)
            pdf.cell(0, 7, line[3:], ln=True)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.ln(2)
            pdf.cell(0, 6, line[4:], ln=True)
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica", "", 10)
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            pdf.cell(0, 5, f"  {clean}", ln=True)
        elif line:
            pdf.set_font("Helvetica", "", 10)
            clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            pdf.multi_cell(0, 5, clean)
        else:
            pdf.ln(2)

    zones = parsed.get("zones", [])
    if zones:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Zone Data Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for z in zones:
            pdf.cell(0, 5,
                     f"{z.get('zone', '?')}: Risk={z.get('risk_level', '?')} "
                     f"({z.get('risk_score', 0):.0%}), Count={z.get('crowd_count', 0):.0f}",
                     ln=True)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=crowdvision_report.pdf"},
    )


# Chat
_chat_histories: Dict[str, list] = {}

@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    context: str = Form(""),
    session_id: str = Form("default"),
):
    client = get_openai()
    if not client:
        return {"reply": "OpenAI not configured. Add OPENAI_API_KEY to .env"}

    if session_id not in _chat_histories:
        _chat_histories[session_id] = [{
            "role": "system",
            "content": (
                "You are CrowdVision AI, an intelligent crowd safety assistant. "
                "Help event commanders understand crowd analysis, make safety decisions, "
                "and answer questions. You use hybrid ML+AI anomaly detection. "
                "Be specific, professional, and actionable."
            ),
        }]

    hist = _chat_histories[session_id]
    if context and len(hist) < 3:
        hist.append({"role": "system", "content": f"Current analysis:\n{context[:3000]}"})

    hist.append({"role": "user", "content": message})
    if len(hist) > 20:
        hist = [hist[0]] + hist[-18:]
    _chat_histories[session_id] = hist

    resp = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-4o", messages=hist, max_tokens=800, temperature=0.7
        )
    )
    reply = resp.choices[0].message.content
    hist.append({"role": "assistant", "content": reply})
    return {"reply": reply}


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 CrowdVision v4 — http://localhost:8001")
    print(f"   Device: {DEVICE} | OpenAI: {'✓' if get_openai() else '✗'}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
