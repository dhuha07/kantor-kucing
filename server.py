"""Kantor Kucing - backend agent nyata.

Alur: Moca (router) memilih divisi -> divisi bekerja sesuai urutan serah terima dan
menulis file ke folder miliknya -> Moca menulis REPORT.md. Kode hasil AI TIDAK PERNAH dijalankan.
"""
import asyncio
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


def _need(name):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Variabel {name} belum diisi di backend/.env")
    return v


MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAXTOK = int(os.getenv("KUCING_MAX_TOKENS", "16384"))
TOKEN = os.getenv("KUCING_TOKEN")
if not TOKEN:
    raise RuntimeError("Variabel KUCING_TOKEN belum diisi di Vercel Environment Variables")
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ROOT = Path(os.getenv("KUCING_WORKSPACE", "/tmp/workspaces")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
client = genai.Client(api_key=_need("GEMINI_API_KEY"))
SEM = asyncio.Semaphore(3)

STACK = ("Stack bawaan bila tujuan tidak menyebut lain: backend Python FastAPI, frontend React + Vite + "
         "Tailwind v3, database MySQL (PyMySQL), deploy Vercel.")

# own = satu-satunya folder/file yang boleh ditulis divisi itu (dipaksa di kode)
TEAM = {
    "tuna": {"n": "Tutu", "role": "Tech Lead / Software Architect", "own": ["docs/", "README.md"],
             "sk": "arsitektur sistem, tech stack, kontrak API",
             "fokus": "Tulis docs/ARSITEKTUR.md (komponen, alur data, kontrak API lengkap: path, method, "
                      "request/response, skema tabel) dan README.md (cara menjalankan)."},
    "pixel": {"n": "Ceya", "role": "UI/UX & Product Designer", "own": ["design/"],
              "sk": "user flow, wireframe, design system",
              "fokus": "Tulis design/UX.md: user flow, daftar layar, komponen, token warna dan tipografi, "
                       "kondisi kosong/error/memuat."},
    "sage": {"n": "Jian", "role": "AI / ML Engineer", "own": ["ai/"],
             "sk": "LLM pipeline, RAG, integrasi model",
             "fokus": "Tulis modul AI/LLM yang bisa dipanggil backend, beserta prompt dan cara evaluasinya."},
    "biscuit": {"n": "Kopi", "role": "Senior Fullstack Engineer", "own": ["backend/"],
                "sk": "API, keamanan, modul inti",
                "fokus": "Tulis backend lengkap: entry point, routes, model/skema, validasi input, autentikasi "
                         "(hash password), CORS, requirements.txt."},
    "nori": {"n": "Kino", "role": "Fullstack Engineer", "own": ["frontend/", "db/"],
             "sk": "fitur harian, REST, database",
             "fokus": "Tulis frontend lengkap (termasuk package.json, konfigurasi Vite/Tailwind) dan db/schema.sql."},
    "bolt": {"n": "Juke", "role": "QA & Automation Engineer", "own": ["tests/"],
             "sk": "uji fungsional, automated testing",
             "fokus": "Tulis tes otomatis (pytest untuk backend) dan skenario uji UI yang sesuai kontrak API."},
    "miso": {"n": "Kaley", "role": "DevOps & Cloud Engineer",
             "own": ["deploy/", ".github/", "Dockerfile", "docker-compose.yml", ".env.example"],
             "sk": "CI/CD, deployment, keamanan server",
             "fokus": "Tulis Dockerfile, docker-compose.yml, workflow CI/CD, .env.example, dan panduan deploy."},
}

PLAN_RULES = """Aturan:
1. Pilih HANYA divisi yang benar-benar dibutuhkan, minimal 1 dan maksimal 7 tugas, satu tugas per anggota. Permintaan kecil atau satu bidang (memperbaiki satu bug, satu desain, satu modul) cukup 1-2 divisi. Libatkan banyak divisi hanya untuk tujuan besar lintas bidang. Jangan libatkan divisi agar terlihat sibuk.
2. "butuh" berisi id divisi yang hasilnya harus selesai dulu sebagai bahan tugas itu; kosongkan bila bisa mulai langsung. Contoh alur: tuna dan pixel mulai lebih dulu; biscuit dan nori butuh tuna (dan pixel untuk UI); sage butuh tuna; bolt butuh biscuit dan nori; miso butuh tuna dan biscuit. Jangan membuat lingkaran.
3. Setiap tugas wajib punya "alasan" (satu kalimat: kenapa divisi ini dipilih) dan "rincian" (2-3 kalimat: keluaran konkret berupa file apa, dan untuk siapa).
Balas hanya JSON array, contoh: [{"id":"tuna","judul":"Rancang arsitektur sistem","rincian":"Tentukan komponen, tech stack, kontrak API.","alasan":"Semua divisi butuh kontrak API.","butuh":[]}]. Gunakan id persis dari daftar; judul maksimal 8 kata."""

FILE_RE = re.compile(r"<<<FILE:\s*(.+?)\s*>>>\r?\n(.*?)\r?\n?<<<END>>>", re.S)


class State(TypedDict, total=False):
    goal: str
    context: str
    ws: str
    tasks: list
    results: dict
    report: str


# ---------- util ----------
async def llm(prompt, json_mode=False, max_tokens=MAXTOK):
    cfg = types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else None,
    )
    async with SEM:
        for attempt in range(3):
            try:
                r = await client.aio.models.generate_content(model=MODEL, contents=prompt, config=cfg)
                return r.text or ""
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))


def tree_of(ws, limit=120):
    items = sorted(p.relative_to(ws).as_posix() for p in ws.rglob("*") if p.is_file())
    return "\n".join(items[:limit]) or "(masih kosong)"


def read_files(ws, files, per=6000, total=24000):
    out, used = [], 0
    for f in files:
        try:
            s = (ws / f).read_text(encoding="utf-8")[:per]
        except Exception:
            continue
        if used + len(s) > total:
            break
        used += len(s)
        out.append(f"#### {f}\n{s}")
    return "\n\n".join(out)


def parse_json(raw):
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.S)
        v = json.loads(m.group(0)) if m else []
    if isinstance(v, dict):
        v = next((x for x in v.values() if isinstance(x, list)), [])
    return v


def sanitize(arr):
    seen, ok = set(), []
    for x in arr if isinstance(arr, list) else []:
        i = x.get("id") if isinstance(x, dict) else None
        if isinstance(i, str) and i in TEAM and i not in seen:
            seen.add(i)
            ok.append(x)
    ok = ok[:7]
    ids = {x["id"] for x in ok}
    tasks = []
    for x in ok:
        b = x.get("butuh") if isinstance(x.get("butuh"), list) else []
        dep = [d for d in b if isinstance(d, str) and d in ids and d != x["id"]]
        tasks.append({
            "id": x["id"],
            "judul": str(x.get("judul") or "Tugas")[:80],
            "rincian": str(x.get("rincian") or "")[:600],
            "alasan": str(x.get("alasan") or "")[:160],
            "butuh": list(dict.fromkeys(dep)),
        })
    done, changed = set(), True  # buang lingkaran ketergantungan
    while changed:
        changed = False
        for t in tasks:
            if t["id"] not in done and all(d in done for d in t["butuh"]):
                done.add(t["id"])
                changed = True
    for t in tasks:
        if t["id"] not in done:
            t["butuh"] = []
    return tasks


def safe_path(ws, rel, own):
    rel = rel.strip().replace("\\", "/").lstrip("/")
    if not rel or ":" in rel or ".." in rel.split("/"):
        return None
    if not any(rel == p or rel.startswith(p) for p in own):
        return None
    p = (ws / rel).resolve()
    return p if ws in p.parents else None


def clean(body):
    b = body.strip("\n")
    m = re.match(r"^```[\w+-]*\n(.*)\n```$", b, re.S)
    return (m.group(1) if m else b) + "\n"


def apply_blocks(ws, own, raw):
    files, skipped = [], 0
    matches = list(FILE_RE.finditer(raw))
    for m in matches:
        p = safe_path(ws, m.group(1), own)
        if not p or len(m.group(2)) > 200_000:
            skipped += 1
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(clean(m.group(2)), encoding="utf-8")
        files.append(p.relative_to(ws).as_posix())
    if not files:
        raise ValueError("Tidak ada file valid (cek format atau folder milik divisi)")
    s = re.search(r"RINGKASAN:\s*(.+)", raw)
    summary = s.group(1).strip() if s else f"{len(files)} file dibuat"
    notes = []
    if skipped:
        notes.append(f"{skipped} file ditolak karena di luar folder divisi")
    if raw.count("<<<FILE:") > len(matches):
        notes.append("keluaran terpotong, ada file yang tidak lengkap dan tidak ditulis")
    return files, summary + (f" ({'; '.join(notes)})" if notes else "")


def py_check(ws):
    bad, n = [], 0
    for p in ws.rglob("*.py"):
        n += 1
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")  # hanya mengompilasi, tidak menjalankan
        except SyntaxError as e:
            bad.append(f"{p.relative_to(ws).as_posix()}:{e.lineno} {e.msg}")
    return {"py_files": n, "syntax_errors": bad}


def agent_prompt(tid, t, state, results):
    c, ws = TEAM[tid], Path(state["ws"])
    dep_txt = ""
    for d in t["butuh"]:
        r = results.get(d)
        if r and r["ok"]:
            dep_txt += f"\n### {TEAM[d]['n']} ({TEAM[d]['role']}): {r['summary']}\n{read_files(ws, r['files'])}\n"
        else:
            dep_txt += f"\n### {TEAM[d]['n']}: hasil tidak tersedia, asumsikan seperlunya dan tandai asumsinya.\n"
    return f"""Kamu {c['n']}, {c['role']} di tim software "Kantor Kucing". Keahlian: {c['sk']}.
{STACK}
Konteks proyek dari CEO: {state.get('context') or '-'}

## Tujuan proyek
{state['goal']}

## Tugasmu
{t['judul']}: {t['rincian']}
Fokus divisimu: {c['fokus']}

## Folder milikmu (HANYA boleh menulis di sini)
{', '.join(c['own'])}

## Struktur proyek saat ini
{tree_of(ws)}

## Bahan dari divisi lain
{dep_txt or '(tidak ada, kamu mulai lebih dulu)'}

## Aturan keluaran
- Tulis file lengkap dan siap dipakai, tanpa placeholder seperti "TODO" atau "...".
- Ikuti kontrak dan arsitektur dari divisi lain; jangan mengarang endpoint atau tabel yang tidak ada di bahan.
- Jangan menulis rahasia atau API key; pakai variabel lingkungan dan .env.example.
- Setiap file ditulis persis dengan format ini (path relatif terhadap akar proyek):
<<<FILE: folder/nama.ext>>>
isi file
<<<END>>>
- Setelah semua file, tulis satu baris: RINGKASAN: <maksimal 2 kalimat bahasa Indonesia>"""


# ---------- graph ----------
async def plan_node(state: State, config: RunnableConfig):
    emit = config["configurable"]["emit"]
    team = "\n".join(f"- {k}: {v['n']}, {v['role']} ({v['sk']}); menulis di {', '.join(v['own'])}"
                     for k, v in TEAM.items())
    prompt = (f"Kamu Moca, Project Manager yang rapi dan tenang. Pilih divisi yang tepat untuk tujuan ini.\n"
              f"{STACK}\nKonteks: {state.get('context') or '-'}\n\n## Tujuan dari CEO\n{state['goal']}\n\n"
              f"## Tim yang tersedia\n{team}\n\n{PLAN_RULES}")
    tasks = sanitize(parse_json(await llm(prompt, json_mode=True, max_tokens=4096)))
    if not tasks:
        raise ValueError("Moca tidak menghasilkan rencana yang valid")
    await emit({"t": "plan", "tasks": tasks})
    return {"tasks": tasks}


async def execute_node(state: State, config: RunnableConfig):
    emit = config["configurable"]["emit"]
    tasks = {t["id"]: t for t in state["tasks"]}
    results = {}

    async def run_one(tid):
        await emit({"t": "task_start", "id": tid})
        try:
            raw = await llm(agent_prompt(tid, tasks[tid], state, results))
            files, summary = apply_blocks(Path(state["ws"]), TEAM[tid]["own"], raw)
            results[tid] = {"ok": True, "summary": summary, "files": files}
            await emit({"t": "task_done", "id": tid, "summary": summary, "files": files})
        except Exception as e:
            results[tid] = {"ok": False, "summary": "", "files": []}
            await emit({"t": "task_fail", "id": tid, "error": str(e)[:300]})

    pending, running, settled = set(tasks), {}, set()
    while pending or running:
        for i in [i for i in pending if all(d in settled for d in tasks[i]["butuh"])]:
            pending.discard(i)
            running[asyncio.create_task(run_one(i))] = i
        if not running:
            break
        done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            settled.add(running.pop(d))
    return {"results": results}


async def compile_node(state: State, config: RunnableConfig):
    emit = config["configurable"]["emit"]
    await emit({"t": "compile_start"})
    ws = Path(state["ws"])
    checks = py_check(ws)
    rows = "\n".join(
        f"- {TEAM[i]['n']} ({TEAM[i]['role']}): " + (r["summary"] if r["ok"] else "GAGAL")
        for i, r in state["results"].items())
    prompt = (f"Kamu Moca, Project Manager. Tulis REPORT.md dalam Bahasa Indonesia untuk CEO.\n\n"
              f"## Tujuan\n{state['goal']}\n\n## Hasil per divisi\n{rows}\n\n"
              f"## File yang benar-benar ada\n{tree_of(ws, 200)}\n\n## Hasil cek sintaks Python\n"
              f"{json.dumps(checks, ensure_ascii=False)}\n\n"
              "Format: ## Ringkasan, ## Struktur proyek, ## Cara menjalankan (hanya berdasarkan file yang ada, "
              "jangan mengarang file), ## Masalah yang perlu diperiksa (divisi gagal, error sintaks, celah antar hasil), "
              "## Langkah berikutnya. Jangan menambah fakta di luar data di atas.")
    report = await llm(prompt, max_tokens=4096)
    (ws / "REPORT.md").write_text(report, encoding="utf-8")
    await emit({"t": "done", "report": report, "ws": str(ws), "checks": checks,
                "files": [p.relative_to(ws).as_posix() for p in ws.rglob("*") if p.is_file()]})
    return {"report": report}


_g = StateGraph(State)
_g.add_node("plan", plan_node)
_g.add_node("execute", execute_node)
_g.add_node("compile", compile_node)
_g.add_edge(START, "plan")
_g.add_edge("plan", "execute")
_g.add_edge("execute", "compile")
_g.add_edge("compile", END)
GRAPH = _g.compile()

# ---------- API ----------
app = FastAPI(title="Kantor Kucing", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
RUNNING = {"on": False}


class RunReq(BaseModel):
    goal: str = Field(min_length=3, max_length=4000)
    context: str = Field(default="", max_length=1500)


@app.get("/api/health")
async def health():
    return {"ok": True, "model": MODEL}


@app.post("/api/run")
async def run(req: RunReq, x_token: str = Header(default="")):
    if not secrets.compare_digest(x_token.encode(), TOKEN.encode()):
        raise HTTPException(401, "Token salah")
    if RUNNING["on"]:
        raise HTTPException(409, "Masih ada proses yang berjalan")
    slug = re.sub(r"[^a-z0-9]+", "-", req.goal.lower())[:30].strip("-") or "proyek"
    ws = ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    ws.mkdir(parents=True)
    RUNNING["on"] = True

    async def stream():
        q = asyncio.Queue()

        async def emit(ev):
            await q.put(ev)

        async def runner():
            try:
                await GRAPH.ainvoke(
                    {"goal": req.goal, "context": req.context, "ws": str(ws)},
                    config={"configurable": {"emit": emit}})
            except Exception as e:
                await q.put({"t": "error", "error": str(e)[:400]})
            finally:
                await q.put(None)

        task = asyncio.create_task(runner())
        try:
            yield json.dumps({"t": "start", "ws": str(ws)}) + "\n"
            while (ev := await q.get()) is not None:
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        finally:
            task.cancel()
            RUNNING["on"] = False

    return StreamingResponse(stream(), media_type="application/x-ndjson")
