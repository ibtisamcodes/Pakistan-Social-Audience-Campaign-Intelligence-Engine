from pathlib import Path
from io import BytesIO
import csv
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from .analysis import analyze_comments, analyze_posts

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Audience Intelligence Engine")
app.add_middleware(SessionMiddleware, secret_key="change-this-secret-key")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

# In-memory demo storage. Uploaded CSVs are retained in the current server process.
COMMENTS = pd.DataFrame()
POSTS = pd.DataFrame()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    result = analyze_comments(COMMENTS) if not COMMENTS.empty else None
    post_result = analyze_posts(POSTS) if not POSTS.empty else None
    return templates.TemplateResponse("index.html", {
        "request": request, "result": result, "post_result": post_result,
        "comment_count": len(COMMENTS), "post_count": len(POSTS),
        "has_comments": not COMMENTS.empty, "has_posts": not POSTS.empty
    })

@app.post("/upload/comments")
async def upload_comments(file: UploadFile = File(...)):
    global COMMENTS
    try:
        data = pd.read_csv(file.file, encoding="utf-8-sig")
        required = {"post_id", "comment_text"}
        missing = required - set(data.columns)
        if missing:
            return RedirectResponse("/?error=Comments CSV missing: " + ",".join(sorted(missing)), status_code=303)
        data["comment_text"] = data["comment_text"].fillna("").astype(str)
        data = data[data["comment_text"].str.strip() != ""].copy()
        if data.empty:
            return RedirectResponse("/?error=No usable comment_text rows found", status_code=303)
        COMMENTS = data.reset_index(drop=True)
        return RedirectResponse("/", status_code=303)
    except Exception as exc:
        return RedirectResponse("/?error=Could not read CSV: " + str(exc), status_code=303)

@app.post("/upload/posts")
async def upload_posts(file: UploadFile = File(...)):
    global POSTS
    try:
        data = pd.read_csv(file.file, encoding="utf-8-sig")
        required = {"post_id", "brand", "platform", "content_type", "likes", "comments_count"}
        missing = required - set(data.columns)
        if missing:
            return RedirectResponse("/?error=Posts CSV missing: " + ",".join(sorted(missing)), status_code=303)
        for col in ["likes", "comments_count", "shares", "views", "days_since_post"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        POSTS = data.reset_index(drop=True)
        return RedirectResponse("/", status_code=303)
    except Exception as exc:
        return RedirectResponse("/?error=Could not read CSV: " + str(exc), status_code=303)

@app.get("/sample/comments.csv")
async def sample_comments():
    path = ROOT.parent / "sample_data" / "comments_template.csv"
    return StreamingResponse(open(path, "rb"), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="comments_template.csv"'})

@app.get("/sample/posts.csv")
async def sample_posts():
    path = ROOT.parent / "sample_data" / "posts_template.csv"
    return StreamingResponse(open(path, "rb"), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="posts_template.csv"'})
