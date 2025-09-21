"""Main UI routes for Tag Sentinel web interface."""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Initialize Jinja2 templates
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Create UI router
ui_router = APIRouter(
    prefix="/ui",
    tags=["Web Interface"],
    responses={404: {"description": "Not found"}},
)


@ui_router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view."""
    context = {
        "request": request,
        "title": "Dashboard",
        "page": "dashboard"
    }
    return templates.TemplateResponse("dashboard.html", context)


@ui_router.get("/runs", response_class=HTMLResponse)
async def runs_list(request: Request):
    """Runs list view."""
    context = {
        "request": request,
        "title": "Audit Runs",
        "page": "runs"
    }
    return templates.TemplateResponse("runs.html", context)


@ui_router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    """Run detail view with reports."""
    context = {
        "request": request,
        "title": f"Run {run_id}",
        "page": "run_detail",
        "run_id": run_id
    }
    return templates.TemplateResponse("run_detail.html", context)


@ui_router.get("/runs/{run_id}/pages/{page_id}", response_class=HTMLResponse)
async def page_detail(request: Request, run_id: str, page_id: str):
    """Page detail view."""
    context = {
        "request": request,
        "title": f"Page Detail",
        "page": "page_detail",
        "run_id": run_id,
        "page_id": page_id
    }
    return templates.TemplateResponse("page_detail.html", context)


@ui_router.get("/health", response_class=HTMLResponse)
async def health_status(request: Request):
    """System health status view."""
    context = {
        "request": request,
        "title": "System Health",
        "page": "health"
    }
    return templates.TemplateResponse("health.html", context)