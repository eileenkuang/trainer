from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routes import router

app = FastAPI()

app.include_router(router)

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/api/ping")
def ping():
    return {"message": "backend is alive"}
