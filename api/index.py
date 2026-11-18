from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Kantor Kucing Backend is running"}

@app.get("/api/health")
async def health():
    return {"ok": True}

# Import your actual app logic here if needed
# from server import app as actual_app
# ...
