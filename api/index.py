import os
from server import app
from fastapi.middleware.cors import CORSMiddleware

# Force load environment variables for Vercel
# Although Vercel injects them, sometimes explicit loading helps
from dotenv import load_dotenv
load_dotenv()

# Ensure CORS is permissive if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This is the entrypoint that Vercel looks for
app = app 
