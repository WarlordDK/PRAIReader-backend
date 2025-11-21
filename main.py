from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import router as router_analyze
import uvicorn

app = FastAPI()
app.include_router(router_analyze)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def home_page():
    return {'message' : 'uvicorn running'}

if __name__ == "__main__":
    uvicorn.run(app=app, host='127.0.0.1', port=8000)
