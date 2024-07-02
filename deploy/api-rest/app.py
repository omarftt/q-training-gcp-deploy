from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from routes.router import router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["DELETE", "GET", "POST", "PUT"],
    allow_headers=["*"],
)

app.include_router(router,prefix="/api",tags=["user"])