from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth

app = FastAPI(
    title="Biometric Auth API",
    description="Backend de autenticação por biometria facial com verificação anti-deepfake.",
    version="1.0.0",
)

# Em produção, restrinja allow_origins ao domínio/app real em vez de "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Autenticação"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
