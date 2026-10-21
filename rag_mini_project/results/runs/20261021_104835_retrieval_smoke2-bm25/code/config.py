"""환경변수(.env)에서 설정을 읽는다.

.env는 이 파일에서 위로 올라가며 처음 찾은 것을 쓴다(지금은 rag-mini-project/.env).
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # rag_mini_project/


def _load_env() -> None:
    for d in [ROOT, *ROOT.parents]:
        if (d / ".env").is_file():
            load_dotenv(d / ".env")
            return


_load_env()


@dataclass(frozen=True)
class PgSettings:
    host: str
    port: int
    user: str
    password: str
    dbname: str

    def conninfo(self, dbname: str | None = None) -> dict:
        return dict(host=self.host, port=self.port, user=self.user,
                    password=self.password, dbname=dbname or self.dbname)


def pg_settings() -> PgSettings:
    return PgSettings(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["RAG_POSTGRES_DB"],
    )


MD_DIR = ROOT / "md"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
