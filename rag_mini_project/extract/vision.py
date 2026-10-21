"""Gemini 비전으로 그림·표 이미지를 Markdown으로 옮긴다.

- 지시문은 대상별로 나눈다(figure: 그래프·그림, table: 규칙 검사에 실패한 표).
- 결과는 sha256(이미지 + 지시문 + 모델)로 cache/vision/에 저장해 다시 돌려도 비용이 들지 않는다.
- 모델은 .env의 VISION_MODEL(기본 gemini-3.8-flash).
"""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
for d in [ROOT, *ROOT.parents]:
    if (d / ".env").is_file():
        load_dotenv(d / ".env")
        break

MODEL = os.getenv("VISION_MODEL", "models/gemini-3.8-flash")
CACHE = ROOT / "cache" / "vision"

COMMON = """
공통 규칙:
- 요약·생략·해석하지 말고 보이는 그대로 옮긴다.
- 수치·단위·기호(%, %p, △, *, 주석 번호)는 그대로 보존한다.
- 읽을 수 없는 부분은 [확인 불가]로 쓰고 추측하지 않는다.
- 설명 문장이나 코드블록(```) 없이 Markdown 본문만 출력한다.
"""

PROMPTS = {
    "figure": """이 이미지는 한국어 금융 보도자료의 그림(그래프·도표·인포그래픽)이다. Markdown으로 옮겨라.
- 그래프 제목, 단위, 범례, 축 이름, 주석·출처를 그대로 쓴다.
- 그래프 값은 그림에 숫자 라벨로 적힌 값만 Markdown 표로 옮긴다. 막대 높이나 눈금으로 값을 짐작하지 않는다.
- 여러 그래프가 함께 있으면 그래프마다 제목과 표를 따로 쓴다.
- 도식·인포그래픽은 적힌 글과 숫자를 읽는 순서대로 옮긴다.
- 로고·장식처럼 정보가 없는 그림이면 '내용 없음'만 출력한다.""" + COMMON,
    "table": """이 이미지는 한국어 금융 보도자료의 표다. Markdown 표 하나로 옮겨라.
- 열 이름은 원문 그대로 쓴다.
- 머리글이 여러 줄이면 위에서 아래로 이어 붙여 한 줄 머리글로 만든다(예: '은행' 아래 '금액' → '은행 금액').
- 가로·세로로 병합된 칸은 걸친 모든 칸에 같은 값을 채운다.
- 왼쪽 위 빈 머리글은 '구분'으로 쓴다.
- 합친 열 이름이 서로 겹치면(예: '비중'이 여러 번) 바로 왼쪽 열의 하위 머리글을 붙여 구분한다(예: '예·적금 은행 비중', '예·적금 저은 비중').
- 위 첨자 주석 번호는 '은행1)'처럼 본문 글자로 쓰고 HTML 태그(<sup> 등)는 쓰지 않는다.
- 표 제목·단위·주석은 표에 넣지 않는다.""" + COMMON,
    # 쪽 전체 이미지에서 본문·일반 표가 아닌 시각 자료만 뽑는다(extract_p4l.py)
    "page_visual": """이 이미지는 한국어 금융 문서의 한 쪽 전체다. 본문 글과 일반 표는 이미 따로 추출했으니 무시하라.
표로 옮겨지지 않은 시각 자료(차트·그래프, 인포그래픽, 흐름도·구조도, 사진·일러스트)만 Markdown으로 옮겨라.
- 시각 자료마다 첫 줄을 `[graph] 제목 | box=[ymin, xmin, ymax, xmax]` 또는 `[image] 제목 | box=[ymin, xmin, ymax, xmax]`로 시작한다.
  graph: 수치를 나타내는 막대·선·원형·분포 차트
  image: 인포그래픽, 흐름도, 구조도, 도식, 사진, 일러스트
  제목: 그림 위나 옆에 적힌 제목·소제목을 그대로 쓴다(예: '1 퇴직연금 운용현황'). 없으면 내용을 짧게 쓴다.
  box: 그 시각 자료(제목·범례·주석 포함)를 감싸는 사각형. 이 쪽 이미지 기준 0~1000 정규화 좌표, 순서는 [ymin, xmin, ymax, xmax].
- graph: 단위·범례·축 이름·주석·출처를 쓰고, 그림에 숫자 라벨로 적힌 값만 Markdown 표로 옮긴다. 막대 높이나 눈금으로 값을 짐작하지 않는다.
- image: 적힌 글과 숫자를 읽는 순서대로 옮긴다. 흐름도·구조도는 화살표 방향을 'A → B'로 쓴다.
- 로고·아이콘·장식·배경 그림은 무시한다.
- 이런 시각 자료가 없으면 정확히 '차트 없음'만 출력한다.""" + COMMON,
}


def _key(png: bytes, kind: str) -> str:
    h = hashlib.sha256()
    for part in (png, PROMPTS[kind].encode(), MODEL.encode()):
        h.update(part)
    return h.hexdigest()


def describe(png: bytes, kind: str) -> str:
    path = CACHE / f"{_key(png, kind)}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"), PROMPTS[kind]],
                config=types.GenerateContentConfig(temperature=0),
            )
            text = (resp.text or "").strip()
            break
        except Exception as e:  # 일시적 오류(429·503 등)는 잠시 뒤 다시
            if attempt == 3:
                raise
            print(f"  vision retry ({e.__class__.__name__}: {str(e)[:80]})", flush=True)
            time.sleep(5 * (attempt + 1))
    text = text.removeprefix("```markdown").removeprefix("```").removesuffix("```").strip()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def describe_many(jobs: list[tuple[bytes, str]], workers: int = 5) -> list[str]:
    """여러 이미지를 동시에(기본 5개) 처리한다. 캐시에 있으면 호출하지 않는다."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: describe(*j), jobs))
