# 문서 안내와 실행 방법

## 문서

| 문서 | 내용 |
|---|---|
| `experiment_plan.md` | 실험 계획: 평가 세트, 기준 설정(B0), 지표, 비교 실험(E1~E8) 가설, 최종 설정 선택 기준, 진행 순서 |
| `experiment_results.md` | 실험 결과: 검색 단계·답변 단계 표, 아이디어별 해석, 실패 사례 분석, 잠정 선택, 한계 |
| `experiment_results_emb2.md` | 같은 실험을 임베딩 gemini-embedding-2로 다시 돌린 결과와 001 비교(결과 `results_emb2/`, 표는 `rag/compare_embeddings.py`). 환경변수 `EMBEDDING_MODEL`·`CHROMA_DIR`·`RAG_RESULTS_DIR`만 바꾸면 같은 명령으로 재현 |
| `experiment_records.md` | 결과를 어떻게 저장·비교하는지: 폴더 구조, config·문항별 결과 스키마, 지표 정의, 비교 도구 |
| `agent_graph.md` | Agent(검색 재시도형) 그래프 구조도: 노드, 분기, 종료 조건, 요소 끄기(ablation) |
| `인수인계서.md` | **다른 AI에게 넘기는 문서**: 현재 상태, 남은 일, 사용자 선호, 환경, 함정 |
| `회고.md` | 겪은 문제와 개선 방안 |
| `progress.md` | **진행 기록**: 끝낸 것, 결론, 남은 것, 평가를 더 돌릴 때 |
| `answer_tables.md` | 답변 단계 설정별 결과 표(`python -m rag.report --md`로 생성) |
| `plan_extract_p4l_fixes.md` | PDF → md 추출 개선 계획(추출 단계 기록) |

## 코드 구성

| 파일 | 역할 |
|---|---|
| `extract/` | PDF → md 추출(pymupdf4llm, 표·그림은 vision 모델) |
| `rag/md_parser.py`, `rag/doc_chunking.py`, `rag/chunking.py` | md 읽기와 청킹 전략(Recursive, 구조 단위, parent-child, 제목 접두어, 표 보존) |
| `rag/build_index.py`, `rag/chroma_store.py`, `rag/pg_store.py` | 색인: 청크는 Chroma, parent 원문은 Postgres |
| `rag/retrievers.py`, `rag/search.py` | 검색(similarity, MMR, BM25, hybrid), 터미널 시험 |
| `rag/evaluate.py` | 검색 단계 평가 |
| `rag/rag.py`, `rag/agent.py`, `rag/llm.py` | 기본 RAG, Agent, Gemini 호출(재시도·차단 처리·LangSmith 추적) |
| `rag/judge.py`, `rag/answer_eval.py` | 답변 판정, 답변 단계 평가(RAG·Agent) |
| `rag/experiment.py` | 실험 기록 공통 코드 |
| `rag/compare.py`, `rag/report.py`, `rag/failure_analysis.py`, `rag/human_sheet.py` | 비교, 반복 평균 집계, 실패 분류, 사람 대조 시트 |

## 준비

- 상위 폴더 `.env`에 `GOOGLE_API_KEY`, `POSTGRES_*`, `RAG_POSTGRES_DB`, `CHROMA_DIR`, (선택) `LANGSMITH_*`를 둔다.
  Chroma는 경로에 한글이 있으면 1,000개 넘는 컬렉션의 인덱스를 못 쓰므로 `CHROMA_DIR`는 한글이 없는 경로(예: `C:/rag_data/chroma_db`)여야 한다.
- `pip install -r requirements.txt`, Postgres 실행.

## 색인 만들기 (컬렉션)

```bash
# 기준 B0: 문서 전체를 Recursive 500/50으로
python -m rag.build_index --doc-type standard_terms glossary --strategy recursive --collection base_rec_c500_o50
python -m rag.build_index --doc-type life_guide additional_p4l --strategy recursive --collection extended_rec_c500_o50
# 구조 단위 / parent-child / 제목 접두어 / 표 보존
python -m rag.build_index --doc-type standard_terms glossary --strategy structure --collection base_struct_c500_o50
python -m rag.build_index --doc-type standard_terms glossary --strategy parent-child --collection base_pc_c500_o50
python -m rag.build_index --doc-type standard_terms glossary --strategy recursive --title-prefix --collection base_rec_title_c500_o50
python -m rag.build_index --doc-type life_guide additional_p4l --strategy recursive --table-aware --collection extended_rec_table_c500_o50
```

## 평가 실행

```bash
# 검색 단계(기준 실험 → 비교 실험은 --baseline)
python -m rag.evaluate --name B0-base --collection base_rec_c500_o50 --methods similarity
python -m rag.evaluate --name E5-base --collection base_rec_c500_o50 --methods bm25 hybrid --baseline <B0 실험 ID>

# 답변 단계: 기본 RAG, Agent, k 변경, Agent 요소 끄기
python -m rag.answer_eval --name B0-base --collection base_rec_c500_o50
python -m rag.answer_eval --name E3-b-base --collection base_pc_c500_o50 --baseline <B0 실험 ID>
python -m rag.answer_eval --name E6-a-base --collection base_rec_c500_o50 --baseline <B0 실험 ID> --k 3
python -m rag.answer_eval --mode agent --name E7-base --collection base_rec_c500_o50 --baseline <B0 실험 ID>
python -m rag.answer_eval --mode agent --agent-options retry=0 --name E7a-base --collection base_rec_c500_o50 --baseline <B0 실험 ID>

# 시험 실행(앞 5문항, 정식 기록과 따로 저장)
RAG_RESULTS_DIR=/tmp/trial python -m rag.answer_eval --name smoke --collection base_rec_c500_o50 --limit 5
```

답변 평가 한 번은 기본 세트 약 7분, 확장 세트 약 6분(문항 6개 동시 처리)이다. 판정 대기가 대부분이라 `--workers 16`으로 올리고 여러 평가를 동시에 돌리면 더 빠르다(응답 시간 열은 동시 처리 수가 다르면 비교할 수 없다). 문항 하나가 15분 안에 끝나지 않으면 오류로 기록하고 넘어간다.

## 결과 보기

```bash
python -m rag.report                      # 설정별 평균과 [최소–최대]
python -m rag.report --runs               # 설정별 실험 ID
python -m rag.compare --baseline-of <ID>  # 기준 대비 차이
python -m rag.failure_analysis <ID>       # 실패 원인 분류
python -m rag.human_sheet make <기본 B0 ID> <확장 B0 ID>   # 사람 대조 시트
python -m rag.human_sheet compare         # 채점한 시트와 판정 모델의 일치율
```
