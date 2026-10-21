# 진행 기록 (2026-10-21 21:46)

**계획한 평가를 모두 마쳤다.** 돌고 있는 작업은 없다. 남은 것은 사용자 확인 두 가지와 후속 실험 후보뿐이다.

## 1. 끝낸 것

| 단계 | 내용 | 산출물 |
|---|---|---|
| 데이터 | PDF → md 추출, 질문·답안(180문항), 질문 유형 라벨 초안 | `md/`, `03_answers/`, `eval_labels/question_types.csv` |
| 청킹·색인 | Recursive, 구조 단위, parent-child, 제목 접두어, 표 보존. Chroma 컬렉션 20개 | `rag/doc_chunking.py`, `rag/build_index.py`, `C:/rag_data/chroma_db`(`.env`의 `CHROMA_DIR`) |
| 검색 평가 | 기준 B0와 E1~E6. 실행 35개 | `rag/evaluate.py`, `results/runs/*_retrieval_*` |
| 기본 RAG·판정 | 검색 1번 → 답변 1번, 판정 모델(`gemini-2.5-pro`)이 항목별 채점 | `rag/rag.py`, `rag/judge.py` |
| 답변 평가 | B0, E3, E4, E6, 합친 설정 E8, 반복. 실행 33개 | `rag/answer_eval.py`, `results/runs/*_rag_*` |
| Agent | 검색 재시도형 구현, 구조도, 평가(전체·재검색 끔·분석 끔, 두 세트). 실행 8개 | `rag/agent.py`, `docs/agent_graph.md`, `results/runs/*_agent_*` |
| 분석 도구 | 비교, 반복 평균 집계, 실패 원인 분류, 사람 대조 시트 | `rag/compare.py`, `rag/report.py`, `rag/failure_analysis.py`, `rag/human_sheet.py` |
| 문서 | 계획, 결과, 기록 설계, Agent 구조도, 실행 방법, 회고 | `docs/` 아래 |

실험 기록: `results/experiments.csv`(76행), 설정별 실험 ID는 `python -m rag.report --runs`.

## 2. 결론

상세는 `docs/experiment_results.md`(0절 결론, 5절 최종 선택).

- **청킹이 가장 크다.** 구조 단위 parent-child(E3-b)가 기본 세트 pass 70.3 → 80.5%, 확장 세트 64.6 → 71.9%로 오차를 넘어 좋아졌다. 토큰은 1.3배, 2.5배.
- 검색 방식, 제목 접두어, 표 보존은 이득이 없었다. k=3은 확장 세트에서 나빠졌다.
- parent-child 위에서 k=10은 추가 이득이 없고(비용 1.8~1.9배), k=3은 기본 세트에서만 비용 절감 대안이다(pass 76.5%, 토큰 0.85배).
- **Agent는 채택하지 않는다.** 기본 세트에서 +7.7%p지만 비용 1.9배이고 E3-b(+10.2%p, 1.3배)가 더 효율적이다. 확장 세트에서는 효과가 없다(−0.8%p, 비용 2.0배). 효과는 질문 분석·분기에서 나오고 재검색은 비용만 늘린다.
- **최종 선택:** 두 세트 모두 E3-b(k=5). 비용을 줄이려면 기본 세트에서 E8a(k=3).
- 평가가 흔들린다(같은 설정의 pass 5~6%p). 5%p 안쪽 pass 차이는 무시한다.

## 3. 남은 것

**사용자 확인**
- 판정 모델의 사람 대조: `eval_labels/human_check_sheet.xlsx`(20문항) 채점 후 `python -m rag.human_sheet compare`. 이게 끝나야 판정 지표(특히 근거 준수)의 신뢰도를 안다.
- 질문 유형 라벨 검수: `eval_labels/question_types.csv`(`review` 열).

**후속 실험 후보 (계획 밖, 미검증)**
- E3-b + Agent `retry=0`: parent-child는 답할 수 있는 질문에, 질문 분석은 되묻기·거절에 강해 약점이 반대다.
- 프롬프트 개선: 근거에 있는 부분은 답하고 없는 부분만 보류, 예시 수치의 한계 밝히기, 피해 상황이면 다음 행동 안내(실패 분석 4절).
- 응답 시간을 따로 재기(설정당 20문항을 한 개씩 순서대로).

**확인할 것**
- README는 '제공 세트를 참고해 직접 작성·검수한 골든셋'을 요구하는데 이 프로젝트는 제공된 180문항만 썼다. 제출 전에 필요한지 확인한다.
- 코드와 문서는 **마지막 커밋(`67ca3ce`) 이후 커밋하지 않았다.** 원격에는 `v0.1.0`까지만 올라가 있다.

## 4. 평가를 더 돌릴 때

준비: Postgres 실행, 상위 폴더 `.env`(API 키, DB, `CHROMA_DIR`) 확인. 명령은 `rag_mini_project/`에서 실행한다.

- 평가 시간은 판정 모델 대기가 대부분이라 `--workers 16`으로 올리고 여러 평가를 터미널 여러 개에서 동시에 돌린다(평가 하나가 약 3분). 동시 처리 수가 다르면 `응답 시간` 열은 비교할 수 없다.
- 명령 형식과 컬렉션 이름은 `docs/README.md`에 있다. 반복은 이름 끝에 `-rep2`처럼 붙이면 `python -m rag.report`가 한 설정으로 묶는다.
- 끝난 뒤 `python -m rag.report --md docs/answer_tables.md`로 표를 다시 만들고 `docs/experiment_results.md`를 갱신한다.

## 5. 알아둘 점

- 평가 중 하나가 응답 없이 멈추면 15분 뒤 오류로 기록하고 넘어간다(문항별 제한 시간). 일괄 실행 때는 새로 끝나는 간격을 감시한다.
- 시험 실행은 `RAG_RESULTS_DIR=<폴더>`로 따로 돌려 정식 기록과 섞이지 않게 한다.
- 오류가 있던 실행 3개는 고친 뒤 다시 돌려 이전 실행을 지웠다(`docs/experiment_results.md` 6절). 병렬 실행 사고로 E4-b·E6-a·E6-b 확장 세트는 두 번씩 있다.
- LangSmith 월간 추적 한도를 넘겨 `.env`의 `LANGSMITH_TRACING=false`로 껐다. 그 뒤 실행의 `trace.langsmith_run_id`는 비어 있고, 한도에 걸리기 전후 어느 실행부터 조회되지 않는지는 확인하지 못했다.
- 평가를 시작하기 전에 총 소요 시간을 계산한다(설정 수 × 실행당 시간 ÷ 동시 처리 수).
