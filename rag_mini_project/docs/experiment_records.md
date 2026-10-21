# 실험 결과 기록 설계

실험을 돌리기 전에 무엇을, 어떤 형식으로 저장하고, 어떻게 비교할지 정한다.
검색 평가(`rag/evaluate.py`), 기본 RAG, Agent가 모두 같은 형식을 쓴다. 공통 코드는 `rag/experiment.py`에 있다.

## 원칙

1. **실험 ID 하나 = 설정 하나.** 검색기 4개를 한 번에 돌리면 실험 ID가 4개 생기고, 이들을 `sweep_id`로 묶는다.
2. **실험 ID로 모든 기록이 연결된다.** 설정(`config.json`), 문항별 결과(`per_case.jsonl`), 요약(`summary.json`), 실험 목록(`experiments.csv`)이 모두 `experiment_id`를 가진다.
3. **기준 대비 변경은 사람이 적지 않고 자동으로 뽑는다.** `--baseline <실험ID>`를 주면 두 config의 `settings`를 비교해 달라진 항목만 `changes`에 남긴다. 사람이 쓰는 `note`는 의도·가설용이다.
4. **재실행에 필요한 것은 모두 남긴다.** 실행 명령, 설정 전체, 프롬프트 원문, 코드 버전(git 커밋 해시), 데이터 파일 해시, 패키지 버전.
   평가할 때마다 커밋할 필요는 없다. 커밋하지 않은 변경이 있으면 해시 뒤에 `-dirty`가 붙는다.
5. **파일로 저장한다(JSON/JSONL/CSV).** DB 없이 열어 보고 diff할 수 있고 제출하기 쉽다.

## 폴더 구조

```
results/
  experiments.csv                  실험 목록: 실험당 한 줄(핵심 설정 + 기준 대비 변경 + 대표 지표)
  runs/<experiment_id>/
    config.json                    설정 전체(아래 스키마)
    per_case.jsonl                 문항별 실행 결과와 점수(한 줄 = 한 문항)
    summary.json, summary.csv      전체·그룹별 집계
    prompts/<name>.txt             이 실험에서 쓴 프롬프트 원문(rag·agent)
```

실험 ID 형식: `YYYYMMDD_HHMMSS_<stage>_<이름>` (예: `20261021_103000_retrieval_base-hybrid`).
sweep ID 형식: `YYYYMMDD_HHMMSS_<이름>` (한 번의 실행 명령으로 만든 실험들의 묶음).

## config.json

```jsonc
{
  "experiment_id": "...", "sweep_id": "...", "stage": "retrieval | rag | agent",
  "created_at": "2026-10-21T10:30:00",
  "command": "python -m rag.evaluate --collection base_c350_o50 ...",
  "baseline_id": "기준 실험 ID 또는 null",
  "changes": [{"path": "retrieval.method", "baseline": "similarity", "value": "hybrid"}],
  "note": "의도·가설(사람이 씀)",

  "settings": {                     // 결과에 영향을 주는 설정. changes는 이 안만 비교한다
    "data": {                       // 질문 세트와 검색 대상
      "answer_files": [...], "case_filter": "...", "n_cases": 68,
      "doc_types": [...], "source_doc_ids": [...]
    },
    "chunking": {                   // 청킹 전략
      "collection": "base_c350_o50", "strategy": "parent-child | flat",
      "chunk_size": 350, "chunk_overlap": 50, "separators": [...], ...
    },
    "embedding": {"model": "models/gemini-embedding-001"},
    "retrieval": {                  // 검색 설정(그 방식에 실제로 쓰이는 인자만)
      "method": "hybrid",
      "k": 5,                       // 컨텍스트에 넣는 결과 수
      "depth": 10,                  // 지표 계산용으로 받아 두는 결과 수(eval.ks 최댓값)
      "fetch_k": 40, "hnsw_ef_search": 1000, "weight_vector": 0.5,
      "result_unit": "parent | chunk", "bm25": {...}, "fusion": "RRF", "rrf_k": 60
    },
    "generation": {                 // rag·agent만
      "model": "모델 ID", "model_version": "응답에 찍힌 버전",
      "temperature": 0, "max_output_tokens": 1024,
      "prompts": {"system": {"file": "prompts/system.txt", "sha256": "..."}}
    },
    "agent": {                      // agent만
      "graph": "노드·분기 이름", "max_iterations": 3, "stop_conditions": [...]
    },
    "eval": {"ks": [1, 3, 5, 10], "relevance_unit": "...", "labels": "eval_labels/question_types.csv",
             "source_coverage": "정의", "key_phrase_recall": "정의", "dup_rate": "정의",
             "judge": {"model": "...", "prompt": {...}}}      // judge는 rag·agent만
  },

  "env": {                          // 재현 정보(결과 해석보다는 재실행용, changes에 넣지 않음)
    "python": "3.13.x", "packages": {"chromadb": "1.5.9", ...},
    "code_version": "143b2e9",                       // git 커밋 해시, 커밋 안 한 변경이 있으면 "143b2e9-dirty"
    "data_files": {"03_answers/positive_cases.jsonl": "sha256", "md/glossary/...md": "sha256"},
    "collection_count": 1036
  },
  "env_changes": ["기준 실험과 다른 코드 버전·데이터 파일"]   // baseline이 있을 때
}
```

## per_case.jsonl (한 줄 = 한 문항)

| 묶음 | 필드 | 단계 |
|---|---|---|
| 연결 | `experiment_id`, `sweep_id`, `case_id` | 공통 |
| 문항 | `question`, `set`, `category`, `difficulty`, `expected_behavior`, `expected_source_ids`, `excluded_source_ids`, `source_doc`, `question_type`, `needs_table`, `must_include`, `answer_file` | 공통 |
| 검색 | `retrieved[]`: `rank`, `source_id`, `parent_id`, `chunk_ids`, `score`, `source_file`, `page_start`, `page_end`, `relevant`, `content`(컨텍스트에 넣은 원문) | 공통 |
| 검색 점수 | `metrics`: `hit@k`, `recall@k`, `all_hit@k`, `precision@k`, `source_coverage@k`, `key_phrase_recall@k`, `dup_rate@k`, `context_chars@k`, `mrr`, `first_relevant_rank`, `key_phrases_checked` | 공통 |
| 답변 | `answer`, `citations`(파일명·쪽), `answerable`(근거 판단), `final_action`(답변·확인·보류·거절) | rag·agent |
| 답변 점수 | `answer_metrics`: 정답 포함(`must_include`), 금지어(`must_not_include`), 근거 준수, 판정 점수 | rag·agent |
| 비용 | `usage`: `llm_calls`, `retrieval_calls`, `input_tokens`, `output_tokens` | rag·agent |
| 시간 | `latency_ms` (검색만 / 전체), 단계별 시간 | 공통 |
| 추적 | `trace`: `langsmith_project`, `langsmith_run_id` | rag·agent |
| 오류 | `error` (없으면 null) | 공통 |

검색 점수와 답변 점수를 따로 두어 **검색에서 근거를 놓친 경우**(`metrics.hit@k = 0`)와
**근거를 찾았지만 답을 잘못 쓴 경우**(`hit@k = 1`인데 답변 점수가 낮음)를 구분한다.

## 지표 기준

- 검색 지표는 **실제 컨텍스트에 넣는 상위 k개 결과**로 계산한다.
  - parent-child(약관·용어집): 결과 하나 = parent 하나.
  - parent 없음(가이드북·추가 문서): 결과 하나 = 청크 하나. 같은 쪽 청크 두 개면 두 자리.
- `precision@k`는 자리 단위, `recall@k`·`hit@k`·`all_hit@k`는 정답 근거 ID 집합 기준.
- `source_coverage@k`: 정답 근거 원문(조항·용어·쪽, `rag/sources.py`) 중 컨텍스트에 들어간 글자 비율.
  컨텍스트를 줄 단위로 원문에서 찾아 표시한다(5자 미만 줄 제외). 근거가 잘렸는지 보는 지표다.
- `key_phrase_recall@k`: 기준 답안 `must_include` 중 정답 근거 원문에 **그대로 있는** 구절만 골라 컨텍스트 포함률을 본다.
  그런 구절이 없는 문항은 평균에서 뺀다(사실상 기본 positive 세트만 계산됨). 요약에 `n_key_phrase_cases`를 남긴다.
- `dup_rate@k`: 상위 k개 중 같은 근거 ID가 겹친 비율. `context_chars@k`: 컨텍스트 글자 수.
- 대표 지표는 컨텍스트 k(`settings.retrieval.k`, 기준 5) 기준이고 `experiments.csv`에 `ctx_*` 열로 둔다.
- 검색은 HNSW `ef_search` 1000으로 고정해 전수 검색과 같은 결과를 낸다. 기본값 100에서는 MMR 후보가 실행마다 달라져
  같은 설정으로도 지표가 바뀌었다(2026-10-21 확인).
- 문항 세트가 다르면 지표를 비교하지 않는다. `experiments.csv`의 `n_cases`·`answer_files`로 확인한다.

## summary.json / summary.csv

그룹(전체, difficulty, set, category, 근거 수, source_doc, question_type, needs_table)마다 문항 수, 지표 평균, 시간(mean·p50·p95·max)을 담는다.
rag·agent는 여기에 답변 점수 평균과 비용 합계·평균(호출 수, 검색 횟수, 토큰)을 더한다.

## experiments.csv (실험 비교표)

실험당 한 줄. 열: `experiment_id`, `sweep_id`, `stage`, `created_at`, `baseline_id`, `changes`(예: `retrieval.method: similarity→hybrid`),
`env_changes`(기준과 다른 코드 버전·데이터 파일 수), `note`, `collection`, `strategy`, `chunk_size`, `chunk_overlap`,
`retrieval_method`, `k`, `fetch_k`, `weight_vector`, `lambda_mult`, `generation_model`, `n_cases`, `answer_files`,
대표 지표 `ctx_hit`·`ctx_recall`·`ctx_precision`·`ctx_source_coverage`·`ctx_key_phrase_recall`·`ctx_dup_rate`·`ctx_context_chars`·`mrr`,
k별 전체 지표, 시간(rag·agent는 비용 합계 추가).

실험 계획(기준 설정, 비교 실험 목록, 가설)은 `docs/experiment_plan.md`에 있다.

## 비교 방법

1. 기준 실험을 먼저 돌리고 그 실험 ID를 기억한다.
2. 비교 실험은 `--baseline <기준 ID>`로 돌린다. `changes`에 한 항목만 있어야 단일 요소 비교다.
   두 개 이상이면(예: 기본 RAG와 Agent) 전체 구성의 효과로 해석한다.
3. `experiments.csv`에서 같은 `answer_files`·`n_cases` 줄끼리 지표를 비교한다.
4. 차이가 난 문항은 두 실험의 `per_case.jsonl`을 `case_id`로 맞춰 본다.
