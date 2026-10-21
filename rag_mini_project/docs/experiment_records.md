# 실험 결과 기록 설계

무엇을, 어떤 형식으로 저장하고, 어떻게 비교할지 정한다. 검색 평가(`rag/evaluate.py`), 기본 RAG·Agent 답변 평가(`rag/answer_eval.py`)가
같은 형식을 쓴다. 공통 코드는 `rag/experiment.py`에 있다. 실험 계획은 `docs/experiment_plan.md`, 결과는 `docs/experiment_results.md`.

## 원칙

1. **실험 ID 하나 = 설정 하나.** 검색기 4개를 한 번에 돌리면 실험 ID가 4개 생기고, 이들을 `sweep_id`로 묶는다.
2. **실험 ID로 모든 기록이 연결된다.** 설정(`config.json`), 문항별 결과(`per_case.jsonl`), 요약(`summary.json`), 실험 목록(`experiments.csv`)이 모두 `experiment_id`를 가진다.
3. **기준 대비 변경은 사람이 적지 않고 자동으로 뽑는다.** `--baseline <실험ID>`를 주면 두 config의 `settings`를 비교해 달라진 항목만 `changes`에 남긴다. 사람이 쓰는 `note`는 의도·가설용이다.
4. **재실행에 필요한 것은 모두 남긴다.** 실행 명령, 설정 전체, 프롬프트 원문, 코드 버전(git 커밋 해시), 데이터 파일 해시, 패키지 버전.
   평가할 때마다 커밋할 필요는 없다. 커밋하지 않은 변경이 있으면 해시 뒤에 `-dirty`가 붙는다.
5. **파일로 저장한다(JSON/JSONL/CSV).** DB 없이 열어 보고 diff할 수 있고 제출하기 쉽다.
6. **같은 설정을 여러 번 돌린 실행은 이름으로 묶는다.** 이름 끝의 `-r2`, `-rep2`가 반복이다. 설정 하나의 값은 평균과 범위로 본다
   (`python -m rag.report`). 같은 설정도 실행마다 달라지기 때문이다(결과 문서 3.1절).
7. **시험 실행은 정식 기록과 따로 둔다.** `RAG_RESULTS_DIR=<폴더>`를 주면 `results/` 대신 그 폴더에 쓴다.

## 폴더 구조

```
results/
  experiments.csv                  실험 목록: 실험당 한 줄(핵심 설정 + 기준 대비 변경 + 대표 지표)
  runs/<experiment_id>/
    config.json                    설정 전체(아래 스키마)
    per_case.jsonl                 문항별 실행 결과와 점수(한 줄 = 한 문항)
    summary.json, summary.csv      전체·그룹별 집계
    prompts/<name>.txt             이 실험에서 쓴 프롬프트 원문(rag·agent)
eval_labels/
  question_types.csv               질문 유형 라벨(term/situation, 표 질문 여부)
  human_check_sheet.xlsx           판정 모델의 사람 대조 채점 시트(20문항)
```

실험 ID 형식: `YYYYMMDD_HHMMSS_<stage>_<이름>-<검색 방식>` (예: `20261021_142853_rag_B0-base-similarity`).
sweep ID 형식: `YYYYMMDD_HHMMSS_<이름>` (한 번의 실행 명령으로 만든 실험들의 묶음).

## config.json

```jsonc
{
  "experiment_id": "...", "sweep_id": "...", "stage": "retrieval | rag | agent",
  "created_at": "2026-10-21T14:28:53",
  "command": "python -m rag.answer_eval --name B0-base --collection base_rec_c500_o50 ...",
  "baseline_id": "기준 실험 ID 또는 null",
  "changes": [{"path": "retrieval.k", "baseline": 5, "value": 3}],
  "note": "의도·가설(사람이 씀)",

  "settings": {                     // 결과에 영향을 주는 설정. changes는 이 안만 비교한다
    "data": {                       // 질문 세트와 검색 대상
      "answer_files": [...], "case_filter": "...", "n_cases": 100, "n_cases_with_sources": 68,
      "doc_types": [...], "source_doc_ids": [...]
    },
    "chunking": {                   // 청킹 전략(build_index가 컬렉션 메타데이터에 남긴 값)
      "strategy": "recursive | structure | parent-child | page", "unit": "문서 전체 | 약관=조항, …",
      "splitter": "RecursiveCharacterTextSplitter", "chunk_size": 500, "chunk_overlap": 50, "separators": [...],
      "title_prefix": false, "table_aware": false, "result_unit": "chunk | parent"
    },
    "embedding": {"model": "models/gemini-embedding-001"},
    "retrieval": {                  // 검색 설정. 방식과 상관없이 모든 인자를 항상 넣는다(방식만 바꿔도 changes에 method 하나만 잡히게)
      "method": "similarity", "k": 5, "depth": 5, "fetch_k": 40, "hnsw_ef_search": 1000,
      "lambda_mult": 0.5, "mmr_candidate_pool": "2 x fetch_k", "weight_vector": 0.5, "rrf_k": 60, "bm25": {...}
    },
    "generation": {                 // rag·agent만. Agent는 같은 자리에 아래 항목이 더 붙는다
      "mode": "rag | agent", "model": "models/gemini-3.8-flash", "temperature": 0.0, "thinking_budget": 0,
      "max_output_tokens": 1500, "prompt_version": "v1", "context_format": "[근거 N] 파일명 p.쪽 + 원문(제목 없음)",
      "response_format": "JSON(answer, citations)",
      "prompts": {"generation_system": {"file": "prompts/generation_system.txt", "sha256": "..."}},
      // Agent 추가: "options": {"analyze": 1, "retry": 2, "route": 1}, "graph", "nodes", "max_retries",
      //   "stop_conditions", "step_max_output_tokens", "merge", prompts에 agent_analyze·agent_grade·agent_hints
    },
    "eval": {"ks": [5], "relevance_unit": "...", "labels": "eval_labels/question_types.csv",
             "judge": {"model": "models/gemini-2.5-pro", "temperature": 0.0, "max_output_tokens": 16000,
                       "prompt": {"file": "prompts/judge.txt", "sha256": "..."}, "sees_expected_behavior": false},
             "behavior_groups": {...}, "pass": "정의", "citation": "정의"}   // judge 이하는 rag·agent만
  },

  "collection_stats": {             // 설정에 따라 정해지는 값이라 changes에 넣지 않음
    "collection": "base_rec_c500_o50", "n_chunks": 510, "chunk_chars_mean": 380.2,
    "chunks_over_2plus_sources": 285, "n_parents": "(parent-child만)"
  },
  "env": {                          // 재현 정보(changes에 넣지 않음)
    "python": "3.13.x", "packages": {"chromadb": "1.5.9", ...},
    "code_version": "143b2e9",                       // git 커밋 해시, 커밋 안 한 변경이 있으면 "143b2e9-dirty"
    "data_files": {"03_answers/positive_cases.jsonl": "sha256", "md/glossary/...md": "sha256"}
  },
  "env_changes": ["기준 실험과 다른 코드 버전·데이터 파일"]   // baseline이 있을 때
}
```

## per_case.jsonl (한 줄 = 한 문항)

| 묶음 | 필드 | 단계 |
|---|---|---|
| 연결 | `experiment_id`, `sweep_id`, `case_id` | 공통 |
| 문항 | `question`, `set`, `category`, `difficulty`, `expected_behavior`, `expected_group`(5종), `expected_source_ids`, `excluded_source_ids`, `source_doc`, `question_type`, `needs_table`, `must_include`, `must_not_include`, `answer_file` | 공통 |
| 검색 결과 | `retrieved[]`: `rank`, `source_ids`(결과가 걸친 근거 ID 전부), `parent_id`, `chunk_ids`, `score`, `source_file`, `heading`, `page_start`, `page_end`, `relevant`, `content`(컨텍스트에 넣은 원문. 검색 평가는 k개까지만 저장) | 공통 |
| 검색 점수 | 검색 평가는 `metrics`, 답변 평가는 `retrieval_metrics`(정답 근거가 있는 문항만): `hit@k`, `recall@k`, `all_hit@k`, `precision@k`, `source_coverage@k`, `key_phrase_recall@k`, `dup_rate@k`, `context_chars@k`, `mrr`, `first_relevant_rank` | 공통 |
| 답변 | `answer`, `citations`(인용한 근거 번호), `cited_sources`(번호 → 파일명·쪽·근거 ID), `blocked`(API 차단 사유, 없으면 없음) | rag·agent |
| 판정 | `judge`: `model_version`, `latency_ms`, `usage`, `output`(must_include·must_not_include 항목별 판정, claims 근거 뒷받침 여부, behavior, note). 차단 문항은 없음 | rag·agent |
| 답변 점수 | `answer_metrics`: `must_include_rate`, `must_include_all`, `forbidden_violation`, `faithfulness`, `unsupported_claims`, `n_claims`, `behavior`, `behavior_match`, `coarse_match`, `cited_any`, `invalid_citations`, `citation_precision`, `citation_recall`, `judge_incomplete`, `blocked`, `pass` | rag·agent |
| 비용 | `usage`: `llm_calls`, `retrieval_calls`, `input_tokens`, `output_tokens`, `thinking_tokens`(판정 호출은 제외, `judge.usage`에 따로) | rag·agent |
| 시간 | 검색 평가는 `latency_ms`, 답변 평가는 `latency`: `retrieval_ms`, `generation_ms`, `total_ms`(판정 시간 제외) | 공통 |
| Agent | `steps[]`: 노드별 기록(analyze의 request_type·검색어, retrieve, grade의 충분 여부·새 검색어, answer의 처리 지침) | agent |
| 추적 | `trace`: `langsmith_project`, `langsmith_run_id` | rag·agent |
| 오류 | `error`(없으면 null). 오류 문항은 지표에서 빠지고 `n_errors`로 센다 | 공통 |

검색 점수와 답변 점수를 따로 두어 **검색에서 근거를 놓친 경우**(`hit@k = 0`)와 **근거를 찾았지만 답을 잘못 쓴 경우**
(`hit@k = 1`인데 답변 점수가 낮음)를 구분한다. 원인 분류는 `python -m rag.failure_analysis <실험 ID>`.

## 지표 기준

**검색 지표**
- 실제 컨텍스트에 넣는 **상위 k개 결과**로 계산한다. parent-child는 결과 하나 = parent 하나, parent 없음은 결과 하나 = 청크 하나.
- 결과가 근거 여러 개에 걸치면(문서 전체를 자른 청크) 하나라도 정답이면 정답 자리(hit·precision), 걸친 정답은 모두 찾은 것(recall), 앞 자리에서 이미 나온 근거로만 된 자리는 중복(dup_rate)이다.
- `source_coverage@k`: 정답 근거 원문(`rag/sources.py`) 중 컨텍스트에 들어간 글자 비율(줄 단위 매칭, 5자 미만 줄 제외).
- `key_phrase_recall@k`: `must_include` 중 정답 근거 원문에 **그대로 있는** 구절의 컨텍스트 포함률. 그런 구절이 없는 문항은 평균에서 뺀다.
- `dup_rate@k`, `context_chars@k`(컨텍스트 글자 수). 대표 지표는 컨텍스트 k(기준 5)이고 `experiments.csv`에 `ctx_*` 열로 둔다.
- HNSW `ef_search`를 1000으로 고정해 전수 검색과 같은 결과를 낸다. 기본값 100에서는 MMR 후보가 실행마다 달라졌다.

**답변 지표** (판정 모델이 문항마다 채점, 기대 행동은 보여 주지 않는다)
- `must_include_rate`: 기준 답안의 핵심 항목 중 답변에 담긴 비율(의미가 같으면 인정). `forbidden_violation`: 금지 항목을 주장했는가.
- `faithfulness`: 답변의 사실 주장 중 제공한 근거로 뒷받침되는 비율. `unsupported_claims`: 뒷받침되지 않는 주장 수.
- `behavior`: 답변·정정·되묻기·보류·거절 5종 중 답변의 중심 행동. 기대 행동 22종을 5종으로 묶어 `behavior_match`로 비교하고,
  답변 계열(답변·정정)과 보류 계열(되묻기·보류·거절)만 보는 `coarse_match`도 낸다. 질문에 잘못된 전제가 섞인 문항은 기대가 '답변'인데 모델이 '정정'으로 답해 `behavior_match`가 낮게 나오므로 확장 세트는 `coarse_match`로 본다.
- 인용 정확도: 인용한 근거 번호를 검색 결과의 근거 ID로 바꿔 정답 근거와 비교한다(`citation_precision`, `citation_recall`).
- **`pass`**: 핵심 항목을 모두 담고 AND 금지 내용이 없고 AND `coarse_match`가 1. 엄격한 지표라 흔들림이 크고, 설정 비교는 `must_include_rate`를 먼저 본다.
- **API 차단:** 모델 API가 프롬프트를 막으면(`block_reason`) 문항을 제외하지 않고 `blocked`로 기록해 실패로 센다(`behavior = blocked`, `pass = 0`).
- 문항 세트가 다르면 지표를 비교하지 않는다. `experiments.csv`의 `n_cases`·`answer_files`로 확인한다.

## summary.json / summary.csv

그룹(전체, difficulty, set, category, 근거 수, source_doc, question_type, needs_table, 답변 단계는 expected_group도)마다 문항 수와 지표 평균을 담는다.
- 검색 평가: 검색 지표, 시간(mean·p50·p95·max).
- 답변 평가: 답변 지표 평균, `n_errors`, `n_blocked`, 사용량(문항 평균 `*_mean`, 합계 `*_total`), 시간(`retrieval_ms`·`generation_ms`·`total_ms`의 p50·p95).
  `summary.json`에는 기대 행동 × 판정 행동 혼동표(`behavior_confusion`), 평가 소요 시간(`wall_seconds`), 판정 토큰 합계(`judge_usage_total`)도 있다.

## experiments.csv (실험 비교표)

실험당 한 줄. 공통 열: `experiment_id`, `sweep_id`, `stage`, `created_at`, `baseline_id`, `changes`(예: `retrieval.k: 5→3`),
`env_changes`(기준과 다른 코드 버전·데이터 파일 수), `note`, `collection`, `strategy`, `chunk_size`, `chunk_overlap`, `title_prefix`, `table_aware`,
`retrieval_method`, `k`, `fetch_k`, `weight_vector`, `lambda_mult`, `generation_model`, `answer_files`, `n_cases`.
- 검색 평가: 대표 지표 `ctx_hit`·`ctx_recall`·`ctx_precision`·`ctx_source_coverage`·`ctx_key_phrase_recall`·`ctx_dup_rate`·`ctx_context_chars`·`mrr`, k별 전체 지표, 시간.
- 답변 평가: `judge_model`, `n_errors`, `n_blocked`, `pass_rate`, `answer_accuracy`(= must_include_rate), `forbidden_rate`, `faithfulness`, `unsupported_claims_mean`,
  `behavior_match`, `coarse_match`, `citation_precision`, `citation_recall`, `ctx_hit`·`ctx_recall`·`ctx_source_coverage`, `llm_calls_mean`, `retrieval_calls_mean`,
  `input_tokens_mean`, `output_tokens_mean`, `thinking_tokens_mean`, `input_tokens_total`, `output_tokens_total`, `total_ms_p50`, `total_ms_p95`.

## 비교 방법과 도구

1. 기준 실험을 먼저 돌리고 그 실험 ID를 기억한다.
2. 비교 실험은 `--baseline <기준 ID>`로 돌린다. `changes`에 한 항목만 있어야 단일 요소 비교다. 두 개 이상이면(예: 기본 RAG와 Agent) 전체 구성의 효과로 해석한다.
3. 같은 `answer_files`·`n_cases`인 실험끼리만 비교한다.

| 도구 | 용도 |
|---|---|
| `python -m rag.compare --baseline-of <ID> ...` | 기준 대비 지표 차이, 나아진/나빠진 문항 수, 그룹별 차이, (답변 단계) 품질 1%p당 추가 자원 |
| `python -m rag.report [--runs] [--md 파일]` | 설정별 평균과 [최소–최대](반복 실행을 한 설정으로 묶음), 설정별 실험 ID 목록 |
| `python -m rag.failure_analysis <ID> ...` | pass 실패를 검색 놓침·행동·금지 내용·근거 일부·답변 누락으로 분류 |
| `python -m rag.human_sheet make/compare` | 판정 모델의 사람 대조 채점 시트 만들기, 채점 결과와 일치율 계산 |

차이가 난 문항은 두 실험의 `per_case.jsonl`을 `case_id`로 맞춰 본다.
