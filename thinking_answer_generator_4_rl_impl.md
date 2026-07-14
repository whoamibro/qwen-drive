# Thinking-Based Answer Generation v2 — RL Dataset Module: Implementation Spec

## 배경 및 목표

기존: Instruct 8B SFT용 데이터셋 생성 (thinking-based answer generation 모듈).
신규: **Thinking 8B RL(GRPO) 학습용 데이터셋 생성** — 기존 모듈 복사 후 수정.

핵심 설계 원칙 (구현 전 반드시 이해할 것):

1. **Native `<think>`는 통제 대상이 아니다.** 235B teacher의 `<think>` preamble(평균 ~16K chars)은 프롬프트로 길이 제어가 불가능하다. 그대로 두고 파서가 strip한다. 대신 **최종 출력 JSON 안에 tier-banded `think` 필드를 신설**하고, 이것이 student의 `<think>` 학습 타겟이 된다. 배열(array of strings)로 받아 `len()`으로 band 검증한다.
2. **Answer는 JSON key 순서로 마지막에 강제한다.** Autoregressive 생성에서 key 순서 = 토큰 순서이므로, `think → reasoning → answer` 순서 고정이 "생각 후 답변"의 가장 강력한 조건이다.
3. **Contrastive trap 처리를 사후 포렌식에서 계약 필드로 승격한다.** 모델이 contrast 성립 여부를 `contrast_status`로 자가 신고하게 하여, flip fabrication의 유인을 구조적으로 제거하고 same_answer 케이스를 데이터 손실 없이 수확한다.
4. **Grounding 계약은 동결이다.** `tag_mappings` / `selected_obj_id` / `grounded_description` / `relevant_cameras`의 필드명·구조·좌표 규약은 일절 변경 금지 — 하위 SFT 변환기와 bbox 해소 로직이 물려 있다.

---

## TODO List

### T1. 모듈 복제 및 분리

- 기존 thinking answer generation 모듈 복사 → `answer_generator_rl.py` (또는 v2 네이밍 컨벤션에 맞게)
- 출력 디렉토리 분리: `qa_results_rl_thinking/` — 기존 SFT 파이프라인 산출물과 절대 혼합 금지

### T2. Per-pair 출력 스키마 확장 (key 순서 고정)

positive / contrastive 객체 각각:

```json
{
  "tag_mappings": { "...": "..." },
  "instantiated_question": "...",
  "think": [
    "Step 1 (first-person, cites OBJ id / Image idx / distance)",
    "..."
  ],
  "reasoning": "- ...",
  "answer": "yes",
  "confidence": "high",
  "relevant_cameras": [1, 2],
  "contrast_status": "achieved"
}
```

필드 주석:

- `tag_mappings` — 기존 grounding 포맷 그대로, **변경 금지**
- `think` — **NEW**: tier-banded steps, 배열
- `reasoning` — 기존 ≤5 bullet 포맷 유지 (SFT 호환 채널). RL 전용 단순화(`reasoning` 제거)는 하지 않는다
- `answer` — 반드시 마지막 결정 key
- `contrast_status` — **NEW**: contrastive 객체에만 (T6 참조)

### T3. 프롬프트에 Tier 스펙 주입

카테고리 → tier 매핑과 step band를 시스템 프롬프트에 삽입 (하단 프롬프트 스켈레톤 참조):

| Tier | 카테고리 | steps | 사고 구조 |
|---|---|---|---|
| T1 지각·열거 | OBS, IDN, ESC | 2–4 (target 3) | evidence 인용 → (view 확인) → 결론 |
| T2 지각+매핑 | TSS, RML, AAS | 3–5 (target 4) | 관측 → 적용성 매핑 → 결론. AAS는 ambiguity 해소 1단계 명시 허용 |
| T3 관계·동적 | SRO, DRA | 4–6 (target 5) | 다중 객체 위치/모션 인용 → 비교/교차뷰 정합 → 관계·위험 판정 → 결론 |
| T4 규칙·인과 체인 | RWP, CHR | 5–8 (target 6) | 전제 → 규칙/개입 → 다중 agent 상호작용 or 인과 전파 → 결과 → 결론 |

- 배열 원소 1개 = 사고 1단계로 정의 (band 검증의 단위)

### T4. Think 스타일 규정

- 1인칭 자연 prose: "Looking at Image 2, OBJ 21 is a car 12.9m directly ahead..."
- 각 step은 구체 evidence(OBJ id / Image index / 거리) 인용 필수
- 마지막 step은 결론 직전의 판단까지만 — **answer 문자열 자체를 think 안에서 선언 금지** (answer key에서만 도출)
- `reasoning` bullet의 복사·재서술 금지 — think는 도출 과정, reasoning은 확정 요약으로 역할 분리

### T5. Absent-premise Short-Circuit Override

- 질문의 전제 대상(entity/place)이 scene에 부재하면 think는 **정확히 2 steps**: (1) 확인한 뷰와 함께 부재 진술, (2) non-existent-reference rule 적용
- 카테고리 tier보다 우선 적용
- 카메라 교차 +1 / 다객체 비교 +1 override는 **이번 스코프에서 제외** (phase 2 이연; band 상한을 +2 여유로 흡수)

### T6. Contrastive Protocol 전면 교체

- **T6a. Answer-first ordering 명문화** — positive와 contrastive에 **각각 독립적으로, 증거만으로** 답한 후에 contrast 성립을 판정. "contrast를 만들기 위해 결론 낸 답을 조정"을 최상위 금지 조항으로 (t92 패턴: thinking 'no' 결론 → 최종 'yes' flip + reasoning 역짜맞춤 — 이것이 v1의 주 오염원)
- **T6b. `contrast_status` 자가 신고 필드** — 3값 enum:
  - `"achieved"` — 증거 기반으로 답이 실제로 갈림
  - `"same_answer"` — 유효한 contrastive 질문이지만 답이 같음. **드랍하지 않고 정직하게 생성** → 후처리에서 unpaired 단건 QA로 수확 (v1에서 template 단위 격리로 버려지던 honest_same 물량, 샘플당 ~41 pairs 규모)
  - `"skipped"` — 유효한 contrastive 구성 자체가 불가 → `contrastive: null` + `skip_reason`
  - 설계 의도: v1에서는 same-answer 상황에서 flip(오염) 또는 skip(손실) 양자택일이었음. 세 번째 합법 출구가 flip 유인을 구조적으로 제거
- **T6c. Threshold-subjective 카테고리 gating** — DRA/RML(v1 quarantine 52%/64%)은 contrastive를 optional로 명시: 경계가 주관적이라 flip이 불확실하면 marginal contrast 시도 대신 same_answer/skipped 선택
- **T6d. Absent-premise 상호 참조** — contrastive가 absent-entity swap으로 성립하는 경우(v1 contrastive 'no' 94%의 원천) T5의 2-step 규칙 자동 적용을 프롬프트에 명시

### T7. 파서 이식 및 확장

- 재파싱 스크립트의 `</think>` prefix-strip extractor를 이 모듈의 인라인 파서로 이식 — **구모듈의 array-scan 우선 fallback 버그 상속 금지** (thinking 내 `["vehicles",...]` 조각 오탐의 원인)
- `</think>` 부재 시 fallback: object-scan을 array-scan보다 먼저
- 신규 검증: `think` 필드 존재 + 배열 타입 + `len(think)` band 검사
- `contrast_status` 정합 검증: `"achieved"`인데 pos==con 답이면 `status_mismatch` flag (자가 신고 오류 검출 — trust but verify)
- Malformed JSON: repair 시도 → 실패 시 단건 re-inference 큐. **Silent drop 금지**
- `contrastive.answer: null` → 해당 side만 `skipped`로 강등, template 전체 드랍 금지

### T8. 검증 게이트 + Retry + Run 로깅

- 파싱 직후 per-item 검사: think 개수 band 이탈 또는 absent-premise인데 >2 steps → **1회 재생성 큐**, 재실패 시 `band_violation` flag로 보존 (드랍 금지)
- Run 종료 로깅 (`verify_sampler` 철학): parse success rate, band 준수율(카테고리별), `contrast_status` 분포(achieved/same_answer/skipped), status_mismatch 건수
- 해석 기준: DRA에서 same_answer+skipped 합산이 50%대 유지되면 T6c 문구 강화 신호

### T9. Provenance

- 모든 레코드에 global `template_num` + `qa_id` (`s0710_t002_p1_pos` 형식) 각인
- Bank join용 `(category, template_idx)` 병기 유지

### T10. Smoke Test

- 카테고리별 2 template × 5 샘플 규모로: band 준수율, 파싱 성공률, `contrast_status` 분포, grounding 필드 무결성(tag_mappings diff 없음) 확인 후 full run 승인

---

## 프롬프트 스켈레톤 (시스템 프롬프트에 삽입할 뼈대)

```
## Output Contract (STRICT)
After your analysis, output ONLY a JSON object. Within each pair's
"positive"/"contrastive" objects, generate keys IN THIS EXACT ORDER:
  1. "tag_mappings"          — unchanged format (grounded object selection)
  2. "instantiated_question"
  3. "think"                 — an ARRAY of step strings (see Think Budget)
  4. "reasoning"             — at most 5 evidence-citing bullets (existing format)
  5. "answer"                — MUST be the final decision key. Derive it only
                               after "think" and "reasoning" are complete.
  6. "confidence", "relevant_cameras"
  7. "contrast_status"       — contrastive objects only (see Contrastive Protocol)

## Think Budget (per category)
Write "think" as first-person deliberation steps. One array element =
one thinking step. Each step must cite concrete evidence (OBJ id,
Image index, or distance). Step count MUST fall in the band:
  - Observation / Identification / Environmental_and_Sensor_Conditions: 2-4 steps
  - Traffic_Signs_and_Signals / Road_Markings_and_Lane_Configuration /
    Attributes_and_States:                                              3-5 steps
  - Spatial_Relationships_and_Occlusion /
    Dynamic_Agents_and_Risk_Assessment:                                 4-6 steps
  - Right_of_Way_and_Planning / Causal_and_Hypothetical_Reasoning:      5-8 steps

OVERRIDE: if the questioned entity/place is ABSENT from the scene,
"think" is EXACTLY 2 steps: (1) state the absence with the views
checked, (2) apply the non-existent-reference rule.

STYLE: natural first-person deliberation ("Looking at Image 2, OBJ 21
is..."). Do NOT state the final answer inside "think"; do NOT copy
"reasoning" bullets — "think" is the derivation, "reasoning" is the
committed summary.

## Contrastive Protocol (STRICT — read before answering)
1. ANSWER-FIRST ORDERING: Answer the positive and the contrastive
   questions INDEPENDENTLY, based only on scene evidence. Only after
   both answers are committed, evaluate whether they differ.
2. NEVER adjust, reverse, or soften an evidence-based answer to
   manufacture a contrast. A fabricated flip is a critical failure;
   a same-answer pair is NOT a failure.
3. Set "contrast_status" on every contrastive object:
   - "achieved":    answers genuinely differ based on evidence
   - "same_answer": the contrastive question is valid but yields the
                    same answer — output it honestly with this flag
   - "skipped":     no valid contrastive question can be constructed;
                    set "contrastive": null and give "skip_reason"
4. For risk/clarity judgment templates (Dynamic_Agents_and_Risk_
   Assessment, Road_Markings_and_Lane_Configuration): contrast is
   OPTIONAL. When the threshold is subjective and the flip is
   uncertain, prefer "same_answer" or "skipped" over a marginal flip.
5. If the contrastive swaps in an entity/place ABSENT from the scene,
   the 2-step think override applies (see Think Budget OVERRIDE).

## Hard Conditions
- "tag_mappings" format is frozen; do not add/rename grounding fields.
- "answer" is always the last decision; never pre-commit it in "think".
```

---

## 후처리 라우팅 (참고 — 이 모듈의 소비자 관점)

| contrast_status | 라우팅 |
|---|---|
| `achieved` | pos+con pair로 학습 데이터 편입 (pair 링크는 GRPO grouping key) |
| `same_answer` | pos, con 각각 **unpaired 단건 QA**로 수확 |
| `skipped` | positive만 단건 수확 |
| `status_mismatch` flag | quarantine (자가 신고 신뢰 불가 케이스) |
| `band_violation` flag | 보존하되 학습 편입은 보류 (band 준수율 개선 후 재평가) |

정규식 기반 trap 마이닝(stated-answer cross-check)은 폐기하지 않고 **자가 신고의 감사 수단**으로 유지 — 초기 run에서 `contrast_status` 신뢰도가 검증되면 감사 빈도를 낮춘다.

---

## 구현 노트 (Claude Code 전달용)

- T1부터 순서대로 진행. T7 파서는 기존 재파싱 스크립트 코드 재사용.
- Smoke test(T10) 결과에서 band 준수율과 contrast_status 분포를 리포트할 것 — 프롬프트 문구 튜닝 1라운드의 입력이 된다.
