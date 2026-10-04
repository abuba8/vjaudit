# Research Direction v1 (for lock): Topic C, Judge-Scored Open-Ended VideoQA

**Date:** 2026-10-03 · **Status:** **LOCKED 2026-10-03.** Direction **B** (error budget + system-dependent error + human-anchored protocol). Venue: **ICCV 2027, full scope**, with the GPT-3.5 capture and SP0 done immediately. This replaces §1–§3 of `REQUIREMENTS_CONTRACT.md` (see §12 for the delta).
**Inputs:** `REQUIREMENTS_CONTRACT.md` v1.0; BlackboxNLP 2026 camera-ready; `CVPR_3.pdf` (earlier draft); literature sweep of 2024 to Sep 2026 (references at the end).

---

## 0. Verdict

1. **The contract's headline is already known in weaker form.** Prior work already shows that the judge moves open-ended VideoQA scores by double digits and can reorder models: LLaVA-Hound, MMBench-Video, EditVid, FreeVA. In July 2026 the same audit-and-repair template was published *with human validation* for image short-answer VQA (arXiv 2607.10240). Leading with magnitude makes the paper look incremental.
2. **The contract's chosen novelty (C3) is the weakest part for CVPR.** C3 is the logit-level test of demonstration label bias. It is NLP-centric, and its core model (demonstrations add a shift to the label logit) is the premise of contextual and batch calibration. Judge analysis that separates "operating point" from "sensitivity" was published for text judges in June 2026. A CVPR reviewer will ask why it is not at ACL. It should become a section, not a headline.
3. **The open, vision-specific gap.** No one has scored open-ended VideoQA answers against *what the video shows*. Every validation so far compares to the reference answer, or uses no humans at all. So it is unknown how much judged-accuracy error comes from the **reference**, how much from the **judge's text matching**, and how much from the **judge not seeing the video**. It is also unknown whether judge errors depend on **how the answer was produced**: blind guessing, hallucinated detail, or verbosity.
4. **Pivot.** Move from "audit of judge configurations" to three linked parts:
   (a) an **error budget of open-ended VideoQA with the video in the loop**, built on a dual-label human set;
   (b) **system-dependent judge error** and its consequences for vision conclusions: rankings, the measured benefit of seeing the video, and failure of standard bias correction;
   (c) a **human-anchored replacement protocol** for the retiring default judge.
   Demonstration label bias becomes evidence about the operating point inside (c).
5. **The clock.** `gpt-3.5-turbo`, the protocol's default judge, **shuts down on 2026-10-23**, 20 days from today. The CVPR deadline is **2026-11-16** (registration 11-10), about 6 weeks away. The full 11-week contract does not fit. A focused cut might. Capturing GPT-3.5 verdicts must happen now whatever venue we pick: after Oct 23 it can never be done.

---

## 1. Research landscape

### 1.1 Judge instability in open-ended VideoQA (direct prior work)

| Work | What it established | Implication for us |
|---|---|---|
| Video-ChatGPT (Maaz et al., ACL 2024) | GPT-3.5 yes/no + 0–5 protocol for MSVD/MSRVTT/TGIF/ActivityNet-QA | The protocol family under study. |
| LLaVA-Hound-DPO (Zhang et al., 2024), App. A | Same model across ChatGPT 0301/0613/1106: MSVD-QA 88.5/80.7/86.1; MSRVTT-QA 82.1/70.2/76.8. Rankings "consistent" within a version. No humans. | **H1a (≥10-pt range) is already shown** for one judge family. Main counter-hypothesis: rankings stable. |
| FreeVA (Wu, 2024) | GPT version changes open-ended VideoQA results significantly | Same; no human ground truth. |
| MMBench-Video (Fang et al., NeurIPS 2024 D&B) | GPT-3.5 gives inflated scores and has larger version variance. On 100 questions, MAE vs humans is 0.98 for GPT-3.5 vs 0.36 for GPT-4 (Video-LLaVA outputs). | Leniency of the default judge is documented, at small scale and on a Likert scale. |
| EditVid / *Beyond Raw Videos* (Xu et al., CVPRW 2025) | A "sorry" refusal gets 38–64% accuracy under the GPT-3.5 judge and 0% under GPT-4. Ranking reversal on ActivityNet-QA. | Shows refusals and hedges are credited. Anecdotal, no human labels. |
| Video-ChatGPT GitHub issue #135 | Re-running the official evaluation gives numbers *above* the paper's | Real reproducibility drift, seen in practice. |
| VideoQA in the Era of LLMs (Xiao et al., 2024) | Blind QA keeps >70% of normal accuracy (NExT-QA MC). GPT-3.5 judging of MSVD/ActivityNet used without validation. | Language priors are strong. **No one has measured blind answers under judges against humans.** |

### 1.2 Multimodal and video judges

| Work | Finding | Implication |
|---|---|---|
| VideoJudge (Waheed et al., ICLR 2026) | Trained 3B/7B video judges (1–5 pointwise and pairwise). MLLM judges beat text judges given video descriptions; "video input is crucial". Does not study the Video-ChatGPT QA protocol. | "Video helps judging" is claimed for **rating** tasks. Our object is reference-based binary correctness, where video helps only when the reference is insufficient. That has to be shown, not assumed. |
| *Is Your Video LM a Reliable Judge?* (Liu & Zhang, ICLR 2025) | Weak VLM judges over-rate. Mixing reliable and unreliable judges does not help. No human labels. | Ensembles (contract M6) are a weak bet. |
| Perception-Judge (KAIST, ICML 2026; image) | MLLM judges "reward plausible narratives over perceptually correct answers" and anchor on the response instead of their own perception. | Predicts **perception error** from VLM judges, so the H3a gain is not guaranteed. This makes the vision experiment non-trivial (good). |
| VideoRewardBench (2025); VURB (2026) | Text- and image-trained reward models collapse to about 50% on video. Position bias. | Supports including video-specialised judges. |
| PlaylistEval (arXiv 2609.34314, Sep 2026) | Pairwise judging of day-scale videos. Best judge 75.4% vs 93% human; open judges near chance. | Contemporaneous. Different regime (long video, pairwise, no reference). **Must read in full** (fetch was rate-limited). |
| AutoEval-Video (Chen et al., ECCV 2024) | Instance-specific rules let GPT-4 reach 97.0% vs 97.5% single-human accuracy; a unified prompt gives 87.0%. | Strong judges with rich references can work, at high annotation cost per item. |
| MLLM-as-a-Judge (ICML 2024); Judge Anything (2025); LLaVA-Critic (CVPR 2025) | Image-centric pairwise and score judging | Background only. |

### 1.3 Reference-based answer matching and verifiers (text)

| Work | Finding | Implication |
|---|---|---|
| Answer Matching (Chandak et al., 2025) | Small modern LLMs (Qwen3-4B) matching against a reference reach Scott's π ≈ 0.97 with humans on MATH, about human level on MMLU-Pro/GPQA free-form. Reference-free verification: 0.72. MC: 0.26. | **Strongest counter-hypothesis.** With a modern judge, text matching may be nearly solved. The remaining video-specific error is then the *reference* and *system-dependent* error, which is our pivot. |
| Judging the Judges (Thakur et al., GEM 2025) | On TriviaQA, only the largest judges align well (π ≈ 0.87–0.88 vs 0.96 human–human). Percent agreement hides 10-pt score gaps. Lenient bias. Even "contains" ranks models at ρ = 0.99. Judges accept "Yes"/"Sure". | Rankings may survive weak judges, so the ranking-flip claim needs care. Report κ/AC1, not percent agreement. |
| One Token to Fool (Zhao et al., 2025) | Reference-based binary verifiers accept content-free "master keys". "Thought process:" gives 60–90% FPR on open judges; GPT-4o gives 35% on punctuation. | Stress-probe precedent. The sorry and refusal probes are the video analogue. |
| CompassVerifier (EMNLP 2025); Pitfalls of rule/model verifiers (2025) | Trained answer verifiers; model verifiers can be hacked | Should appear as a judge family. Missing from the contract. |

### 1.4 Validity of VQA evaluation and benchmarks

| Work | Finding | Implication |
|---|---|---|
| *What Does Your Short-Answer VQA Score Actually Measure?* (arXiv 2607.10240, Jul 2026; image) | Six image short-answer benchmarks. 19–48% of official errors are semantically acceptable. Benign prompt rewrites flip 1.8–29.4% of question outcomes. Rankings largely preserved (adjacent swaps only). Text-only judge validated at κ = 0.917. CPU "contract repair". | **The closest prior work, and it predates the CVPR cutoff.** It covers image VQA and never shows the image to the judge or annotators. Our delta: video, video-grounded human labels, system-dependent error, judge-side mechanisms. |
| KB-VQA audit (arXiv 2607.00159, Jul 2026; image) | InfoSeek: 22% unsupported answers. 47–59% underspecified questions. Rankings shift after repair. | Reference and question defects can be large. Video benchmarks have not been audited this way. |
| Open-ended VQA benchmarking (Ging et al., ICLR 2024) | GPT-4 matches individual human raters on VQAv2 (r = 0.97 few-shot). "Contains" is a strong cheap metric. | Reference-based judging works when references are good. |
| LAVE (Mañas et al., AAAI 2024) | LLM-assisted VQA evaluation correlates better with humans than VQA accuracy | Image precedent for in-context judge prompts. |
| Breaking Down Video LLM Benchmarks (2025) | GPT-4o gets up to 50% on Video-MME and NExT-QA MC with no video | Blind priors are large. Their interaction with judges is unmeasured. |

### 1.5 Statistics of judge-based evaluation (the "repair" literature)

| Work | Finding | Implication |
|---|---|---|
| Lee et al., *How to Correctly Report LLM-as-a-Judge Evaluations* (arXiv 2511.21140) | Bias-adjusted accuracy with CIs from a calibration set. **Assumes sensitivity and specificity are the same across evaluated models.** | Our differential-error finding would directly test this assumption. |
| AutoEval Done Right (Boyeau et al., ICML 2025); Stratified PPI (Fisch et al., NeurIPS 2024); PP-Ranking (Chatzi et al., NeurIPS 2024) | Prediction-powered inference: unbiased estimates and ranks from a few human labels plus many judge labels | Contract M5/M8 already exist as methods. Our contribution is the empirical validity test in video, not a new estimator. |
| Dorner et al. (ICLR 2025, oral) | If the judge is not better than the evaluated model, debiasing saves at most a 2× factor in human labels | Caps the "repair" claim. Must be cited and respected. |
| Trust or Escalate (Jung et al., ICLR 2025) | Selective escalation with provable human-agreement guarantees | Contract M8 is this idea. Cite it; do not claim it. |
| ANCHOR (arXiv 2609.19599, Sep 2026; text, pairwise) | Human-anchored ranking of new models with feature-dependent judge bias | Contemporaneous, text-only. |

### 1.6 Judge bias mechanisms (where C3 lives)

| Work | Finding | Implication |
|---|---|---|
| Zhao 2021; Fei 2023; Zhou 2024 (batch calibration) | Demonstration and label-prior bias as a shift in label logits; calibration fixes it | H2b ("additive shift") is the premise of this literature. Confirming it is not a contribution. |
| *LLM Judges Have Dark Current* (arXiv 2606.15610, Jun 2026; text) | Signal-detection framing of judges, separating sensitivity from criterion. Prompts shift the operating point without improving resolution. Proposes a "Judge Datasheet". | **H2a's dichotomy and the "Judge Card" both have text precedent.** Cite; do not claim. |
| Jain et al. 2025 (agreeableness bias) | Judges accept valid outputs well but reject invalid ones poorly (TNR < 25%) | Same one-sided pattern as the BlackboxNLP finding. |
| Authors' BlackboxNLP 2026 | Demonstration label bias in an 8B text judge, inflation-only within the tested range | Prior work. See the §3 integrity note. |

### 1.7 Exposure: is the protocol still used?

- **Yes, at CVPR.** A CVPR 2026 token-compression paper (Du et al.) evaluates ActivityNet-QA with "GPT-3.5-Turbo as an evaluator". Training-free video-LLM papers from 2024 (SlowFast-LLaVA, FreeVA and their successors) built their headline tables on MSVD/MSRVTT/ActivityNet-QA with GPT-3.5. Gain sizes in the 2025–26 literature are unverified; SP8 must measure them.
- **New open-ended benchmarks keep using unvalidated judges.** EG-VQA (2026) uses Gemini-2.5-Pro, with "high consistency" established only by manual inspection.
- **The default judge is being retired.** OpenAI shuts down all `gpt-3.5-turbo` snapshots on **2026-10-23** and recommends `gpt-5.6-terra` as the replacement. Every future paper must switch judges, and nobody knows how new numbers relate to the legacy table. This is the strongest "why now" the project has.
- **Caveat.** Frontier model reports (Qwen-VL, InternVL) mostly report MC benchmarks now. SP8 (survey) must quantify exposure honestly.

---

## 2. What is genuinely missing

**Already known. Do not claim:**
- the judge or its version moves scores by ~10 points;
- GPT-3.5 is lenient;
- refusals and hedges get credit;
- video input helps *rating-style* judging;
- modern text answer matchers are near-human on text QA;
- PPI, bias correction and selective escalation as methods;
- the signal-detection framing for judges.

**Open gaps (no prior work found as of 2026-10-03):**

- **G1: Ground truth that looks at the video.** In VideoQA no study measures how often a candidate is correct *given the clip* but wrong *given the reference*, or the reverse. In particular, no study measures how often references are wrong, ambiguous, or one of several valid answers. The image analogue (2607.10240) never showed the image to anyone. Without this, every judge-vs-human number in the field is "judge vs reference-reading human," and the error floor set by the benchmark itself is unknown.
- **G2: System-dependent judge error tied to visual grounding.** Leniency is known on average. Unknown: whether the false-accept rate depends on *how the answer was produced*. Blind language-prior guesses, hallucinated detail and verbose answers could each be credited at different rates than grounded short answers. If so, (i) score corrections that assume equal error rates across models (Lee et al.; pooled calibration) are biased, and (ii) conclusions drawn by *comparing* systems are distorted. That includes the value of visual input.
- **G3: Judge-induced distortion of vision conclusions.** "With frames vs. without," "more frames help," and close leaderboard gaps are all measured through the judge. None has been checked against humans.
- **G4: Protocol continuity after GPT-3.5.** There is no human-validated open replacement, and no evidence on whether legacy GPT-3.5 numbers can be mapped onto a new scale. If error is system-dependent, a single mapping should fail.

---

## 3. Adversarial critique of the original formulation

### 3.1 By contribution

| Contract item | Critique | Verdict |
|---|---|---|
| **C1** dual-label set (`L_ref` + `L_vid`) | The strongest asset, but under-used. `L_vid` is only a T1 subset and only feeds H3. It should be the backbone of the paper. `L_vid` is also under-specified: MSVD/MSRVTT questions are generated from one caption, so many clips admit **several valid answers**. Without a tag for "alternative valid answer," reference insufficiency will be under-counted. | **Promote and refine.** |
| **C2** magnitude audit + variance decomposition | Magnitude is known (§1.1). A variance decomposition over a grid we choose is partly an artefact of which configurations we include. "Judge-shopping range" is quotable but depends on the grid. | Keep as **setup** (one figure), not as a contribution. |
| **C3** logit additive shift + asymmetry | The additive shift is the calibration literature's premise. The asymmetry is about one 8B text judge from the authors' own prior paper (self-incremental). The SDT dichotomy has text precedent. About 3M judge calls (G-SWEEP + G-ORDER) and 1.2 pages, for the least vision-relevant result. | **Demote** to an operating-point section and supplement. Drop G-ORDER and G-FLIP. |
| **C4** video-grounded judging | H3a as written is close to circular. A text-only judge approximates `L_ref` by construction, so "frames agree better with `L_vid`" mostly re-measures `L_ref ≠ L_vid`. The informative question is whether frames *reduce reference-inherited error more than they add perception error* (Perception-Judge predicts the added error). | **Reframe** as an error budget. |
| **C5** protocol + toolkit | M1–M8 are known methods (PPI, escalation, calibration). A "protocol selected on dev" is fine engineering but not novel. The novel and testable part is whether the **assumptions** of these repairs hold in video, especially equal error across systems. | **Reframe** around assumption testing and the GPT-3.5 sunset. |

### 3.2 By hypothesis

- **H1a (≥10-pt range).** Near-certain once GPT-3.5 and modern judges sit in one grid; LLaVA-Hound already shows 12 points. It cannot fail, so it is not a hypothesis. Keep it as descriptive.
- **H1b ("bias correlates with answer length", Spearman over systems).** With 6–8 systems, ρ over 6–8 points has almost no power. Test at item level instead: a mixed model with length, blindness and hallucination flags as covariates.
- **H1c (flip rate for <3-pt pairs).** **Underpowered by design.** With 300 annotated questions per benchmark and about 20–25% answer discordance between two systems, the 95% CI on a human-measured paired gap is roughly ±5–6 points. Humans cannot resolve <3-pt gaps at this n, so the target pairs will all be "unresolved." Resolving a 3-pt gap needs roughly 850 (CI half-width) to 1,750 (80% power) questions **for that pair**.
- **Counter-hypothesis rule (τ_b ≥ 0.9) and H4 (τ_b ≥ 0.9 on every benchmark).** With 6 systems, a single swap gives τ = 0.87. The threshold therefore demands a perfect ranking, while the human ranking is itself uncertain. Replace with "no reversals among human-resolved pairs" plus per-system |bias| bounds.
- **H2b (b ∈ [0.9, 1.1], R² ≥ 0.8).** Likely true for small shifts and uninformative if true. Fine for the supplement.
- **H3a.** See C4: the hypothesis needs to be about the error decomposition.
- **H4 (κ ≥ human−0.10, cost ≤ 5×).** Reasonable, but Dorner et al. bound what any judge-plus-human protocol can save. State the claim in those terms.

### 3.3 Integrity risk inherited from the prior paper (resolve first)

The contract's R-SP0-02 notes that validation generations were **byte-identical across zero-, one- and few-shot**, even though the exemplars were supposedly injected at generation. If the exemplars never reached the generator:
- the one-shot test-set differences (for example +16.09 for S1) are **judge-side**, not generation-side;
- the "directional transfer effect" in BlackboxNLP §4.3 is mis-attributed and may need a correction;
- ironically, this would make the judge-fragility evidence *stronger* (one demonstration moves judged accuracy by 16 points).

Either way, this has to be settled (SP0) **before** the new paper cites the prior one. An unresolved discrepancy is the kind of thing a reviewer who knows the prior paper will find.

### 3.4 Feasibility

- 6 weeks to deadline vs 11 planned. University of Essex ethics approval for paid annotators is an unknown-length dependency.
- The judge pool is dated: Gemma 3 → Gemma 4 and Qwen3.5 now exist. Frontier OpenAI models expose logprobs inconsistently (the GPT-5.x line). A frontier video judge with video input (a Gemini-class model) is needed as the ceiling, or "your judges are too weak" will stick.
- The NLP-heavy mechanism content plus a broad audit plus a protocol is too much for 8 pages. The contract has 9 hypotheses; a strong CVPR paper has 3 findings.

---

## 4. Alternative formulations

| | **A. Trimmed contract** | **B. Error budget + system-dependent error (recommended)** | **C. GPT-3.5 sunset / migration** | **D. Method: video-grounded verifier** |
|---|---|---|---|---|
| Core claim | Judges move scores and rankings; demonstration bias is a logit shift; video helps; here is a protocol | Judged accuracy = truth + reference error + matching error + perception error. Judge error depends on the system, which distorts rankings and the measured value of vision. A human-anchored protocol fixes it. | The default judge is gone; here is a validated replacement and why legacy numbers can't be mapped | A training-free "verify reference against video, then judge" protocol beats text judges |
| Novelty vs §2 | Low–medium (magnitude known, mechanism NLP) | **High** (G1–G4) | Medium (timely, narrow) | Medium; depends on effect size |
| Vision relevance | Medium | **High** | Medium | High |
| Risk | "Known / belongs at ACL" | `L_vid` agreement may be low; differential error may be small | "Workshop paper" | Gains may be marginal; perception error |
| Fits 6 weeks | No | **Focused cut: yes, tight** | Yes | Risky |
| Reuses contract infrastructure | Fully | ~85% (same SP0–SP4, SP6–SP8) | ~70% | ~60% |

**Recommendation: B, with C built in as the protocol section and a minimal version of D as one candidate protocol** (a reference-check prompt for VLM judges). A's demonstration-bias material survives as the "operating point" evidence. It shows why cheap calibration fixes average leniency but not system-dependent error, which is exactly the bridge into B's main claim.

---

## 5. Refined research hypothesis

> **In reference-based judging of open-ended VideoQA, the reported accuracy departs from video-grounded correctness through three separable errors.**
> - **Reference error:** the benchmark answer is wrong, ambiguous, or one of several valid answers.
> - **Matching error:** the judge mis-compares candidate and reference.
> - **Perception error:** a video-seeing judge mis-perceives the clip.
>
> **The rates of these errors depend on the system being evaluated.** Answers produced from language priors (blind), with hallucinated detail, or with verbosity are credited at different rates than visually grounded answers. Consequently:
> 1. modern text judges largely remove matching error, but **the reference becomes the binding constraint** on auto-generated benchmarks;
> 2. **average-level fixes** (better prompts, balanced demonstrations, threshold or bias correction with pooled error rates) **do not restore valid comparisons between systems**;
> 3. judge-measured **vision conclusions**, such as the gain from seeing the video, are distorted;
> 4. valid comparisons need **video-aware judging plus a small, per-system human anchor**, and legacy GPT-3.5 numbers cannot be translated by any single mapping.

### Pre-registered sub-hypotheses (default thresholds, fixed before results)

| ID | Hypothesis | Test | Threshold | Fallback wording |
|---|---|---|---|---|
| **HA** reference floor | `L_ref ≠ L_vid` is substantial on auto-generated QA and smaller on human-authored QA | Disagreement rate with Wilson CI, by error tag | ≥ 5% of units on MSVD-QA; MSVD > ActivityNet-QA | "References are largely sufficient; the floor is X%." Results go to the supplement and G-B weight rises. |
| **HB** matching is near-solved | The best modern open text judge reaches κ vs `L_ref` within 0.05 of human–human κ; GPT-3.5 (T-VCG) does not | κ/AC1 with clustered bootstrap | ≥ (human κ − 0.05) for the best open judge | If no judge reaches it, matching error stays a major term and the budget still holds. |
| **HC** system-dependent error | For a fixed configuration, the false-accept rate on human-incorrect answers differs across systems; blind and verbose variants are among the highest | Mixed logistic `verdict ~ human × (judge × system) + covariates + (1\|question)`; per-system FPR with CIs pooled over benchmarks | Max−min FPR gap ≥ 10 pts with CI excluding 0, for GPT-3.5 **and** ≥1 modern judge | "Error is approximately non-differential; pooled corrections suffice." That is a positive result for practitioners, and the paper pivots to HA/HE plus the protocol. |
| **HC2** pooled correction fails | Pooled bias correction (Lee et al.) leaves ≥1 system with residual \|bias\| > 3 pts on test; per-system stratified correction brings all within 2 pts | Held-out human test split | As stated | Report the residuals as found. |
| **HD** visual-gain distortion | Judge-measured gain Δ_J = Acc_J(frames) − Acc_J(blind) differs from the human gain Δ_H | Paired, clustered bootstrap on Δ_J − Δ_H | \|Δ_J − Δ_H\| ≥ 5 pts for the default config on ≥1 benchmark (two-sided; direction reported) | "Judges preserve the measured visual gain within ±X." |
| **HE** video judges trade errors | With frames, the same backbone fixes reference-inherited errors (units with `L_ref ≠ L_vid`) but adds perception errors on units with `L_ref = L_vid`; net gain only above a capability level | Matched-backbone paired comparison, split by unit type | Accuracy on `L_ref≠L_vid` units +15 pts; net κ vs `L_vid` +0.05 for the strongest open VLM judge | "Current open VLM judges cannot use the video to fix reference errors; only frontier judges can." That is still a finding. |
| **HF** operating point vs discrimination | Template wording and demonstration label balance shift acceptance but not AUC; ≤ 50 dev labels of threshold calibration remove the *average* bias but not the per-system spread from HC | ΔAUC paired bootstrap; per-system bias before and after calibration | \|ΔAUC\| ≤ 0.02 while acceptance shifts ≥ 10 pts; per-system spread reduced by < 50% | Report ΔAUC and Δbias per factor without the dichotomy. |
| **HG** protocol and legacy mapping | A protocol selected on dev gives, on test, per-system \|bias\| ≤ 2 pts on all T1 benchmarks and **no reversals among human-resolved pairs**. A single GPT-3.5→human mapping misses ≥1 system by > 3 pts | `final_eval.py` on the test split | As stated | "How far automated judging goes; where human anchoring remains necessary." |

---

## 6. Research question and research statement

**Research question.** When an LLM judge scores open-ended VideoQA answers against a reference, how far does the reported accuracy depart from what the video shows? Which parts of that departure depend on the system being evaluated? What is the minimum human anchoring that restores valid comparisons between video models?

**Research statement (for the intro and abstract, to be locked):**
> Open-ended VideoQA accuracy is a joint measurement of the model, the reference answer, and the judge. Using the first video-grounded, dual-label human meta-evaluation of modern video LLMs, we decompose judged accuracy into reference, matching and perception errors. We show that judge errors are system-dependent in ways tied to visual grounding, which invalidates pooled corrections and distorts the measured value of visual input. We also release a human-anchored protocol that keeps model comparisons valid after the community's default judge is retired.

**Working titles:** *The Reference Is Not the Video: Auditing Judge-Scored Open-Ended VideoQA*; *Who Grades the Video? Reference, Matching and Perception Error in Open-Ended VideoQA*.

---

## 7. Proposed contributions

| ID | Contribution | Novel vs. §1? | R-NOV-03 |
|---|---|---|---|
| **C1′** | **Dual-label meta-evaluation set.** `L_ref` + `L_vid` labels; reference-validity labels (correct / wrong / ambiguous / *one-of-several*); error tags; **annotator-written alternative valid answers**; answers from modern video LLMs incl. **blind** and **verbose** variants, on 2–3 benchmarks. Bundled with the **last captured GPT-3.5 verdicts and logprobs** for every annotated answer, which can never be regenerated. | Yes (G1) | **Novel item 1** |
| **C2′** | **Error budget** of open-ended VideoQA (HA, HB, HE): how much error is the reference's, the judge's matching, and the judge's perception. Includes a direct test of *fix the benchmark* (text judge + expanded references) vs *fix the judge* (video judge). | Yes (G1) | — |
| **C3′** | **System-dependent judge error and its consequences** (HC, HC2, HD): unequal false-accept rates for blind, hallucinating and verbose answers; failure of pooled bias correction; distortion of the measured visual gain and of close rankings. | Yes (G2, G3) | **Novel item 2** |
| **C4′** | **Human-anchored protocol + GPT-3.5 sunset bridge** (HF, HG): operating-point fixes (calibration, balanced demonstrations), per-system stratified anchoring, Judge Card, open scorer; evidence on whether legacy numbers can be mapped. | Partly (G4 new; methods known) | — |
| (supp.) | Demonstration label-bias sweep at the logit level; continuity with BlackboxNLP, delta stated explicitly (R-NOV-02) | Incremental | — |

---

## 8. Key experiments (minimum set to validate the claims)

**E0. Integrity (SP0), first, ≤ 3 days.** Diff the prior pipeline's generations. Locate the exemplar-injection cause. Decide on any correction to the prior paper. Reproduce BlackboxNLP Tables 2–3.

**E1. Generations.** Default benchmarks: **MSVD-QA** (auto-generated; tests the reference floor) and **ActivityNet-QA** (human-authored; still reported with GPT-3.5 at CVPR 2026) as T1. **NExT-OE** as T1-if-on-track, because it brings the judge-free MC anchor.
- Systems: 5 generators (≥ 3 families, 1 weak, ≥ 1 expected close pair); a **verbose** variant of one generator; **blind (no frames)** variants of 2–3 generators.
- T2: a frame-count ablation (1/4/16 frames) for one generator.
- Order of work: annotated subset first (300 questions/benchmark ≈ 6–7k generations, a few GPU-hours), then the main 2,000-question sample.

**E2. GPT-3.5 capture: hard deadline 2026-10-23.**
- Run `gpt-3.5-turbo-0125` (and `-1106` if reachable) with the verbatim T-VCG prompt plus a logprob readout on every generation in the annotated subset, then the main sample.
- Cost estimate: ~10⁵ calls × ~300 tokens ≈ tens of USD.
- Cache all responses (R-GLB-03). If generations are not ready by about Oct 18, capture on the annotated subset only.

**E3. Human labels (SP4), with refinements.**
- `L_ref` on all units.
- `L_vid` on ≥ 100 questions/benchmark. The annotator watches the clip, **writes every answer they would accept** (before seeing candidates), labels the candidates, then labels the reference: correct / wrong / ambiguous / one-of-several.
- New error tag **E10, "alternative valid answer not covered by reference."** Split E8 into "reference wrong" and "reference underspecified."
- Report pre-adjudication κ/AC1/α **separately for `L_vid`**; it will be lower than for `L_ref`, and the ceiling matters.
- Blind-variant answers need only `L_ref` (correctness does not depend on what the model saw).
- Estimated added cost: about 10–12 annotator-hours over the contract (blind units, alternative-answer field). This is offset by shrinking the demonstration pool (E8).

**E4. Judge grid.**
- Text judges: Llama-3.1-8B (continuity), Qwen3-8B, a current mid-size (Gemma 4 or Qwen3.5 class), a large open model (≥ 27B AWQ), CompassVerifier (a trained answer verifier), GPT-3.5 (E2).
- VLM judges in matched text-only / frames pairs: Qwen3-VL-8B, InternVL3.5-8B.
- VideoJudge-7B.
- **One frontier judge with video input** (Gemini-class, cost-capped) as the ceiling.
- Templates: T-VCG and T-RUB, zero-shot. Margin readout wherever logprobs exist.

**E5. Error budget (C2′).** Against `L_vid`, split each judge's errors into reference-inherited, matching, and perception terms. **"Fix the benchmark vs. fix the judge":** compare a text judge given the annotator-written expanded references against video judges, scored on held-out `L_vid` items.

**E6. System-dependent error (C3′).** Mixed model (HC). Per-system FPR/FNR. Covariates: answer length, `has_hallucinated_detail`, blind flag, hedge/refusal flag. **Bias-correction test (HC2):** pooled vs per-system Rogan–Gladen / Lee et al., and PPI, on test.

**E7. Visual-gain distortion (HD).** Δ_J vs Δ_H for frames vs blind, per judge, per benchmark. T2: frame-count ablation slopes, judge vs human.

**E8. Operating point (HF), reduced.** k = 4 demonstrations, Yes-fraction ∈ {0, .25, .5, .75, 1}, R = 5 draws, 3 judges (Llama-3.1-8B, Qwen3-8B, Qwen3-VL-8B text-only). Threshold calibration with n ∈ {25, 50, 100} dev labels. Demonstration pool shrinks to 100 human-verified items per benchmark. **Drop** G-ORDER and G-FLIP.

**E9. Rankings, power-aware.**
- Reversals are counted **only among human-resolved pairs**.
- The judge-shopping range for close pairs is reported descriptively against the survey's claimed deltas (SP8).
- T2: deep-annotate **one** close pair on MSVD-QA to ≈ 1,000 questions (only that pair's units; single-word answers make this fast, about 12–16 annotator-hours). The aim is to show or refute a reversal at a small gap.

**E10. Protocol (C4′).**
- Selection on dev; evaluation on test via `final_eval.py`.
- Candidates: best single open judge ± calibration; video judge; text judge + reference check; per-system stratified anchoring with n ∈ {25, 50, 100}; selective escalation (cite Trust-or-Escalate).
- **Legacy mapping test:** fit GPT-3.5→human on dev; measure per-system error on test.
- Report the cost–agreement frontier and the Dorner et al. bound.

**E11. Survey (SP8). Cheap, keep.** Judge-version disclosure rate, plus the distribution of claimed gains vs the measured judge-induced spread. This motivates the intro.

**Compute.** Well below the contract's 240 GPU-hours, because G-SWEEP and G-ORDER shrink by more than 80%. **Annotation:** about 55–65 annotator-hours, similar to the contract.

---

## 9. Failure modes and likely reviewer objections

| # | Objection or failure | Likelihood | Pre-emption | Gate / fallback |
|---|---|---|---|---|
| 1 | "Known: LLaVA-Hound, MMBench-Video, EditVid; and 2607.10240 did this for images." | High if we lead with magnitude | Lead with C1′ and C3′. Magnitude is one setup figure. Explicit delta to 2607.10240: video, video-grounded labels, system-dependent error. | — |
| 2 | "Modern judges are fine" (Chandak; Thakur), so the problem is legacy-only | Medium–high | HB *expects* this and uses it: if matching is solved, the reference floor and system-dependent error are what remain | If HC also fails, pivot to the reference audit plus the GPT-3.5 bridge (approach C). Weaker but publishable. |
| 3 | "`L_vid` is subjective" | Medium | Rubric with worked examples; alternative-answer field; pilot; report the `L_vid` human ceiling separately | If pre-adjudication κ(`L_vid`) < 0.5 after the pilot, restrict `L_vid` to clear cases and report the ambiguity rate as a finding |
| 4 | "Your human comparisons are underpowered" | High for close pairs | Power analysis after the pilot (R-ANA-03); claims only on human-resolved pairs; optional deep-annotated pair | — |
| 5 | "This is an NLP paper" | Medium | Vision-specific findings in the headline (reference vs video, perception error, visual-gain distortion); mechanism in the supplement | — |
| 6 | "Legacy benchmarks nobody uses" | Medium | CVPR 2026 usage; GPT-3.5 sunset; SP8 numbers; optional modern open-ended benchmark (T3) | — |
| 7 | "Judges too weak" | Medium | Frontier video judge as ceiling; results reported by judge size | — |
| 8 | Contemporaneous collision (PlaylistEval, others) | Medium | Novelty audit at start and T−14 days; the dual-label set and system-dependent error are distinctive | — |
| 9 | Overlap with or error in the prior BlackboxNLP paper | Medium | E0 first; third-person citation; transparent correction if needed | G0 |
| 10 | "Protocol = PPI / Trust-or-Escalate" | High if over-claimed | Claim the *assumption test* and the video-specific evidence, not the estimator | — |
| 11 | Ethics delay blocks annotation | Unknown | Ask Essex on day 1. Ask whether author annotators can start the pilot. | Retarget venue |
| 12 | GPT-3.5 capture missed | Low if acted on now | E2 starts in week 1 | If missed, the protocol section loses the legacy bridge; use whatever was captured |

---

## 10. What would make this a strong CVPR paper

1. **One quotable, vision-specific headline**, for example: "On MSVD-QA, X% of answers marked wrong are correct given the video; the reference, not the judge, caps measurable accuracy at Y%." Or: "Under the community-default judge, a model that never sees the video scores within Z points of one that does; humans put the gap at W."
2. **A teaser figure that shows the decomposition.** Per system: GPT-3.5, best modern text judge, video judge, human `L_ref`, human `L_vid`, plus the blind variant. The arrows (visual gain) shrink or grow by judge.
3. **A resource people will use the week it's released.** The dual-label set, the last GPT-3.5 verdicts, a validated open scorer for the post-GPT-3.5 protocol, and a Judge Card.
4. **Rigour that can't be waved away.** Pre-registered thresholds, a dev/test split of human labels, power analysis, clustered CIs.
5. **Three findings, not nine.** Error budget; system-dependent error and its consequences; a protocol that works on held-out humans.

---

## 11. Time-critical actions and the venue decision (decided: ICCV 2027, full B; see §13)

**Do these this week, whatever we choose:**
1. **E0 integrity check** (prior pipeline).
2. **Ethics query** to Essex.
3. **Pin generators**, and generate the annotated subset.
4. **GPT-3.5 capture before 2026-10-23** (E2).
5. Draft annotation guidelines, including the new `L_vid` fields, and pilot.

**Venue options:**

| Option | Scope | Risk |
|---|---|---|
| **CVPR 2027 (Nov 16)** | Focused B: 2 benchmarks T1, `L_vid` 100 q/bench, E1–E7 + E10–E11, reduced E8, no E9 deep pair | Very tight; needs annotation running by ~Oct 12 and no ethics delay |
| **ICCV 2027 (~Mar 2027, dates not yet announced)** | Full B: 3 benchmarks, `L_vid` 150–200 q/bench, E9 deep pair, frame ablation, modern benchmark | Collision risk grows with time; the GPT-3.5 capture still has to happen now |
| NeurIPS 2027 D&B (~May–Jun 2027) | Dataset-centric framing | Off the stated targets; keep as backup |

---

## 12. Delta to `REQUIREMENTS_CONTRACT.md`

| Section | Change |
|---|---|
| §1.2 Thesis | Replace with the §6 research statement |
| §1.3 Contributions | C1–C5 → C1′–C4′ (§7). R-NOV-03 novel items = C1′ (dual-label set) and C3′ (system-dependent error). C3's logit test leaves the novelty list. |
| §2.1 Related work | Add: MMBench-Video, AutoEval-Video, 2607.10240, KB-VQA audit, Chandak et al., Thakur et al., One Token to Fool, CompassVerifier, Perception-Judge, Lee et al. 2511.21140, PPI family, Dorner et al., Trust-or-Escalate, Dark Current, PlaylistEval, ANCHOR, VURB/VideoRewardBench |
| §3 Hypotheses | Replace with HA–HG (§5). Remove the τ ≥ 0.9 rules. H1a becomes descriptive. H2b/H2c move to the supplement. |
| §4.1 Factors | Add blind variants (T1). Frame-count ablation (T2). Benchmarks: MSVD-QA + ActivityNet-QA T1, NExT-OE T1-if-on-track |
| §4.3 Judges | Add CompassVerifier and a frontier video judge (T1-capped). Refresh Gemma 3 → Gemma 4 / Qwen3.5. GPT-3.5 is time-boxed to Oct 23. Expect no reliable logprobs from GPT-5.x. |
| §4.6 Grids | G-SWEEP reduced (E8); **drop G-ORDER, G-FLIP**; G-STRESS reduced to refusal/hedge/sorry probes (natural, from blind outputs) |
| §5 Annotation | `L_vid` adds annotator-written valid answers, reference label "one-of-several", tag E10, split E8; blind units `L_ref` only; demonstration pool 200 → 100 per benchmark |
| §6 Analysis | Add error-budget decomposition, HC2 correction test, Δ_J − Δ_H, power-aware ranking rule |
| §15 Phases | Re-plan to 6 weeks (CVPR) or a two-stage plan (capture now, full study for ICCV) |
| §16 Decisions | Add D9 venue, D10 GPT-3.5 capture scope, D11 frontier-judge budget |

---

## 13. Decisions

**Decided 2026-10-03:**
1. Direction **B** locked.
2. Venue **ICCV 2027, full B**. Stage 1, now to Oct 23: SP0 integrity check, pin generators, generate the annotated subset and then the main sample, capture GPT-3.5 verdicts and logprobs. Stage 2: the full study (3 benchmarks, `L_vid` 150–200 q/bench, E9 deep-annotated pair, frame-count ablation). Re-run the novelty audit when ICCV dates are announced, and at T−14 days.

**Still open (defaults apply if unconfirmed):**
3. **T1 benchmarks:** default MSVD-QA + ActivityNet-QA + NExT-OE. Full scope makes all three T1.
4. **Frontier video judge budget** (D2/D11): default is cost-capped on annotated items only.

---

## 14. Must-read and unverified items

- **PlaylistEval** (arXiv 2609.34314) and **MovieRecapsQA** (arXiv 2601.02536, CVPR 2026): fetches were rate-limited. Summaries here come from the project page and search snippets.
- Venues of Chandak et al. (arXiv 2507.02856), One Token to Fool (arXiv 2507.08794) and Lee et al. (arXiv 2511.21140) were not confirmed. Cite as arXiv until verified.
- Exposure claim: one CVPR 2026 paper was verified (Du et al. supplement). SP8 must quantify the rest.
- `gpt-5.6-terra` is named as the replacement on OpenAI's deprecation page; its logprob support is not verified.
- Power numbers in §3.2 are back-of-envelope (normal approximation, 20–25% discordance). Replace them after the pilot (R-ANA-03).

---

## References (accessed 2026-10-03)

- CVPR 2027 dates: https://cvpr.thecvf.com/Conferences/2027/Dates
- OpenAI deprecations (gpt-3.5-turbo shutdown 2026-10-23): https://developers.openai.com/api/docs/deprecations
- Video-ChatGPT: https://arxiv.org/html/2306.05424v2 · issue #135: https://github.com/mbzuai-oryx/Video-ChatGPT/issues/135
- LLaVA-Hound-DPO: https://arxiv.org/html/2404.01258v1
- FreeVA: https://arxiv.org/abs/2405.07798
- MMBench-Video: https://proceedings.neurips.cc/paper_files/paper/2024/file/a2326c9715a516c91174132e0170073a-Paper-Datasets_and_Benchmarks_Track.pdf
- Beyond Raw Videos / EditVid: https://arxiv.org/html/2406.10484v1
- VideoQA in the Era of LLMs: https://arxiv.org/html/2408.04223v1
- VideoJudge: https://arxiv.org/html/2509.21451v1 · https://mlanthology.org/iclr/2026/waheed2026iclr-videojudge/
- Is Your Video LM a Reliable Judge?: https://proceedings.iclr.cc/paper_files/paper/2025/file/dc4f891373d19087d1ddda33e81e00e4-Paper-Conference.pdf
- Perception-Judge: https://arxiv.org/html/2606.02578 · https://github.com/kaist-cvml/perception-judge
- VideoRewardBench: https://arxiv.org/html/2509.00484v1 · VURB: https://arxiv.org/html/2605.07872
- PlaylistEval: https://arxiv.org/abs/2609.34314 · https://playlisteval.github.io/
- AutoEval-Video: https://arxiv.org/html/2311.14906
- Answer Matching (Chandak et al.): https://arxiv.org/pdf/2507.02856
- Judging the Judges (Thakur et al.): https://aclanthology.org/2025.gem-1.33.pdf
- One Token to Fool: https://arxiv.org/html/2507.08794v1
- CompassVerifier: https://arxiv.org/abs/2508.03686 · Pitfalls of verifiers: https://arxiv.org/html/2505.22203v1
- Short-Answer VQA evaluator instability: https://arxiv.org/html/2607.10240v1
- KB-VQA audit: https://arxiv.org/html/2607.00159v1
- Open-ended VQA benchmarking (Ging et al.): https://arxiv.org/html/2402.07270v2
- LAVE: https://arxiv.org/abs/2310.02567v2
- Breaking Down Video LLM Benchmarks: https://arxiv.org/html/2505.14321v1
- How to Correctly Report LLM-as-a-Judge Evaluations: https://arxiv.org/html/2511.21140v2
- AutoEval Done Right: https://arxiv.org/pdf/2403.07008 · Stratified PPI: https://arxiv.org/html/2406.04291 · PP-Ranking: https://arxiv.org/pdf/2402.17826
- Limits to scalable evaluation (Dorner et al.): https://arxiv.org/pdf/2410.13341
- Trust or Escalate: https://arxiv.org/html/2407.18370v1
- ANCHOR (human-anchored ranking): https://arxiv.org/html/2609.19599
- LLM Judges Have Dark Current: https://arxiv.org/html/2606.15610
- Batch Calibration: https://arxiv.org/pdf/2309.17249
- Agreeableness bias (Jain et al.): https://arxiv.org/abs/2510.11822
- EG-VQA: https://arxiv.org/html/2606.24797v1
- CVPR 2026 token compression using GPT-3.5 judge (Du et al., supp.): https://openaccess.thecvf.com/content/CVPR2026/supplemental/Du_Unified_Spatiotemporal_Token_CVPR_2026_supplemental.pdf
- OpenAI logprobs on GPT-5.x (community thread): https://community.openai.com/t/gpt-5-2-logprobs-support-removed/1378114
