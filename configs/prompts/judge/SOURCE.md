# Judge template sources

- `t_vcg_system.txt`, `t_vcg_user.txt`: the Video-ChatGPT GPT-3.5 evaluation prompt (system string and
  user string with `{question}`, `{answer}`, `{pred}` slots), copied from
  `mbzuai-oryx/Video-ChatGPT/quantitative_evaluation/evaluate_activitynet_qa.py`. Confirm against the
  upstream file with `uv run python -m vjaudit.cli judge verify-vcg --upstream <file> --commit <sha>`.
- `t_rub.txt`: the BlackboxNLP three-class rubric (Direct Match / Indirect Context / Different Context →
  Yes/No), zero-shot, ending with "Answer with a single word: Yes or No.". The shipped file is a
  reconstruction from the paper's description; replace it with the prior repo's template using
  `uv run python -m vjaudit.cli judge vendor-rub --prior <LLM_as_judge_base.j2> --var-map q=question,...`.
- Slots are filled in one pass (`{question}`, `{answer}`, `{pred}` only); other braces stay literal.
