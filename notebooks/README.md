# Notebooks

`pipeline.ipynb` runs the whole pipeline in one notebook: data, samples, frames, answers, judging,
labelling with Claude, and the report. Every section explains what it does and what to change for the
Mac or the GPU server. It does not import the `vjaudit` package.

Open it with the `dl` environment as the kernel:

```bash
cd vjauditv2/notebooks
conda activate dl
pip install -e "..[mac,notebook]"     # once (on the GPU server: "..[gpu,notebook]")
jupyter lab pipeline.ipynb
```

Start with section 0 of the notebook. Results go to `notebooks/runs/<profile>/` (ignored by git).
