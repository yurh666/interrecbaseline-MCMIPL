FROM continuumio/miniconda3:4.9.2

WORKDIR /workspace/main_table_experiments/baselines/mcmipl_official

COPY environment.yml /workspace/main_table_experiments/baselines/mcmipl_official/environment.yml

RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "mcmipl-reproduce", "/bin/bash", "-c"]

RUN python - <<'PY'
import sys
import torch
import dgl
print("python", sys.version)
print("torch", torch.__version__)
print("dgl", dgl.__version__)
PY

CMD ["conda", "run", "-n", "mcmipl-reproduce", "bash"]
