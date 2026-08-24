FROM ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

RUN python3 -m pip install --no-cache-dir --no-deps "peft==0.20.0"

WORKDIR /workspace/CHELATEDAI
COPY pyproject.toml README.md ./
COPY egv ./egv
RUN python3 -m pip install --no-deps -e .

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

ENTRYPOINT ["python3", "-m", "egv"]
