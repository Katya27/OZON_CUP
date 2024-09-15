FROM python:3.10-slim
WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install imbalanced-learn && \
    python -c "from torchvision.models import efficientnet_b1; \
               efficientnet_b1(weights='EfficientNet_B1_Weights.DEFAULT')" && \
    chmod +x /app/entrypoint.sh /app/baseline.py /app/make_submission.py
