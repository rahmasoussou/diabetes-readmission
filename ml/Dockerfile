FROM python:3.11-slim

WORKDIR /app



RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib sqlalchemy psycopg2-binary python-dotenv
RUN pip install --no-cache-dir xgboost
RUN pip install --no-cache-dir shap
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" "python-jose[cryptography]" slowapi pydantic httpx

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]