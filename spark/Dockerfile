# Spark Structured Streaming — Medallion (Bronze → Gold → Silver)
# Runs spark/feature_engineering.py with 4 queries (bronze JSON, gold Redis, silver Parquet)
# Base: Python + OpenJDK (pyspark bundles Spark, no separate Spark install needed)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYSPARK_PYTHON=python3
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Python deps ─────────────────────────────────────────────────────────
COPY spark/requirements.txt ./spark/requirements.txt
COPY requirements.txt ./requirements.txt

# pyspark + runtime deps used by feature_engineering.py / redis_writer.py
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r spark/requirements.txt \
 && pip install --no-cache-dir redis==5.2.0 python-dotenv==1.0.1 pyyaml==6.0.2

# ── App code (copy before JAR download so layer cache works) ───────────
# spark/jars/ is excluded via .dockerignore (keeps build context small)
COPY features ./features
COPY spark ./spark
COPY producer ./producer

# ── Kafka + S3A connector JARs ──────────────────────────────────────────
# Downloaded at build time so spark/jars doesn't need to be committed.
# Versions pinned to Spark 3.5.1 / Scala 2.12 / Hadoop 3.3.4
# NOTE: this must run AFTER `COPY spark ./spark` — otherwise the COPY
# overwrites spark/jars with the (jars-excluded) host directory.
RUN mkdir -p spark/jars \
 && echo "Downloading Spark/Kafka/S3 JARs..." \
 && curl -fsSL -o spark/jars/spark-sql-kafka.jar \
        https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/spark-sql-kafka-0-10_2.12-3.5.1.jar \
 && curl -fsSL -o spark/jars/kafka-clients.jar \
        https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.6.1/kafka-clients-3.6.1.jar \
 && curl -fsSL -o spark/jars/spark-token-provider-kafka.jar \
        https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.1/spark-token-provider-kafka-0-10_2.12-3.5.1.jar \
 && curl -fsSL -o spark/jars/commons-pool2.jar \
        https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar \
 && curl -fsSL -o spark/jars/hadoop-aws.jar \
        https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
 && curl -fsSL -o spark/jars/aws-java-sdk-bundle.jar \
        https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar \
 && ls -lh spark/jars/

# Checkpoints live inside the container — survives restarts via
# Docker volume (spark_checkpoints). Without a volume, Spark reprocesses
# from latest Kafka offset on restart (acceptable for demo).
RUN mkdir -p /tmp/checkpoints/bronze_raw /tmp/checkpoints/windowed_redis \
             /tmp/checkpoints/transaction_flags_redis /tmp/checkpoints/windowed_s3

# Spark connects to kafka:29092 inside the Docker network (see docker-compose.yml)
# Override via env: KAFKA_BOOTSTRAP_SERVERS, REDIS_HOST, MINIO_ENDPOINT, etc.
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:29092
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV MINIO_ENDPOINT=http://minio:9000
ENV MINIO_ACCESS_KEY=minioadmin
ENV MINIO_SECRET_KEY=minioadmin

CMD ["python", "spark/feature_engineering.py"]
