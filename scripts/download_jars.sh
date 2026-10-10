#!/usr/bin/env bash
# Download the Kafka + S3A connector JARs that spark/feature_engineering.py loads
# from spark/jars/. They're gitignored (too large to commit), so a fresh clone
# needs this once. Same versions as spark/Dockerfile (Spark 3.5.1 / Scala 2.12).
# Usage: ./scripts/download_jars.sh   (or: make jars)
set -euo pipefail

JAR_DIR="$(cd "$(dirname "$0")/.." && pwd)/spark/jars"
MAVEN="https://repo1.maven.org/maven2"
mkdir -p "$JAR_DIR"

fetch() {  # fetch <local-name> <maven-path>
  local dest="$JAR_DIR/$1"
  if [ -s "$dest" ]; then
    echo "✓ $1 (already present)"
  else
    echo "↓ $1"
    curl -fSL --retry 3 -o "$dest.part" "$MAVEN/$2" && mv "$dest.part" "$dest"
  fi
}

fetch spark-sql-kafka.jar            org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/spark-sql-kafka-0-10_2.12-3.5.1.jar
fetch kafka-clients.jar              org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar  # what spark-sql-kafka 3.5.1 is built against
fetch spark-token-provider-kafka.jar org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.1/spark-token-provider-kafka-0-10_2.12-3.5.1.jar
fetch commons-pool2.jar              org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar
fetch hadoop-aws.jar                 org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
fetch aws-java-sdk-bundle.jar        com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar  # ~280 MB

echo "JARs ready in $JAR_DIR"
