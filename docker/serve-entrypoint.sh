#!/usr/bin/env sh
# Serve the API. If no model bundle is mounted, build a clearly-labelled
# synthetic demo bundle on first start so `docker compose up` works standalone.
set -e

if [ ! -f models/serving_bundle.joblib ]; then
  echo "[netsentry] No serving bundle found — building a SYNTHETIC demo bundle..."
  netsentry download
  netsentry prep
  python -c "from netsentry.config import load_settings; from netsentry.serving.bundle import build_serving_bundle; build_serving_bundle(load_settings())"
fi

exec uvicorn netsentry.serving.app:create_app --factory \
  --host 0.0.0.0 --port "${NETSENTRY_SERVING__PORT:-8000}"
