# Deploying NetSentry to Kubernetes

Production deployment manifests for the NetSentry inference API (the FastAPI
service in `netsentry/serving`). Two equivalent paths — a **Helm chart** and raw
**Kustomize** manifests — both targeting the same hardened, autoscaled,
Prometheus-scraped deployment.

## Build the image

```bash
docker build -f docker/Dockerfile.serve -t netsentry-serve:0.5.0 .
# push to your registry, or load it into a local cluster:
kind load docker-image netsentry-serve:0.5.0   # kind
minikube image load netsentry-serve:0.5.0       # minikube
```

## Option A — Helm (recommended)

```bash
helm install netsentry deploy/helm/netsentry \
  --namespace netsentry --create-namespace \
  --set image.tag=0.5.0

# lint / preview the rendered manifests without applying:
helm lint deploy/helm/netsentry
helm template netsentry deploy/helm/netsentry | less
```

Common overrides:

| Flag | Effect |
|---|---|
| `--set image.repository=<registry>/netsentry-serve` | pull from your registry |
| `--set replicaCount=3 --set autoscaling.enabled=false` | fixed replica count |
| `--set autoscaling.maxReplicas=10` | wider autoscaling band |
| `--set serviceMonitor.enabled=true` | Prometheus Operator scraping |
| `--set model.persistence.enabled=true --set model.persistence.existingClaim=<pvc>` | serve a real trained bundle from a PVC |
| `--set apiKey.enabled=true --set apiKey.value=<key>` | require `X-API-Key` on `/predict` |

## Option B — Kustomize (no Helm)

```bash
kubectl create namespace netsentry
kubectl -n netsentry apply -k deploy/k8s
```

Edit `deploy/k8s/kustomization.yaml` to set the image registry/tag; remove
`servicemonitor.yaml` from `resources` if the cluster does not run the Prometheus
Operator.

## What the manifests give you

- **Health-gated rollout** — liveness/readiness on the app's real `/health`
  endpoint, plus a `startupProbe` that gives the first-boot bundle bootstrap time so
  a slow cold start never trips liveness.
- **Autoscaling** — a `HorizontalPodAutoscaler` (CPU-target) and a
  `PodDisruptionBudget` so scale-downs and node drains keep a replica serving.
- **Observability** — a Prometheus Operator `ServiceMonitor` scraping `/metrics`
  (the same metrics the Docker-compose Grafana dashboard renders), with pod
  annotations as a fallback for annotation-based Prometheus setups.
- **A hardened runtime** — non-root (uid 1000), `readOnlyRootFilesystem`, all
  Linux capabilities dropped, `RuntimeDefault` seccomp, and no mounted service-account
  token (the API needs no Kubernetes API access).
- **Secret-managed auth** — the optional `X-API-Key` is injected from a Kubernetes
  Secret, never baked into a manifest or image layer.

## The model bundle

By default the model volume is an `emptyDir`, so each pod runs the image's first-boot
**synthetic** bundle bootstrap (clearly labelled) and the deployment works
out-of-the-box for a demo. For real detection, build a trained
`models/serving_bundle.joblib` and mount it from a `PersistentVolumeClaim`
(`model.persistence.existingClaim`), or bake it into a downstream image layer.
