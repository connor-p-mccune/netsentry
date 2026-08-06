# NetSentry — The Detector as a Target (MITRE ATLAS Coverage)

_Mapping pinned to ATLAS **4.5.2 (2024)**. Every coverage claim names the module, report,
and CLI command backing it, and is verified against the repository at export time — a study
that is deleted or renamed downgrades its own technique on the next run. A Navigator layer
is written alongside this report as `atlas_navigator_layer.json`._

## Why this report exists

The [ATT&CK mapping](mitre.md) answers "which adversary behaviours can this system see?".
This answers the question a security reviewer asks next: **which attacks against the
detector itself have been accounted for, and which have not?** A dozen studies in this
repository attack or defend the model, and scattered across a dozen reports they are a
collection of interesting exercises rather than a threat model. MITRE ATLAS is the
ATT&CK-shaped knowledge base for adversarial ML, so mapping onto it turns them into one
governed picture in a vocabulary a security team already reads — and, because the matrix
contains techniques nobody here has touched, it makes the **gaps** as legible as the
coverage.

## Coverage summary

| status | techniques |
|---|---|
| attack + defense, re-measured | 6 |
| control implemented (attack not simulated) | 5 |
| attack implemented | 2 |
| measured, unmitigated | 3 |
| not covered | 4 |
| out of scope | 2 |

Of 22 mapped techniques, 2 are genuinely out of scope (no language model, no interactive user surface), leaving 20 in scope. 6 carry an implemented attack **and** a defense that was re-measured afterwards, 5 carry a control that was built but never attacked here, 2 carry an implemented attack with no defense, 3 are measured but unmitigated, and 4 are not covered at all — **65% of in-scope techniques** have working code behind them.

## The matrix

| ATLAS technique | tactic | status | what NetSentry does | reproduce |
|---|---|---|---|---|
| [AML.T0015](https://atlas.mitre.org/techniques/AML.T0015) Evade ML Model | Defense Evasion | **attack + defense, re-measured** | The headline evasion result: detection under budgeted feature perturbation, before and after adversarial training, with the residual gap stated. | `netsentry harden` |
| [AML.T0018](https://atlas.mitre.org/techniques/AML.T0018) Backdoor ML Model | Persistence | **attack + defense, re-measured** | A BadNets trigger walks attacks through a model whose clean metrics stay green; spectral signatures detect the poisoned rows without knowing the trigger. The same mechanism is used constructively to watermark the model for ownership proof. | `netsentry backdoor && netsentry watermark` |
| [AML.T0020](https://atlas.mitre.org/techniques/AML.T0020) Poison Training Data | Persistence | **attack + defense, re-measured** | Label-flip and benign-pool contamination curves quantify the damage; audit-and-drop sanitization is applied and the recovery re-measured. | `netsentry poisoning && netsentry sanitize` |
| [AML.T0024.000](https://atlas.mitre.org/techniques/AML.T0024.000) Infer Training Data Membership | Exfiltration | **attack + defense, re-measured** | Shokri shadow-model and Yeom threshold attacks measure the leak, with an overfit reference model to price it; DP-SGD with a from-scratch Renyi accountant buys a formal bound and the utility-leakage frontier prices that. | `netsentry privacy && netsentry dp` |
| [AML.T0024.002](https://atlas.mitre.org/techniques/AML.T0024.002) Extract ML Model | Exfiltration | **attack + defense, re-measured** | Query-only model stealing: surrogate fidelity and stolen detection measured against the query budget, and the budget named as the defense that works. | `netsentry extraction` |
| [AML.T0043](https://atlas.mitre.org/techniques/AML.T0043) Craft Adversarial Data | ML Attack Staging | **attack + defense, re-measured** | Mimicry and adaptive query-search craft evading flows; adversarial training hardens against them and the study re-measures rather than assuming the fix worked. Randomized smoothing adds a provable per-flow radius. | `netsentry robustness && netsentry harden && netsentry certify` |
| [AML.T0002](https://atlas.mitre.org/techniques/AML.T0002) Acquire Public ML Artifacts | Resource Development | **control implemented (attack not simulated)** | The published model bundle is an artifact an adversary can acquire. Provenance attestation signs it and the verify gate refuses a bundle whose bytes changed, so an acquired artifact is at least detectably not the deployed one. | `netsentry provenance && netsentry verify` |
| [AML.T0010](https://atlas.mitre.org/techniques/AML.T0010) ML Supply Chain Compromise | Initial Access | **control implemented (attack not simulated)** | A CycloneDX SBOM plus a hashed model manifest, enforced by an integrity gate before serving; the canary then re-checks behaviour at load and at hot reload. | `netsentry verify` |
| [AML.T0024](https://atlas.mitre.org/techniques/AML.T0024) Exfiltration via ML Inference API | Exfiltration | **control implemented (attack not simulated)** | The parent technique for the two sub-techniques below; the rate limit and API-key controls on the prediction endpoints are the shared mitigation, since every variant is paid for in queries. | `netsentry serve` |
| [AML.T0031](https://atlas.mitre.org/techniques/AML.T0031) Erode ML Model Integrity | Impact | **control implemented (attack not simulated)** | Slow degradation is treated as a first-class failure: drift monitoring (PSI, KS+FDR, Page-Hinkley/DDM, a conformal test martingale with an anytime-valid false-alarm bound), retrain-trigger policy, and threshold refresh. | `netsentry driftscan && netsentry retrainpolicy` |
| [AML.T0040](https://atlas.mitre.org/techniques/AML.T0040) ML Model Inference API Access | ML Model Access | **control implemented (attack not simulated)** | The inference API is the adversary's entry point for every query-based attack below. API-key auth and a per-client rate limit bound the query budget that extraction and query-search evasion both depend on. | `netsentry serve` |
| [AML.T0005](https://atlas.mitre.org/techniques/AML.T0005) Create Proxy ML Model | Resource Development | **attack implemented** | A surrogate trained on query answers, measured for fidelity and then used to mount black-box transfer evasion against the real model. | `netsentry extraction` |
| [AML.T0042](https://atlas.mitre.org/techniques/AML.T0042) Verify Attack | ML Attack Staging | **attack implemented** | The query-search evasion attack verifies each candidate against the live decision before committing, which is exactly this technique and the reason a query budget is the defender's most effective lever. | `netsentry robustness` |
| [AML.T0029](https://atlas.mitre.org/techniques/AML.T0029) Denial of ML Service | Impact | **measured** | Serving cost is measured (benchmark percentiles, the SHAP share of latency, the cascade's load reduction) and the rate limiter bounds per-client volume, but no resource-exhaustion attack is implemented or defended against end to end. | `netsentry benchmark && netsentry cascade` |
| [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044) Full ML Model Access | ML Model Access | **measured** | The white-box case is treated as the worst case throughout: the evasion study's mimicry attack and the certification study both assume full knowledge of the model, so the reported robustness is a floor, not a best case. | `netsentry certify` |
| [AML.T0046](https://atlas.mitre.org/techniques/AML.T0046) Spamming ML System with Chaff Data | Impact | **measured** | An adversary who floods the queue with near-threshold traffic attacks the analysts, not the model. The SOC queue simulation and alert-queue capacity study quantify what that does to the attack SLA; no mitigation is implemented. | `netsentry socsim` |
| [AML.T0011](https://atlas.mitre.org/techniques/AML.T0011) User Execution | Execution | **out of scope** | Requires a human operator to be induced into running adversary-supplied content. There is no interactive user surface in the serving path; the only human in the loop is an analyst reading alerts. | — |
| [AML.T0051](https://atlas.mitre.org/techniques/AML.T0051) LLM Prompt Injection | Initial Access | **out of scope** | NetSentry has no language model and no prompt surface anywhere in the pipeline. Recorded as out of scope rather than omitted, so the matrix reads honestly. | — |
| [AML.T0013](https://atlas.mitre.org/techniques/AML.T0013) Discover ML Model Ontology | Discovery | **not covered** | The API returns class names, SHAP feature attributions, MITRE context, and conformal prediction sets — a rich description of the model's ontology, offered deliberately because explanations are a product requirement. The trade-off is named in the extraction study but not defended against. | — |
| [AML.T0014](https://atlas.mitre.org/techniques/AML.T0014) Discover ML Artifacts | Discovery | **not covered** | Locating model files, MLflow runs, and training data on a compromised host. Infrastructure hardening, not modelling; noted here so the gap is on the record. | — |
| [AML.T0025](https://atlas.mitre.org/techniques/AML.T0025) Exfiltration via Cyber Means | Exfiltration | **not covered** | Stealing the model file off disk or out of the registry rather than through the API. This is host and IAM security, outside what a detection pipeline controls; the signed manifest makes tampering detectable but does not prevent theft. | — |
| [AML.T0034](https://atlas.mitre.org/techniques/AML.T0034) Cost Harvesting | Impact | **not covered** | Driving inference spend up by querying an expensive endpoint. The rate limit bounds it incidentally, but no cost model, quota, or per-tenant accounting exists and nothing here measures the attack. | — |

## Residual risk — the part that matters

A threat model that lists only what you did is marketing. These are the in-scope techniques
with no implemented defense, stated plainly so a reviewer does not have to infer them from
absence:

| technique | tactic | status | residual risk |
|---|---|---|---|
| [AML.T0013](https://atlas.mitre.org/techniques/AML.T0013) Discover ML Model Ontology | Discovery | not covered | The API returns class names, SHAP feature attributions, MITRE context, and conformal prediction sets — a rich description of the model's ontology, offered deliberately because explanations are a product requirement. The trade-off is named in the extraction study but not defended against. |
| [AML.T0014](https://atlas.mitre.org/techniques/AML.T0014) Discover ML Artifacts | Discovery | not covered | Locating model files, MLflow runs, and training data on a compromised host. Infrastructure hardening, not modelling; noted here so the gap is on the record. |
| [AML.T0025](https://atlas.mitre.org/techniques/AML.T0025) Exfiltration via Cyber Means | Exfiltration | not covered | Stealing the model file off disk or out of the registry rather than through the API. This is host and IAM security, outside what a detection pipeline controls; the signed manifest makes tampering detectable but does not prevent theft. |
| [AML.T0034](https://atlas.mitre.org/techniques/AML.T0034) Cost Harvesting | Impact | not covered | Driving inference spend up by querying an expensive endpoint. The rate limit bounds it incidentally, but no cost model, quota, or per-tenant accounting exists and nothing here measures the attack. |
| [AML.T0029](https://atlas.mitre.org/techniques/AML.T0029) Denial of ML Service | Impact | measured | Serving cost is measured (benchmark percentiles, the SHAP share of latency, the cascade's load reduction) and the rate limiter bounds per-client volume, but no resource-exhaustion attack is implemented or defended against end to end. |
| [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044) Full ML Model Access | ML Model Access | measured | The white-box case is treated as the worst case throughout: the evasion study's mimicry attack and the certification study both assume full knowledge of the model, so the reported robustness is a floor, not a best case. |
| [AML.T0046](https://atlas.mitre.org/techniques/AML.T0046) Spamming ML System with Chaff Data | Impact | measured | An adversary who floods the queue with near-threshold traffic attacks the analysts, not the model. The SOC queue simulation and alert-queue capacity study quantify what that does to the attack SLA; no mitigation is implemented. |

Three of these are honest scope boundaries rather than oversights: model theft off disk and
artifact discovery on a compromised host are infrastructure and IAM problems that a
detection pipeline does not control, and ontology disclosure is a **deliberate** trade — the
API returns SHAP attributions, class names, ATT&CK context and conformal prediction sets
because explanations are a product requirement here, and the
[extraction study](extraction.md) prices what that costs rather than pretending it is free.
The genuine gaps are cost harvesting (no quota or per-tenant accounting exists) and chaff
flooding, where the [SOC simulation](socsim.md) measures the damage to the analyst queue but
nothing mitigates it.

## How this stays true

The failure mode of every security mapping is that it is written once and then drifts from
the system it describes. Two mechanisms guard against that here. Each claim carries its
evidence — a module path, a report path, and the command that regenerates it — and the
exporter checks those paths exist, downgrading any claim whose code has moved. And the
technique identifiers carry the ATLAS version they were taken from plus a link to the live
entry, because ATLAS revises its matrix; treat the IDs as a pinned snapshot and re-check
them against [the live matrix](https://atlas.mitre.org/matrices/ATLAS) before quoting them
anywhere that matters.

## Scope

The mapping is curated by hand: it reflects a considered reading of which ATLAS techniques
this system's work corresponds to, not an automated derivation, and reasonable people could
grade some entries differently — particularly the line between "measured" and "defended",
which this report draws at *was the defense re-measured after being applied*. Grades describe
the repository's engineering, not an operational assurance: an implemented attack proves the
capability was exercised on this synthetic stand-in, not that a production deployment is
resistant to it. Sub-techniques are mapped only where a study addresses one specifically.