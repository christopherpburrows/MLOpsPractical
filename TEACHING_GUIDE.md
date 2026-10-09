# Instructor Guide: From Real Data to a Containerized ML Microservice

## 1. Scenario and Learning Outcomes

**Scenario:** A property-research application needs an estimated median house
value for a neighborhood. A client supplies income, house age, average rooms,
population, and location. Our model learns from real areas with known median
values, then serves estimates through an HTTP microservice.

We use historical California Housing census data. **One row is a census block
group, not one house.** The data is from 1990; this model does not estimate
today's market price or provide a property appraisal. It is a straightforward
real-data example of regression and the training-to-serving workflow.

By the end, students should be able to:

1. Read real training rows, separate inputs from the target, and save a model.
2. Compare Linear Regression, Random Forest, and Gradient Boosting using dollar errors.
3. Define and test an HTTP contract for an independently runnable model microservice.
4. Build a Docker image and deploy/check it using Compose dependency ordering.
5. Explain what remains necessary before any production use.

Suggested audience: students with basic Python, functions, dictionaries, and terminal
experience. Plan three 60-90 minute labs, plus installation time. Have Python 3.12,
VS Code, internet for package/image downloads, and Docker Desktop available.

## 2. Dataset and Feature Contract

Scikit-Learn downloads and caches the **California Housing** dataset using
`fetch_california_housing()`. The first run needs internet; subsequent local
runs use the cache in your user profile's `scikit_learn_data` folder. No account,
API key, or manually obtained CSV is needed. No synthetic data is substituted.

- Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Dataset background: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- Source: 1990 US Census; 20,640 area records and eight numeric inputs.
- Target: known median house value, originally in units of $100,000.
- Training converts targets to dollars before fitting, evaluation, and CSV export.

| Request Field | Meaning |
| --- | --- |
| `median_income` | Area median income in $10,000 units; 3.2596 means $32,596 |
| `house_age` | Median house age in the area, in years |
| `average_rooms` | Average rooms per household; not one house's room count |
| `average_bedrooms` | Average bedrooms per household |
| `population` | Number of people in the area |
| `average_occupancy` | Average people per household |
| `latitude` | Latitude in degrees |
| `longitude` | Longitude in degrees; California values are negative |

All eight fields must be finite JSON numbers. The first six must be positive.
Latitude must be between -90 and 90, and longitude between -180 and 180.
These are basic input checks, not proof that a location is represented in the
training data or that a prediction is reliable there.

**Actual training data:** open [artifacts/training_data.csv](artifacts/training_data.csv).
Here are its first three rows with selected columns rounded for teaching:

| Income in Dollars | Age | Average Rooms | Known Median Value |
| --- | --- | --- | --- |
| $32,596 | 33 | 5.02 | $103,000 |
| $38,125 | 49 | 4.47 | $382,100 |
| $41,563 | 4 | 5.65 | $172,600 |

Ask: why is the second area more expensive despite fewer rooms? Location and
other features matter; the model does not learn a universal price-per-room rule.
The full CSV preserves all eight inputs without rounding.

## 3. The End-to-End Flow

```text
Historical census areas with known prices
          |
          +--> Random split: 16,512 train / 4,128 held-out test
                       |
             Compare three models using TRAIN-only cross-validation
                       |
             Select lowest CV MAE; fit each model on TRAIN
                       |
             Evaluate once on held-out TEST
                       |
               artifacts/model.joblib
                       |
               FastAPI loads once at startup
                       |
    JSON area features -> validation -> regressor -> estimated dollar value
                       |
       Docker image -> Compose API -> dependent smoke-check service
```

The **model** is the learned predictor. An **artifact** is a saved output of
training. **Serving** means running inference for callers. A **microservice** is
an independently runnable service with a focused responsibility and a network
contract. A microservice does not require a database or many other services.

## 4. Setup Before the First Lab

Open this project folder in VS Code. Run all commands from this folder.
For a fresh Windows machine:

```powershell
py -3.12 -m venv .venv
$Python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $Python -m pip install -r requirements-dev.txt
```

Select `.venv` using **Python: Select Interpreter** in VS Code. Calling its
executable directly avoids PowerShell activation-policy problems. In every new
terminal, set `$Python` again. The existing instructor workspace already has
this environment and these packages installed.

For Linux/macOS, use:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

For subsequent commands on Linux/macOS, replace `& $Python` with `python`.
Direct dependency versions are pinned in [requirements.txt](requirements.txt).
[requirements-dev.txt](requirements-dev.txt) adds the test client and test runner.
This is not a full transitive dependency lock or a security-audited production build.

## 5. Lab 1: Write, Train, and Evaluate the Model

**Program:** [train.py](train.py)

### Step 1: Identify Inputs and Labels

Discuss supervised learning: each training row has area measurements and a known
price. The eight input columns are `X`; `median_house_value_usd` is the target `y`.
Because the target is a number, this is **regression**, not classification.
`FEATURE_KEYS` records input order; the known price is never an input feature.

Explain `dataset.target * 100_000`: a raw target of 1.03 becomes $103,000.
The exported CSVs and API use dollars consistently.

### Step 2: Reserve Unseen Data

`train_test_split` reserves 20% of areas for evaluation: 16,512 training rows
and 4,128 test rows. `random_state=42` makes the split repeatable.
Never tune the model using the test rows. This random split is appropriate for
the lesson, but neighboring areas can be correlated. A real deployment to new
regions needs geographic holdouts; future prices require newer/time-held-out data.

### Step 3: Understand the Three Models

**Model 1: Linear Regression** (`LinearRegression`)

This learns an intercept plus one weight for each feature. Its prediction is
a weighted sum: $\hat{y} = b + \sum_{j=1}^{8} w_j x_j$.
For intuition, an illustrative one-input rule might be
`estimated_value = 50000 + 40000 * median_income`; these are not the learned
coefficients of our actual eight-feature model.

It is fast and useful as an interpretable first learned model. However, it assumes
an additive linear relationship and cannot naturally express complex location
boundaries or feature interactions. Coefficients describe the fitted relationship,
not causal effects. Strongly correlated features complicate their interpretation.
Our pipeline scales inputs before regression for numerical conditioning; scaling
does not turn a linear model into a nonlinear one. The scaler fits within each
training fold, preventing leakage.

**Model 2: Random Forest** (`RandomForestRegressor`)

A decision tree asks questions such as "is income below this threshold?" and
"is latitude above that threshold?" and predicts a value from the resulting leaf.
A forest combines many trees. Here, 50 trees learn from bootstrap-resampled
training rows, with maximum depth 16, then their predictions are averaged.
For illustration, predictions of $180,000, $210,000, and $195,000 average to
$195,000; the actual model averages 50 trees.

The forest can represent nonlinear relationships and feature interactions,
usually improving on a simple linear model for complex tabular data. Averaging
reduces the instability of individual trees, but does not eliminate overfitting.
It is larger and less directly interpretable than linear regression. Tree-based
models do not need feature standardization here and are weak extrapolators beyond
the prices and regions represented in training.

**Model 3: Histogram Gradient Boosting** (`HistGradientBoostingRegressor`)

This also combines trees, but trains them **sequentially**, rather than averaging
independently fitted trees. Each added tree improves the current combined
prediction. With squared-error loss, this can be understood as learning remaining
residual errors. For example, if the current estimate is $180,000 and the true
price is $200,000, later stages can learn an upward correction; one example's
residual is not guaranteed to be corrected exactly.

Our model uses up to 100 boosting iterations and `learning_rate=0.1`, which
shrinks each stage's update. Histogram binning groups numeric values into bins
to make tree construction efficient. It can capture nonlinear structure and
often performs well on tabular data, but tuning and evaluation still matter.
It does not need a scaler here. No model guarantees accurate current property prices.

### Step 4: Cross-Validate on Training Data Only

Three-fold cross-validation fits three fresh copies of **each candidate** using
only training rows. Each candidate receives exactly the same folds and inputs.
The program reports mean absolute error in dollars. Scikit-Learn's scorer returns
negative MAE because higher scores are conventionally better; the code negates
these scores to report ordinary positive errors. Select the candidate with the
lowest mean validation MAE **before looking at test metrics**. Then fit each
candidate on all 16,512 training rows and report held-out errors for comparison.
The selected pipeline becomes the API's model; the three predictions are not
combined. Do not repeatedly tune against the reported test results.

### Step 5: Compare Against a Baseline

A `DummyRegressor` always predicts the mean training price. It ignores inputs.
For example, it would give the same estimate to expensive and inexpensive areas.
The model should make smaller errors than this deliberately simple baseline.

### Step 6: Run Training and Inspect Outputs

```powershell
& $Python train.py
Get-Content artifacts\training_data.csv -TotalCount 6
Get-Content artifacts\test_data.csv -TotalCount 6
Get-Content artifacts\metrics.json
Get-Content artifacts\metadata.json
Import-Csv artifacts\model_comparison.csv | Format-Table
Get-Content artifacts\prediction_examples.json
```

| Output | Contents |
| --- | --- |
| `artifacts/model.joblib` | Fitted regression pipeline and serving metadata |
| `artifacts/training_data.csv` | 16,512 actual training examples, inputs plus known price |
| `artifacts/test_data.csv` | 4,128 held-out examples, never used for fitting |
| `artifacts/metrics.json` | Baseline, CV scores, MAE, RMSE, and R-squared |
| `artifacts/metadata.json` | Version, feature order, units/target description, timestamp |
| `artifacts/prediction_examples.json` | Five held-out actual prices and model predictions |
| `artifacts/sample_request.json` | Input fields from the first test row, ready for the API |
| `artifacts/model_comparison.csv` | One row per model, validation/test metrics, selection flag |
| `artifacts/models/linear_regression.joblib` | Fitted scaler and linear regression pipeline |
| `artifacts/models/random_forest.joblib` | Fitted random forest pipeline |
| `artifacts/models/histgb.joblib` | Fitted gradient boosting pipeline |

Use an alternate output directory with `& $Python train.py --output-dir artifacts-demo`.
The API uses the default directory unless `MODEL_PATH` is configured.

### Step 7: Interpret the Observed Evaluation

With this implementation and pinned dependencies:

| Model | CV MAE | Test MAE | Test RMSE | Test R-squared |
| --- | --- | --- | --- | --- |
| Linear Regression | $52,925 | $53,320 | $74,558 | 0.5758 |
| Random Forest | $34,429 | $33,293 | $51,081 | 0.8009 |
| Gradient Boosting | **$31,972** | **$30,960** | **$46,422** | **0.8355** |

Gradient Boosting wins on the selection criterion: lowest **training CV MAE**.
Compared with linear regression, it reduces test MAE by approximately $22,360
for this run. That does not mean boosting always wins. Complexity is useful only
when the data and evaluation support it. The same data, split, and metric make
this a fair comparison of these configurations, not an exhaustive tuning study.

For the selected model:

| Measure | Result | Interpretation |
| --- | --- | --- |
| Mean-price baseline MAE | $90,607 | Average absolute error from guessing one mean price |
| Model MAE | $30,960 | Average absolute difference between actual and predicted price |
| Model RMSE | $46,422 | Dollar error measure that penalizes large misses more strongly |
| Model R-squared | 0.8355 | Relative squared-error reduction versus the test-set mean |
| Training CV mean MAE | $31,972 | Average dollar error across three training folds |

For the first held-out record, actual value is **$47,700**, predicted value is
**$58,812.23**, and absolute error is **$11,112.23**. The second record is a larger
miss: actual **$45,800**, predicted **$100,228.04**. Open the examples file to compare.

MAE averages `abs(actual - predicted)`. RMSE takes the square root of the average
squared error. R-squared is `1 - SSE / SST`; it is not "83.55% accurate" and can
be negative for poor models. MAE is not a guaranteed maximum error or a prediction
interval. The dataset's top-coded values around $500,001 also limit what the model
can learn about expensive areas.

**Discussion:** does a smaller average error mean every area's estimate is good?
Would a 1990 model work for prices today? Why should evaluation on unseen regions
be different from randomly held-out nearby neighborhoods?

### Lab 1 Exercises and Completion Check

1. Pick a training row and identify its eight inputs and one target.
2. Convert its income to dollars and explain why latitude/longitude matter.
3. Calculate absolute errors for three prediction examples manually.
4. Explain why passing the known target into the API would be target leakage.
5. Compare different model settings using training CV, not repeated test tuning.
6. Explain the difference between averaging trees and adding corrective trees.
7. Compare the three models' validation errors, then justify the chosen artifact.

To compare predictions for the same exported request, run this in a Python REPL
or a separate notebook cell from the project folder:

```python
import json
from pathlib import Path
import joblib

artifacts = Path("artifacts")
request = json.loads((artifacts / "sample_request.json").read_text())
for model_path in sorted((artifacts / "models").glob("*.joblib")):
  bundle = joblib.load(model_path)
  features = [[request[key] for key in bundle["metadata"]["feature_keys"]]]
  estimate = bundle["model"].predict(features)[0]
  print(f"{bundle['metadata']['model_name']}: ${estimate:,.2f}")
```

Only load artifacts you trained or trust. These files are Python deserialization
formats, not plain data files.

Completion: the CSVs and model exist; students can explain inputs versus target,
regression, the train/test split, baseline comparison, and dollar error.

## 6. Lab 2: Write a REST API for Model Serving

**Program:** [api.py](api.py). **Verification client:** [smoke_test.py](smoke_test.py).

### Step 1: Define the HTTP Contract

| Method and Path | Purpose | Normal Status |
| --- | --- | --- |
| `GET /health` | Process liveness | 200 |
| `GET /ready` | Model readiness | 200; 503 if model is not loaded |
| `GET /model` | Feature order, model version, intended-use metadata | 200 |
| `POST /predict` | Area features to an estimated median value in dollars | 200 |
| `GET /docs` | Interactive Swagger API documentation | 200 |
| `GET /openapi.json` | Machine-readable API specification | 200 |

`GET` reads information. `POST` sends a JSON body for computation. Set
`Content-Type: application/json`. Missing/extra fields, nonnumeric or nonfinite
values, nonpositive first-six features, and invalid coordinates return 422.
Negative longitude is valid and expected for California.

### Step 2: Understand Startup and Prediction

FastAPI's lifespan handler loads the saved pipeline once, before accepting
requests. Startup fails clearly if the artifact is missing or the saved
Scikit-Learn version or feature contract differs. A missing model is not silently retrained.

Pydantic validates the request. The API builds the feature array using saved
feature order, calls `predict`, and returns an estimated dollar value rounded to
two decimal places, model version, and a historical-data notice. Synchronous handlers
run in FastAPI's thread pool; this is a simple CPU-inference demonstration, not
a throughput benchmark. HTTP requests never refit the model.

### Step 3: Start the Microservice

```powershell
& $Python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. Open http://127.0.0.1:8000/docs and use **Try it out**
on `POST /predict`. `api:app` means import the `app` object from the `api` module.
Uvicorn is the server hosting the FastAPI application. Stop it with Ctrl+C.

### Step 4: Send a Real-Dataset Example

In a second PowerShell terminal, from this same folder:

```powershell
$body = Get-Content artifacts\sample_request.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
& .\.venv\Scripts\python.exe smoke_test.py
```

The request file contains inputs from the first held-out record and no known
price. Its actual value is in the test CSV, allowing students to compare it with
the response. The smoke checker separately uses the public dataset's first row
to check HTTP behavior; it is not an independent model evaluation.

For the exported request, the expected response is:

```json
{
  "predicted_median_house_value_usd": 58812.23,
  "model_version": "housing-histgb-v1",
  "notice": "Historical census-area estimate for teaching, not a current property appraisal."
}
```

### Step 5: Test the Training-to-Serving Contract

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

[tests/test_workflow.py](tests/test_workflow.py) creates temporary artifacts and
checks training/test CSVs, saved-model/API prediction agreement, exported sample
requests, validation errors, nonfinite values, operations endpoints, documentation,
and missing/incompatible-model startup. Three-model comparison, selection, and
individual saved artifacts are also checked. There are 18 checks. `TestClient` exercises
the application without opening a port; the smoke client checks real HTTP.

For a nondefault artifact, set `$env:MODEL_PATH` to its full trusted path before
starting Uvicorn. Remove it with `Remove-Item Env:MODEL_PATH` when finished.
**Never load untrusted joblib/pickle files:** deserialization can execute code.

### Lab 2 Exercises and Completion Check

1. Send a request with a missing field, then with a string instead of a number.
2. Change JSON property order and explain why the prediction remains unchanged.
3. Explain liveness versus readiness and why startup must load the model first.
4. Describe how a separate application can call the API without training code.

Completion: a valid request returns 200; invalid input returns 422; the smoke
client passes; students can distinguish training from inference.

## 7. Lab 3: Containerize and Deploy with Docker Compose

**Deployment program:** [Dockerfile](Dockerfile) and [compose.yaml](compose.yaml).

### Step 1: Explain Images, Containers, and Compose

An **image** packages code, dependencies, and files. A **container** is a running
instance of that image, sharing the host kernel rather than owning a whole VM.
**Compose** describes service builds, configuration, networking, and lifecycle.
Python packages are managed with pip; Compose manages service dependencies.

### Step 2: Walk Through the Dockerfile

1. `dependencies`: use Python 3.12 slim and install the pinned runtime packages.
2. Run Lab 1 locally first to generate a trusted model using the pinned packages.
3. `serving`: reuse dependencies and copy serving code, the smoke client, and the
  local model, metadata, and metrics. No training script or dataset download is
  needed in the final image. The training/test CSVs remain available on the host.
4. `USER 10001:10001`: run without root privileges; model files are readable.
5. `EXPOSE 8000`: document the container port; this alone does not publish it.
6. `HEALTHCHECK`: call `/ready` using Python's standard library, without curl.
7. `CMD`: start Uvicorn on `0.0.0.0`, so it is reachable through container networking.

[.dockerignore](.dockerignore) excludes the local virtual environment, tests,
and most artifacts, but explicitly includes the model, metadata, and metrics.
The build requires these three files from Lab 1. Dependency layers can be cached.
After changing training code or data, rerun training before rebuilding the image;
the changed model file invalidates its Docker copy layer.

This separates training from deployment: train, evaluate, then package the chosen
artifact. In production, approve/version that artifact and record its content hash.
Do not train independently in every replica. Packaging a local artifact also
avoids dataset-host certificate failures inside the container build; it does not
disable TLS verification or repair the container's certificate trust store.

### Step 3: Verify Docker Desktop

Start Docker Desktop with Linux containers and wait for it to report readiness:

```powershell
docker version
docker info
docker compose version
```

`docker --version` proves only that the CLI exists. `docker info` must connect to
the engine. If the Linux-engine pipe is missing, the engine is not running.
On managed machines, an administrator may need to enable the WSL 2 prerequisites.

### Step 4: Build and Run a Single Container (Optional)

Stop the local server first so port 8000 is free:

```powershell
& .\.venv\Scripts\python.exe train.py
docker build --target serving -t housing-model-api:teaching .
docker run --rm -p 127.0.0.1:8000:8000 --name housing-api housing-model-api:teaching
```

Open http://127.0.0.1:8000/docs, then run the local smoke client in another
terminal. Ctrl+C stops and removes this example container.

### Step 5: Deploy Using Compose

```powershell
& .\.venv\Scripts\python.exe train.py
docker compose config --quiet
docker compose up --build -d --wait model-api
docker compose ps
docker compose --profile checks run --rm smoke-check
docker compose logs model-api
```

`model-api` stays running in the background. Its root filesystem is read-only;
`/tmp` is writable temporary memory storage. Linux capabilities are dropped and
new privileges are disabled. The model is immutable inside the image.
`--wait` waits for readiness; use a current Docker Compose v2 release.

The optional `smoke-check` service depends on `model-api` being **healthy**, not
merely started. It then runs the same verification client in a second container.
Its URL is `http://model-api:8000`: Compose resolves the service name through
internal DNS. Inside a container, `localhost` means that container, not the API
container or the Windows host. The checker exits successfully after verification.

The checks profile keeps this one-shot client out of ordinary startup. It is not
a long-running production dependency. No database is needed for this stateless
inference example; adding one would obscure the core learning objective.

### Step 6: Handle a Port Conflict and Clean Up

The host binding is deliberately localhost-only for classroom safety. To use
another host port, set the variable in the same terminal before deployment:

```powershell
$env:API_PORT = "8001"
docker compose up --build -d --wait model-api
& .\.venv\Scripts\python.exe smoke_test.py --url http://127.0.0.1:8001
docker compose --profile checks run --rm smoke-check
docker compose down
Remove-Item Env:API_PORT
```

Then browse http://127.0.0.1:8001/docs. The internal port remains 8000, so the
containerized smoke-check needs no change. To stop the default deployment:

```powershell
docker compose down
```

### Lab 3 Exercises and Completion Check

1. Explain why binding Uvicorn to `127.0.0.1` inside its container breaks access.
2. Explain why `EXPOSE` is different from Compose's `ports` mapping.
3. Explain why starting a process is not the same as model readiness.
4. Retrain after changing training code, rebuild, and compare artifact metadata.
5. Explain why packaging one approved artifact is preferable to training in every replica.

Completion: the API container is healthy; the checker exits with status 0;
students can explain image versus container, internal DNS, and dependencies.

## 8. Troubleshooting

| Symptom | Action |
| --- | --- |
| Python 3.12 not found | Install Python 3.12, or use `py -3.13` for the local environment |
| Import error | Install requirements using this project's `.venv` Python |
| PowerShell blocks activation | Use the explicit executable commands; no activation is required |
| Model missing at API startup | Run training first; check `MODEL_PATH` if set |
| Scikit-Learn version mismatch | Use pinned packages and retrain with that same environment |
| Port already in use | Stop the other server, or use port 8001 and update the client URL |
| API returns 422 | Check exact keys, positive numeric values, and required fields |
| Docker engine pipe missing | Start Docker Desktop in Linux-container mode; verify `docker info` |
| Image download/install fails | Check internet, proxy, registry access, and Docker Desktop settings |
| Docker COPY reports missing model files | Run Lab 1 locally before building; keep the artifact exceptions in `.dockerignore` |
| Old trainer stage fails with CERTIFICATE_VERIFY_FAILED | Use the updated serving-only build with local artifacts; never disable TLS verification |
| Housing dataset download fails | Check network/proxy access to the dataset host; the first download needs internet |
| API still shows old fields | Restart Uvicorn after retraining; refresh `/docs` |
| Checker cannot reach API | Inspect `docker compose ps` and API logs; use service DNS inside containers |
| Uvicorn module not found | Run from this project folder and use the `.venv` executable |

## 9. Instructor Assessment and Production Boundaries

Suggested grading: data/split explanation 20%, correct regression model 20%, metrics
interpretation 20%, working API and validation 20%, container deployment explanation
20%. Require students to justify design choices, not merely show successful commands.

Before real production use, obtain current data and address geographic holdouts,
performance across regions and income groups, bias, price top-coding, prediction
uncertainty, economic distribution shift, and human review of consequential decisions.
Technical requirements also include authentication, HTTPS, rate limits, privacy,
artifact provenance, full environment locks, vulnerability scanning, monitoring,
latency/load tests, retraining policy, deployment rollback, and reproducible CI/CD.
The version string here is a teaching label, not a model registry or content hash.

**Verification boundary:** three-model training and the 18 automated checks were run.
The Compose configuration, Docker image build, healthy API deployment,
container-to-container smoke check, and host-port example prediction were also
verified. The running API on port 8000 is now served by the Docker container.