import argparse
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SAMPLE = {
    "median_income": 8.3252,
    "house_age": 41.0,
    "average_rooms": 6.984126984126984,
    "average_bedrooms": 1.0238095238095237,
    "population": 322.0,
    "average_occupancy": 2.5555555555555554,
    "latitude": 37.88,
    "longitude": -122.23,
}


def check_service(base_url: str) -> None:
    base_url = base_url.rstrip("/")
    with urlopen(f"{base_url}/ready", timeout=10) as response:
        if json.load(response)["status"] != "ready":
            raise RuntimeError("Service is not ready")
    request = Request(
        f"{base_url}/predict",
        data=json.dumps(SAMPLE).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        prediction = json.load(response)
    if prediction["predicted_median_house_value_usd"] <= 0:
        raise RuntimeError(f"Unexpected prediction: {prediction}")
    with urlopen(f"{base_url}/model", timeout=10) as response:
        metadata = json.load(response)
    if prediction["model_version"] != metadata["model_version"]:
        raise RuntimeError("Unexpected model version")
    invalid_request = Request(
        f"{base_url}/predict",
        data=b'{"median_income": -1}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(invalid_request, timeout=10):
            raise RuntimeError("Invalid input was unexpectedly accepted")
    except HTTPError as error:
        if error.code != 422:
            raise
    print(json.dumps(prediction, indent=2))
    print("PASS: readiness, dollar prediction, model version, and input validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a running model-serving API.")
    parser.add_argument("--url", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    arguments = parser.parse_args()
    check_service(arguments.url)