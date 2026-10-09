import os
import csv
import time
import hmac
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from flask import Flask, render_template, jsonify, request

try:
    import psutil
except ImportError:
    psutil = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

LOCAL_CSV = os.path.join(DATA_DIR, "ai_measurements.csv")
CLOUD_CSV = os.path.join(DATA_DIR, "cloud_benchmark_results.csv")

os.makedirs(DATA_DIR, exist_ok=True)

REQUEST_TIMEOUT = 20
COOLDOWN_SECONDS = 30

benchmark_lock = threading.Lock()
csv_lock = threading.Lock()
last_benchmark_time = 0

PROMPT = (
    "Explain binary search in simple terms in 2 to 3 sentences. "
    "Include its time complexity."
)

PROVIDERS = {
    "OpenRouter": {
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "nvidia/nemotron-3.5-lightning:free",
    },
    "Google Gemini": {
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.8-flash",
    },
    "Groq": {
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
    },
}


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def get_model(provider):
    config = PROVIDERS[provider]
    return os.environ.get(
        config["model_env"], config["default_model"]
    )


def get_host_metrics():
    """Measure the host running this Flask application."""
    metrics = {
        "server_cpu_percent": None,
        "server_ram_percent": None,
    }

    if psutil is None:
        return metrics

    try:
        metrics["server_cpu_percent"] = psutil.cpu_percent(
            interval=0.1
        )
        metrics["server_ram_percent"] = (
            psutil.virtual_memory().percent
        )
    except Exception:
        pass

    return metrics


def empty_result(provider):
    return {
        "timestamp": now_utc(),
        "environment": "Render",
        "provider": provider,
        "model": get_model(provider),
        "prompt_id": "binary_search",
        "response_time_sec": None,
        "server_cpu_percent": None,
        "server_ram_percent": None,
        "data_sent_mb": None,
        "data_received_mb": None,
        "input_tokens": None,
        "output_tokens": None,
        "tokens_per_sec": None,
        "status": "ERROR",
        "error": "",
    }


def make_request(url, payload, headers=None, params=None):
    response = requests.post(
        url,
        json=payload,
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    # Estimate body sizes. This does not include all HTTP headers
    # or all network protocol overhead.
    request_body_bytes = len(
        requests.Request("POST", url, json=payload).prepare().body or b""
    )
    response_body_bytes = len(response.content)

    response.raise_for_status()

    return (
        response,
        request_body_bytes,
        response_body_bytes,
    )


def run_provider(provider):
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["key_env"])
    model = get_model(provider)

    result = empty_result(provider)
    start = time.perf_counter()

    if not api_key:
        result["error"] = (
            f"Missing Render environment variable: {config['key_env']}"
        )
        result.update(get_host_metrics())
        return result

    try:
        if provider == "OpenRouter":
            url = "https://openrouter.ai/api/v1/chat/completions"

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": PROMPT}
                ],
                "max_tokens": 100,
                "temperature": 0.2,
            }

            response, sent_bytes, received_bytes = make_request(
                url,
                payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )

            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")

        elif provider == "Groq":
            url = "https://api.groq.com/openai/v1/chat/completions"

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": PROMPT}
                ],
                "max_tokens": 100,
                "temperature": 0.2,
            }

            response, sent_bytes, received_bytes = make_request(
                url,
                payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )

            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")

        else:
            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{model}:generateContent"
            )

            payload = {
                "contents": [
                    {
                        "parts": [{"text": PROMPT}]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 100,
                    "temperature": 0.2,
                },
            }

            response, sent_bytes, received_bytes = make_request(
                url,
                payload,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
            )

            data = response.json()
            answer = (
                data["candidates"][0]["content"]["parts"][0]["text"]
            )
            usage = data.get("usageMetadata", {})

            input_tokens = usage.get("promptTokenCount")
            output_tokens = usage.get("candidatesTokenCount")

        elapsed = time.perf_counter() - start

        result.update({
            "response_time_sec": round(elapsed, 4),
            "data_sent_mb": round(
                sent_bytes / (1024 * 1024), 6
            ),
            "data_received_mb": round(
                received_bytes / (1024 * 1024), 6
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_sec": (
                round(output_tokens / elapsed, 3)
                if output_tokens is not None and elapsed > 0
                else None
            ),
            "status": "SUCCESS" if answer else "ERROR",
            "error": "" if answer else "Provider returned an empty answer",
        })

    except requests.exceptions.Timeout:
        result["error"] = (
            f"Request timed out after {REQUEST_TIMEOUT} seconds"
        )

    except requests.exceptions.HTTPError as exc:
        response = exc.response
        if response is not None:
            # Keep diagnostic information, but never return API keys.
            result["error"] = (
                f"HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )
        else:
            result["error"] = "HTTP request failed"

    except (KeyError, IndexError, TypeError, ValueError):
        result["error"] = (
            "Could not parse the provider response. Check the model ID "
            "and provider response format."
        )

    except Exception:
        app.logger.exception("Provider benchmark failed: %s", provider)
        result["error"] = (
            "Unexpected benchmark error. Check the Render logs."
        )

    finally:
        result["response_time_sec"] = round(
            time.perf_counter() - start, 4
        )
        result.update(get_host_metrics())

    return result


def save_cloud_results(results):
    if not results:
        return

    columns = list(results[0].keys())

    with csv_lock:
        file_is_new = (
            not os.path.exists(CLOUD_CSV)
            or os.path.getsize(CLOUD_CSV) == 0
        )

        with open(
            CLOUD_CSV,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=columns,
                extrasaction="ignore",
            )

            if file_is_new:
                writer.writeheader()

            writer.writerows(results)


def read_csv_records(path):
    if not os.path.exists(path):
        return None

    frame = pd.read_csv(path)
    frame = frame.astype(object).where(pd.notnull(frame), None)

    return frame.to_dict(orient="records")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/data")
def get_local_data():
    try:
        records = read_csv_records(LOCAL_CSV)

        if records is None:
            return jsonify({
                "error": "Local benchmark CSV was not found.",
                "expected_path": LOCAL_CSV,
            }), 404

        return jsonify(records)

    except Exception:
        app.logger.exception("Could not read local benchmark data")
        return jsonify({"error": "Could not read local benchmark data"}), 500


@app.route("/api/summary")
def get_local_summary():
    try:
        if not os.path.exists(LOCAL_CSV):
            return jsonify([])

        frame = pd.read_csv(LOCAL_CSV)

        if "status" in frame.columns:
            frame = frame[
                frame["status"].astype(str).str.upper() == "SUCCESS"
            ]

        if frame.empty:
            return jsonify([])

        numeric_columns = [
            "response_time_sec",
            "cpu_avg",
            "cpu_peak",
            "ram_avg",
            "ram_peak",
            "data_sent_mb",
            "data_received_mb",
            "input_tokens",
            "output_tokens",
            "tokens_per_sec",
        ]

        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                )

        required = {"provider", "model", "response_time_sec"}
        if not required.issubset(frame.columns):
            return jsonify({
                "error": "Local CSV is missing required columns.",
                "required_columns": sorted(required),
            }), 500

        aggregations = {
            "avg_response_time": ("response_time_sec", "mean"),
            "samples": ("provider", "count"),
        }

        optional_aggregations = {
            "avg_cpu": ("cpu_avg", "mean"),
            "peak_cpu": ("cpu_peak", "max"),
            "avg_ram": ("ram_avg", "mean"),
            "peak_ram": ("ram_peak", "max"),
            "avg_sent": ("data_sent_mb", "mean"),
            "avg_received": ("data_received_mb", "mean"),
            "avg_input_tokens": ("input_tokens", "mean"),
            "avg_output_tokens": ("output_tokens", "mean"),
            "avg_tokens_per_sec": ("tokens_per_sec", "mean"),
        }

        for output_name, definition in optional_aggregations.items():
            if definition[0] in frame.columns:
                aggregations[output_name] = definition

        summary = (
            frame.groupby(["provider", "model"], dropna=False)
            .agg(**aggregations)
            .reset_index()
        )

        summary = summary.astype(object).where(
            pd.notnull(summary), None
        )

        return jsonify(summary.to_dict(orient="records"))

    except Exception:
        app.logger.exception("Could not summarize local benchmark data")
        return jsonify({"error": "Could not summarize benchmark data"}), 500


@app.route("/api/debug")
def debug():
    result = {
        "local_csv_exists": os.path.exists(LOCAL_CSV),
        "local_csv_path": LOCAL_CSV,
        "cloud_csv_exists": os.path.exists(CLOUD_CSV),
        "cloud_csv_path": CLOUD_CSV,
        "provider_keys_configured": {
            provider: bool(os.environ.get(config["key_env"]))
            for provider, config in PROVIDERS.items()
        },
        "benchmark_token_configured": bool(
            os.environ.get("CLOUD_BENCHMARK_TOKEN")
        ),
    }

    if os.path.exists(LOCAL_CSV):
        try:
            frame = pd.read_csv(LOCAL_CSV)
            result["local_rows"] = len(frame)
            result["local_columns"] = frame.columns.tolist()
        except Exception:
            result["local_csv_readable"] = False

    return jsonify(result)


@app.route("/api/cloud-benchmark", methods=["POST"])
def cloud_benchmark():
    global last_benchmark_time

    configured_token = os.environ.get("CLOUD_BENCHMARK_TOKEN", "")
    supplied_token = request.headers.get("X-Benchmark-Token", "")

    if not configured_token:
        return jsonify({
            "error": "CLOUD_BENCHMARK_TOKEN is not configured in Render."
        }), 503

    if not supplied_token or not hmac.compare_digest(
        supplied_token, configured_token
    ):
        return jsonify({"error": "Invalid benchmark token."}), 401

    if not benchmark_lock.acquire(blocking=False):
        return jsonify({
            "error": "A benchmark is already running. Try again shortly."
        }), 409

    try:
        now = time.time()
        remaining = COOLDOWN_SECONDS - (now - last_benchmark_time)

        if remaining > 0:
            return jsonify({
                "error": (
                    f"Please wait {int(remaining) + 1} seconds "
                    "before starting another benchmark."
                )
            }), 429

        last_benchmark_time = now

        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_provider, provider): provider
                for provider in PROVIDERS
            }

            for future in as_completed(futures):
                provider = futures[future]

                try:
                    results.append(future.result())
                except Exception:
                    app.logger.exception(
                        "Benchmark worker failed: %s", provider
                    )
                    failed = empty_result(provider)
                    failed["error"] = "Benchmark worker failed."
                    results.append(failed)

        order = {provider: i for i, provider in enumerate(PROVIDERS)}
        results.sort(
            key=lambda row: order.get(row["provider"], 99)
        )

        try:
            save_cloud_results(results)
        except Exception:
            # The benchmark can still return results even if the
            # temporary filesystem cannot save the CSV.
            app.logger.exception("Could not save cloud results")

        return jsonify({
            "environment": "Render",
            "prompt_id": "binary_search",
            "successful": sum(
                row["status"] == "SUCCESS" for row in results
            ),
            "failed": sum(
                row["status"] != "SUCCESS" for row in results
            ),
            "results": results,
        })

    finally:
        benchmark_lock.release()


@app.route("/api/cloud-results")
def cloud_results():
    try:
        records = read_csv_records(CLOUD_CSV)
        return jsonify(records or [])

    except Exception:
        app.logger.exception("Could not read cloud benchmark results")
        return jsonify({"error": "Could not read cloud results"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
