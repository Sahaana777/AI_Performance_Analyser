import os
import csv
import time
import threading
import requests
import psutil
from datetime import datetime


CSV_FILE = "data/ai_measurements.csv"

PROMPTS = {
    "P1": "Explain binary search in simple terms.",
    "P2": "Write a short Python program to find the largest number in a list.",
    "P3": "Explain the difference between TCP and UDP."
}


MODELS = [
    {
        "provider": "OpenAI",
        "model": os.getenv("OPENAI_MODEL", "")
    },
    {
        "provider": "Anthropic",
        "model": os.getenv("ANTHROPIC_MODEL", "")
    },
    {
        "provider": "Google",
        "model": os.getenv("GEMINI_MODEL", "")
    }
]


CSV_COLUMNS = [
    "timestamp",
    "provider",
    "model",
    "prompt_id",
    "response_time_sec",
    "cpu_avg",
    "cpu_peak",
    "ram_avg",
    "ram_peak",
    "disk_usage",
    "data_sent_mb",
    "data_received_mb",
    "input_tokens",
    "output_tokens",
    "tokens_per_sec",
    "status",
    "error"
]


def get_network_usage():
    net = psutil.net_io_counters()

    return (
        net.bytes_sent,
        net.bytes_recv
    )


def monitor_resources(stop_event, samples):
    """
    Collect CPU and RAM samples while an API request is running.
    """

    while not stop_event.is_set():

        cpu = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory().percent

        samples.append({
            "cpu": cpu,
            "ram": ram
        })


def calculate_resource_stats(samples):

    if not samples:
        return 0, 0, 0, 0

    cpu_values = [sample["cpu"] for sample in samples]
    ram_values = [sample["ram"] for sample in samples]

    return (
        round(sum(cpu_values) / len(cpu_values), 2),
        round(max(cpu_values), 2),
        round(sum(ram_values) / len(ram_values), 2),
        round(max(ram_values), 2)
    )


def save_result(result):

    os.makedirs("data", exist_ok=True)

    file_exists = os.path.exists(CSV_FILE)

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


# ---------------------------------------------------------
# OPENAI
# ---------------------------------------------------------

def call_openai(model, prompt):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    url = "https://api.openai.com/v1/responses"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": prompt
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    usage = data.get("usage", {})

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return input_tokens, output_tokens


# ---------------------------------------------------------
# ANTHROPIC
# ---------------------------------------------------------

def call_anthropic(model, prompt):

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": model,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    usage = data.get("usage", {})

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return input_tokens, output_tokens


# ---------------------------------------------------------
# GOOGLE GEMINI
# ---------------------------------------------------------

def call_google(model, prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    usage = data.get("usageMetadata", {})

    input_tokens = usage.get(
        "promptTokenCount",
        0
    )

    output_tokens = usage.get(
        "candidatesTokenCount",
        0
    )

    return input_tokens, output_tokens


# ---------------------------------------------------------
# PROVIDER DISPATCHER
# ---------------------------------------------------------

def call_provider(provider, model, prompt):

    if provider == "OpenAI":
        return call_openai(model, prompt)

    if provider == "Anthropic":
        return call_anthropic(model, prompt)

    if provider == "Google":
        return call_google(model, prompt)

    raise ValueError(
        f"Unknown provider: {provider}"
    )


# ---------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------

def run_test(provider, model, prompt_id, prompt):

    print("\n" + "=" * 70)
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Prompt   : {prompt_id}")
    print("=" * 70)

    if not model:
        print("Model is not configured. Skipping.")

        return

    time.sleep(2)

    disk_usage = psutil.disk_usage("/").percent

    sent_before, received_before = get_network_usage()

    samples = []

    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(stop_event, samples)
    )

    start_time = time.perf_counter()

    monitor_thread.start()

    status = "SUCCESS"
    error_message = ""

    input_tokens = 0
    output_tokens = 0

    try:

        input_tokens, output_tokens = call_provider(
            provider,
            model,
            prompt
        )

    except Exception as error:

        status = "ERROR"
        error_message = str(error)

        print("ERROR:", error)

    end_time = time.perf_counter()

    stop_event.set()
    monitor_thread.join()

    sent_after, received_after = get_network_usage()

    response_time = end_time - start_time

    sent_mb = (
        sent_after - sent_before
    ) / (1024 * 1024)

    received_mb = (
        received_after - received_before
    ) / (1024 * 1024)

    cpu_avg, cpu_peak, ram_avg, ram_peak = (
        calculate_resource_stats(samples)
    )

    if response_time > 0:
        tokens_per_sec = (
            output_tokens / response_time
        )
    else:
        tokens_per_sec = 0

    result = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "prompt_id": prompt_id,
        "response_time_sec": round(response_time, 3),
        "cpu_avg": cpu_avg,
        "cpu_peak": cpu_peak,
        "ram_avg": ram_avg,
        "ram_peak": ram_peak,
        "disk_usage": round(disk_usage, 2),
        "data_sent_mb": round(sent_mb, 4),
        "data_received_mb": round(received_mb, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tokens_per_sec, 3),
        "status": status,
        "error": error_message
    }

    save_result(result)

    print("\nResult")
    print("-" * 40)
    print("Response time :", round(response_time, 3), "seconds")
    print("CPU average   :", cpu_avg, "%")
    print("CPU peak      :", cpu_peak, "%")
    print("RAM average   :", ram_avg, "%")
    print("RAM peak      :", ram_peak, "%")
    print("Network sent  :", round(sent_mb, 4), "MB")
    print("Network recv  :", round(received_mb, 4), "MB")
    print("Input tokens  :", input_tokens)
    print("Output tokens :", output_tokens)
    print("Tokens/sec    :", round(tokens_per_sec, 3))
    print("Status        :", status)


def main():

    print("\n")
    print("=" * 70)
    print("CLOUD AI MODEL PERFORMANCE BENCHMARK")
    print("=" * 70)

    for model_info in MODELS:

        provider = model_info["provider"]
        model = model_info["model"]

        for prompt_id, prompt in PROMPTS.items():

            run_test(
                provider,
                model,
                prompt_id,
                prompt
            )

            time.sleep(5)

    print("\n")
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {CSV_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
