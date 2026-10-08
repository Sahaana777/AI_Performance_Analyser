import os
import csv
import time
import threading
import requests
import psutil

from datetime import datetime
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# FILE SETTINGS
# ============================================================

CSV_FILE = "data/ai_measurements.csv"


# ============================================================
# TEST PROMPTS
# ============================================================

PROMPTS = {
    "P1": "Explain binary search in simple terms.",

    "P2": "Write a short Python program to find the largest number in a list.",

    "P3": "Explain the difference between TCP and UDP."
}


# ============================================================
# MODELS
# ============================================================

MODELS = [

    {
        "provider": "Openrouter AI",
        "model": os.getenv("OPENROUTER_MODEL", "")
    },

    {
        "provider": "Google Gemini",
        "model": os.getenv("GEMINI_MODEL", "")
    },

    {
        "provider": "Groq",
        "model": os.getenv("GROQ_MODEL", "")
    }

]


# ============================================================
# CSV COLUMNS
# ============================================================

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


# ============================================================
# NETWORK MONITORING
# ============================================================

def get_network_usage():

    network = psutil.net_io_counters()

    return (
        network.bytes_sent,
        network.bytes_recv
    )


# ============================================================
# CPU / RAM MONITORING
# ============================================================

def monitor_resources(stop_event, samples):

    while not stop_event.is_set():

        cpu = psutil.cpu_percent(
            interval=0.2
        )

        ram = psutil.virtual_memory().percent

        samples.append(
            {
                "cpu": cpu,
                "ram": ram
            }
        )


# ============================================================
# CALCULATE CPU / RAM STATISTICS
# ============================================================

def calculate_resource_stats(samples):

    if not samples:

        return (
            0,
            0,
            0,
            0
        )

    cpu_values = [
        sample["cpu"]
        for sample in samples
    ]

    ram_values = [
        sample["ram"]
        for sample in samples
    ]

    cpu_average = (
        sum(cpu_values)
        / len(cpu_values)
    )

    cpu_peak = max(cpu_values)

    ram_average = (
        sum(ram_values)
        / len(ram_values)
    )

    ram_peak = max(ram_values)

    return (

        round(cpu_average, 2),

        round(cpu_peak, 2),

        round(ram_average, 2),

        round(ram_peak, 2)

    )


# ============================================================
# SAVE RESULT TO CSV
# ============================================================

def save_result(result):

    os.makedirs(
        "data",
        exist_ok=True
    )

    file_exists = os.path.exists(
        CSV_FILE
    )

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


# ============================================================
# Openrouter AI
# ============================================================

def call_openrouter(model, prompt):

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "stream": False
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(30, 180)
    )

    response.raise_for_status()

    data = response.json()

    usage = data.get("usage", {})

    input_tokens = usage.get(
        "prompt_tokens",
        0
    )

    output_tokens = usage.get(
        "completion_tokens",
        0
    )

    return (
        input_tokens,
        output_tokens
    )

# ============================================================
# GOOGLE GEMINI
# ============================================================


def call_gemini(model, prompt):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY is not set."
        )

    url = (

        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"

    )

    headers = {

        "x-goog-api-key":
            api_key,

        "Content-Type":
            "application/json"

    }

    payload = {

        "contents": [

            {

                "parts": [

                    {

                        "text":
                            prompt

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

    usage = data.get(
        "usageMetadata",
        {}
    )

    input_tokens = usage.get(
        "promptTokenCount",
        0
    )

    output_tokens = usage.get(
        "candidatesTokenCount",
        0
    )

    return (
        input_tokens,
        output_tokens
    )


# ============================================================
# GROQ
# ============================================================

def call_groq(model, prompt):

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    url = (
        "https://api.groq.com/"
        "openai/v1/chat/completions"
    )

    headers = {

        "Authorization":
            f"Bearer {api_key}",

        "Content-Type":
            "application/json"

    }

    payload = {

        "model": model,

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

    usage = data.get(
        "usage",
        {}
    )

    input_tokens = usage.get(
        "prompt_tokens",
        0
    )

    output_tokens = usage.get(
        "completion_tokens",
        0
    )

    return (
        input_tokens,
        output_tokens
    )


# ============================================================
# SELECT PROVIDER
# ============================================================

def call_provider(
    provider,
    model,
    prompt
):

    if provider == "Openrouter AI":

        return call_openrouter(
            model,
            prompt
        )

    elif provider == "Google Gemini":

        return call_gemini(
            model,
            prompt
        )

    elif provider == "Groq":

        return call_groq(
            model,
            prompt
        )

    else:

        raise ValueError(
            f"Unknown provider: {provider}"
        )


# ============================================================
# RUN ONE BENCHMARK TEST
# ============================================================

def run_test(
    provider,
    model,
    prompt_id,
    prompt
):

    print()
    print("=" * 70)

    print(
        f"Provider : {provider}"
    )

    print(
        f"Model    : {model}"
    )

    print(
        f"Prompt   : {prompt_id}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not model:

        print(
            "Model is not configured."
        )

        print(
            "Skipping this test."
        )

        return

    # --------------------------------------------------------
    # WAIT BEFORE REQUEST
    # --------------------------------------------------------

    time.sleep(2)

    # --------------------------------------------------------
    # INITIAL RESOURCE VALUES
    # --------------------------------------------------------

    disk_usage = psutil.disk_usage(
        "/"
    ).percent

    sent_before, received_before = (
        get_network_usage()
    )

    # --------------------------------------------------------
    # START RESOURCE MONITOR
    # --------------------------------------------------------

    samples = []

    stop_event = threading.Event()

    monitor_thread = threading.Thread(

        target=monitor_resources,

        args=(
            stop_event,
            samples
        )

    )

    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()

    monitor_thread.start()

    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    status = "SUCCESS"

    error_message = ""

    input_tokens = 0

    output_tokens = 0

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    try:

        (
            input_tokens,
            output_tokens
        ) = call_provider(

            provider,

            model,

            prompt

        )

    except Exception as error:

        status = "ERROR"

        error_message = str(error)

        print()

        print(
            "ERROR:",
            error_message
        )

    # --------------------------------------------------------
    # STOP TIMER
    # --------------------------------------------------------

    end_time = time.perf_counter()

    # --------------------------------------------------------
    # STOP RESOURCE MONITOR
    # --------------------------------------------------------

    stop_event.set()

    monitor_thread.join()

    # --------------------------------------------------------
    # FINAL NETWORK VALUES
    # --------------------------------------------------------

    sent_after, received_after = (
        get_network_usage()
    )

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    response_time = (
        end_time
        - start_time
    )

    data_sent_mb = (

        sent_after
        - sent_before

    ) / (1024 * 1024)

    data_received_mb = (

        received_after
        - received_before

    ) / (1024 * 1024)

    (
        cpu_avg,
        cpu_peak,
        ram_avg,
        ram_peak

    ) = calculate_resource_stats(
        samples
    )

    if response_time > 0:

        tokens_per_sec = (

            output_tokens
            / response_time

        )

    else:

        tokens_per_sec = 0

    # --------------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------------

    result = {

        "timestamp":
            datetime.now().isoformat(),

        "provider":
            provider,

        "model":
            model,

        "prompt_id":
            prompt_id,

        "response_time_sec":
            round(
                response_time,
                3
            ),

        "cpu_avg":
            cpu_avg,

        "cpu_peak":
            cpu_peak,

        "ram_avg":
            ram_avg,

        "ram_peak":
            ram_peak,

        "disk_usage":
            round(
                disk_usage,
                2
            ),

        "data_sent_mb":
            round(
                data_sent_mb,
                4
            ),

        "data_received_mb":
            round(
                data_received_mb,
                4
            ),

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "tokens_per_sec":
            round(
                tokens_per_sec,
                3
            ),

        "status":
            status,

        "error":
            error_message

    }

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    save_result(
        result
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print()
    print("Result")
    print("-" * 40)

    print(
        "Response time :",
        round(
            response_time,
            3
        ),
        "seconds"
    )

    print(
        "CPU average   :",
        cpu_avg,
        "%"
    )

    print(
        "CPU peak      :",
        cpu_peak,
        "%"
    )

    print(
        "RAM average   :",
        ram_avg,
        "%"
    )

    print(
        "RAM peak      :",
        ram_peak,
        "%"
    )

    print(
        "Network sent  :",
        round(
            data_sent_mb,
            4
        ),
        "MB"
    )

    print(
        "Network recv  :",
        round(
            data_received_mb,
            4
        ),
        "MB"
    )

    print(
        "Input tokens  :",
        input_tokens
    )

    print(
        "Output tokens :",
        output_tokens
    )

    print(
        "Tokens/sec    :",
        round(
            tokens_per_sec,
            3
        )
    )

    print(
        "Status        :",
        status
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "CLOUD AI MODEL PERFORMANCE BENCHMARK"
    )

    print("=" * 70)

    print()

    print(
        "Configured providers:"
    )

    for model_info in MODELS:

        print(

            f"  {model_info['provider']}"
            f" -> {model_info['model']}"

        )

    print()

    # --------------------------------------------------------
    # RUN ALL PROVIDERS
    # --------------------------------------------------------

    for model_info in MODELS:

        provider = model_info[
            "provider"
        ]

        model = model_info[
            "model"
        ]

        # Run all prompts

        for prompt_id, prompt in PROMPTS.items():

            run_test(

                provider,

                model,

                prompt_id,

                prompt

            )

            # Delay between API calls

            time.sleep(5)

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "BENCHMARK COMPLETE"
    )

    print(
        f"Results saved to: {CSV_FILE}"
    )

    print("=" * 70)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()
