import os
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(
    BASE_DIR,
    "data",
    "ai_measurements.csv"
)

GRAPH_DIR = os.path.join(
    BASE_DIR,
    "graphs"
)


# -------------------------------------------------
# Load benchmark data
# -------------------------------------------------

def load_data():

    if not os.path.exists(CSV_FILE):

        print("No benchmark data found.")
        print("Expected file:")
        print(CSV_FILE)

        return pd.DataFrame()

    print("Loading:")
    print(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    print("\nCSV columns:")
    print(df.columns.tolist())

    # -------------------------------------------------
    # Check required columns
    # -------------------------------------------------

    required_columns = [
        "provider",
        "model",
        "response_time_sec",
        "cpu_avg",
        "cpu_peak",
        "ram_avg",
        "ram_peak",
        "data_received_mb",
        "tokens_per_sec"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print("\nERROR: Missing columns:")
        print(missing)

        print("\nYour CSV contains:")
        print(df.columns.tolist())

        return pd.DataFrame()

    # -------------------------------------------------
    # Convert numeric columns
    # -------------------------------------------------

    numeric_columns = [
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
        "tokens_per_sec"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # -------------------------------------------------
    # Filter successful experiments
    # -------------------------------------------------

    if "status" in df.columns:

        df = df[
            df["status"].astype(str).str.upper() == "SUCCESS"
        ]

    else:

        print("\nWARNING:")
        print("No 'status' column found.")
        print("Assuming all rows are valid benchmark records.")

    # Remove rows where provider is missing

    df = df.dropna(
        subset=["provider"]
    )

    return df


# -------------------------------------------------
# Generate graphs
# -------------------------------------------------

def generate_graphs(df):

    os.makedirs(
        GRAPH_DIR,
        exist_ok=True
    )

    if df.empty:

        print("\nNo successful benchmark data.")
        return

    # -------------------------------------------------
    # Provider summary
    # -------------------------------------------------

    summary = (
        df.groupby("provider")
        .agg(
            avg_response_time=(
                "response_time_sec",
                "mean"
            ),

            avg_cpu=(
                "cpu_avg",
                "mean"
            ),

            peak_cpu=(
                "cpu_peak",
                "mean"
            ),

            avg_ram=(
                "ram_avg",
                "mean"
            ),

            peak_ram=(
                "ram_peak",
                "mean"
            ),

            avg_network_received=(
                "data_received_mb",
                "mean"
            ),

            avg_tokens_per_sec=(
                "tokens_per_sec",
                "mean"
            )
        )
        .reset_index()
    )

    # -------------------------------------------------
    # Response time
    # -------------------------------------------------

    plt.figure()

    plt.bar(
        summary["provider"],
        summary["avg_response_time"]
    )

    plt.title(
        "Average API Response Time by Provider"
    )

    plt.xlabel("Provider")
    plt.ylabel("Response Time (seconds)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "response_time.png"
        )
    )

    plt.close()

    # -------------------------------------------------
    # CPU
    # -------------------------------------------------

    plt.figure()

    plt.bar(
        summary["provider"],
        summary["avg_cpu"]
    )

    plt.title(
        "Average CPU Utilization by Provider"
    )

    plt.xlabel("Provider")
    plt.ylabel("CPU Usage (%)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "cpu_usage.png"
        )
    )

    plt.close()

    # -------------------------------------------------
    # RAM
    # -------------------------------------------------

    plt.figure()

    plt.bar(
        summary["provider"],
        summary["avg_ram"]
    )

    plt.title(
        "Average RAM Utilization by Provider"
    )

    plt.xlabel("Provider")
    plt.ylabel("RAM Usage (%)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "ram_usage.png"
        )
    )

    plt.close()

    # -------------------------------------------------
    # Network
    # -------------------------------------------------

    plt.figure()

    plt.bar(
        summary["provider"],
        summary["avg_network_received"]
    )

    plt.title(
        "Average Network Data Received"
    )

    plt.xlabel("Provider")
    plt.ylabel("Data Received (MB)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "network_usage.png"
        )
    )

    plt.close()

    # -------------------------------------------------
    # Token throughput
    # -------------------------------------------------

    plt.figure()

    plt.bar(
        summary["provider"],
        summary["avg_tokens_per_sec"]
    )

    plt.title(
        "Average Token Throughput"
    )

    plt.xlabel("Provider")
    plt.ylabel("Output Tokens / Second")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "token_throughput.png"
        )
    )

    plt.close()

    # -------------------------------------------------
    # Save summary
    # -------------------------------------------------

    summary.to_csv(
        os.path.join(
            GRAPH_DIR,
            "provider_summary.csv"
        ),
        index=False
    )

    print("\nProvider Summary")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nGraphs saved in:")
    print(GRAPH_DIR)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    df = load_data()

    if df.empty:
        return

    print("\nBenchmark records:")
    print(len(df))

    print("\nProviders:")

    print(
        df["provider"]
        .unique()
        .tolist()
    )

    generate_graphs(df)


if __name__ == "__main__":
    main()
