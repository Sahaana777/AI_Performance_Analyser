from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

CSV_FILE = "data/measurements.csv"

COLUMNS = [
    "timestamp",
    "condition",
    "cpu_usage",
    "ram_usage",
    "disk_usage",
    "data_sent_mb",
    "data_received_mb"
]


def load_data():

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=COLUMNS)

    # First try reading normally
    df = pd.read_csv(CSV_FILE)

    # Check whether the expected columns exist
    if not set(COLUMNS).issubset(df.columns):

        # CSV probably has no header
        df = pd.read_csv(
            CSV_FILE,
            header=None,
            names=COLUMNS
        )

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Convert numeric columns
    numeric_columns = [
        "cpu_usage",
        "ram_usage",
        "disk_usage",
        "data_sent_mb",
        "data_received_mb"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where the important values are invalid
    df = df.dropna(
        subset=[
            "condition",
            "cpu_usage",
            "ram_usage"
        ]
    )

    return df


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/api/data")
def get_data():

    try:

        df = load_data()

        return jsonify(
            df.to_dict(orient="records")
        )

    except Exception as e:

        print("DATA ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/summary")
def get_summary():

    try:

        df = load_data()

        if df.empty:

            return jsonify([])

        summary = (
            df.groupby("condition")
            .agg(
                avg_cpu=("cpu_usage", "mean"),
                avg_ram=("ram_usage", "mean"),
                avg_disk=("disk_usage", "mean"),
                total_sent=("data_sent_mb", "max"),
                total_received=("data_received_mb", "max"),
                samples=("cpu_usage", "count")
            )
            .reset_index()
        )

        return jsonify(
            summary.to_dict(orient="records")
        )

    except Exception as e:

        print("SUMMARY ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    print("=" * 50)
    print("AI PERFORMANCE ANALYZER")
    print("=" * 50)

    print(
        f"Reading data from: {CSV_FILE}"
    )

    try:

        df = load_data()

        print(
            f"Measurements loaded: {len(df)}"
        )

        if not df.empty:

            print(
                "Conditions found:",
                df["condition"].unique().tolist()
            )

    except Exception as e:

        print(
            "Could not load measurements:",
            e
        )

    print("=" * 50)

    app.run(debug=True)
