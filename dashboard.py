import os
from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

CSV_FILE = "data/ai_measurements.csv"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data")
def get_data():

    if not os.path.exists(CSV_FILE):
        return jsonify([])

    df = pd.read_csv(CSV_FILE)

    return jsonify(
        df.to_dict(orient="records")
    )


@app.route("/api/summary")
def get_summary():

    if not os.path.exists(CSV_FILE):
        return jsonify([])

    df = pd.read_csv(CSV_FILE)

    df = df[
        df["status"] == "SUCCESS"
    ]

    if df.empty:
        return jsonify([])

    summary = (
        df.groupby(
            ["provider", "model"]
        )
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
            avg_sent=(
                "data_sent_mb",
                "mean"
            ),
            avg_received=(
                "data_received_mb",
                "mean"
            ),
            avg_input_tokens=(
                "input_tokens",
                "mean"
            ),
            avg_output_tokens=(
                "output_tokens",
                "mean"
            ),
            avg_tokens_per_sec=(
                "tokens_per_sec",
                "mean"
            ),
            samples=(
                "provider",
                "count"
            )
        )
        .reset_index()
    )

    return jsonify(
        summary.to_dict(
            orient="records"
        )
    )


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )