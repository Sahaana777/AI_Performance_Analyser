import os

from flask import Flask, render_template, jsonify
import pandas as pd


app = Flask(__name__)


# -------------------------------------------------
# Absolute project paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSV_FILE = os.path.join(
    BASE_DIR,
    "data",
    "ai_measurements.csv"
)


# -------------------------------------------------
# Home
# -------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# -------------------------------------------------
# Raw benchmark data
# -------------------------------------------------

@app.route("/api/data")
def get_data():

    if not os.path.exists(CSV_FILE):

        return jsonify({
            "error": "CSV file not found",
            "path": CSV_FILE
        }), 404

    try:

        df = pd.read_csv(
            CSV_FILE
        )

        # Convert NaN to JSON-safe values

        df = df.where(
            pd.notnull(df),
            None
        )

        return jsonify(
            df.to_dict(
                orient="records"
            )
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -------------------------------------------------
# Provider summary
# -------------------------------------------------

@app.route("/api/summary")
def get_summary():

    if not os.path.exists(CSV_FILE):

        return jsonify({
            "error": "CSV file not found",
            "path": CSV_FILE
        }), 404

    try:

        df = pd.read_csv(
            CSV_FILE
        )

        # -------------------------------------------------
        # Check status column
        # -------------------------------------------------

        if "status" in df.columns:

            df = df[
                df["status"]
                .astype(str)
                .str.upper()
                == "SUCCESS"
            ]

        # -------------------------------------------------
        # No valid records
        # -------------------------------------------------

        if df.empty:

            return jsonify([])

        # -------------------------------------------------
        # Provider summary
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Convert NaN values
        # -------------------------------------------------

        summary = summary.where(
            pd.notnull(summary),
            None
        )

        return jsonify(
            summary.to_dict(
                orient="records"
            )
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -------------------------------------------------
# Debug endpoint
# -------------------------------------------------

@app.route("/api/debug")
def debug():

    result = {
        "csv_path": CSV_FILE,
        "csv_exists": os.path.exists(CSV_FILE),
        "project_directory": BASE_DIR,
        "current_directory": os.getcwd()
    }

    if os.path.exists(CSV_FILE):

        try:

            df = pd.read_csv(
                CSV_FILE
            )

            result["columns"] = (
                df.columns.tolist()
            )

            result["rows"] = len(df)

            if "status" in df.columns:

                result["successful_rows"] = int(
                    (
                        df["status"]
                        .astype(str)
                        .str.upper()
                        == "SUCCESS"
                    ).sum()
                )

        except Exception as e:

            result["csv_error"] = str(e)

    return jsonify(result)


# -------------------------------------------------
# Run Flask
# -------------------------------------------------

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
