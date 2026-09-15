import psutil
import time
import csv
import os
from datetime import datetime

# Create data folder
os.makedirs("data", exist_ok=True)

file_path = "data/measurements.csv"

# Ask which experiment is being performed
condition = input(
    "Enter condition (baseline / ai_idle / ai_active / ai_workload): "
)

# Record starting network usage
network_start = psutil.net_io_counters()

# Create CSV if it doesn't exist
if not os.path.exists(file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "condition",
            "cpu_usage",
            "ram_usage",
            "disk_usage",
            "data_sent_mb",
            "data_received_mb"
        ])

print("\nMonitoring started.")
print("Press Ctrl+C to stop.\n")

try:

    while True:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # System metrics
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent

        # Network metrics
        network = psutil.net_io_counters()

        bytes_sent = (
            network.bytes_sent -
            network_start.bytes_sent
        )

        bytes_received = (
            network.bytes_recv -
            network_start.bytes_recv
        )

        sent_mb = bytes_sent / (1024 * 1024)
        received_mb = bytes_received / (1024 * 1024)

        # Display
        print(
            f"{timestamp} | "
            f"CPU: {cpu:5.1f}% | "
            f"RAM: {ram:5.1f}% | "
            f"Sent: {sent_mb:6.2f} MB | "
            f"Received: {received_mb:6.2f} MB"
        )

        # Save to CSV
        with open(file_path, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                condition,
                cpu,
                ram,
                disk,
                sent_mb,
                received_mb
            ])

        time.sleep(5)

except KeyboardInterrupt:

    print("\nMonitoring stopped.")
    print(f"Data saved to: {file_path}")
