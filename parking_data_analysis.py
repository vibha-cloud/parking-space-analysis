import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def analyze_parking_data():
    log_file = "parking_space_log.csv"

    try:
        # Load the log file
        data = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: {log_file} not found.")
        return

    if "Timestamp" not in data.columns or "AvailableSpaces" not in data.columns:
        print("Error: Required columns ('Timestamp', 'AvailableSpaces') not found.")
        return

    # Convert Timestamp to datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    # Extract date and hour for analysis
    data["Date"] = data["Timestamp"].dt.date
    data["Hour"] = data["Timestamp"].dt.hour

    # Aggregate data for daily trends
    daily_avg = data.groupby("Date")["AvailableSpaces"].mean()

    # Aggregate data for hourly trends
    hourly_avg = data.groupby("Hour")["AvailableSpaces"].mean()

    # Plot daily average availability
    plt.figure(figsize=(10, 6))
    daily_avg.plot(kind="line", marker="o")
    plt.title("Daily Average Parking Availability")
    plt.xlabel("Date")
    plt.ylabel("Average Available Spaces")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("daily_avg_availability.png")
    plt.show()

    # Plot hourly trends
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=hourly_avg.index,
        y=hourly_avg.values,
        palette="viridis",
        hue=hourly_avg.index,
        dodge=False,
        legend=False,
    )
    plt.title("Average Parking Availability by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Available Spaces")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hourly_avg_availability.png")
    plt.show()

    print(
        "Analysis complete. Check 'daily_avg_availability.png' and 'hourly_avg_availability.png' for visualizations."
    )


if __name__ == "__main__":
    print("Script execution started...")
    analyze_parking_data()
