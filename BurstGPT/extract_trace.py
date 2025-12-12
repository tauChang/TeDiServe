import json

def load_arrivals(path):
    """Load arrival times from a text file, one per line."""
    arrivals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                arrivals.append(float(line))
    return arrivals


def extract_window(arrivals, start_min, end_min):
    """
    Extract arrival times between start_min and end_min (minutes).
    Normalize so first timestamp = 0.
    """
    start_s = start_min * 60
    end_s = end_min * 60

    segment = [t for t in arrivals if start_s <= t <= end_s]
    if not segment:
        return []

    base = segment[0]
    return [t - base for t in segment]


def main():
    # ------------------------------------------------------
    # EDIT THESE VALUES DIRECTLY
    # ------------------------------------------------------
    IN_FILE = "burstgpt_8hrs_35qps.txt"
    OUT_FILE = "burstgpt_8hrs_35qps_170_to_190.txt"
    START_MIN = 170
    END_MIN = 190
    # ------------------------------------------------------

    arrivals = load_arrivals(IN_FILE)
    trimmed = extract_window(arrivals, START_MIN, END_MIN)

    with open(OUT_FILE, "w") as f:
        for t in trimmed:
            f.write(f"{t:.6f}\n")

    print(f"Extracted {len(trimmed)} arrivals from {START_MIN} to {END_MIN} min.")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
