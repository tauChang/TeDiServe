import numpy as np

def generate_poisson_trace():
    SEED = 42
    np.random.seed(SEED)

    arrivals = []

    def gen_segment(start_t, duration_min, rps):
        duration_sec = duration_min * 60
        end_t = start_t + duration_sec
        t = start_t

        while True:
            inter = np.random.exponential(1.0 / rps)
            t += inter
            if t >= end_t:
                break
            arrivals.append(t)

    # Segment 1: 0–8 min, 4 RPS
    gen_segment(start_t=0.0, duration_min=8, rps=4)

    # Segment 2: 8–16 min, 12 RPS
    gen_segment(start_t=8*60, duration_min=8, rps=12)

    # Segment 3: 16–20 min, 20 RPS
    gen_segment(start_t=16*60, duration_min=4, rps=20)
    
    # total
    print(f"Total arrivals generated: {len(arrivals)}")

    return arrivals


# -----------------------------------------------------
# Generate and save to arrival_trace.txt
# -----------------------------------------------------
arrivals = generate_poisson_trace()

np.savetxt("arrival_trace_4_12_20.txt", arrivals, fmt="%.18e")

print(f"Generated {len(arrivals)} arrivals.")
print("Saved to arrival_trace.txt")
