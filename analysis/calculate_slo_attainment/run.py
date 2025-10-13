import json
path = "/u/tchang85/dllm/eval/results/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9_normal_inter3_SLO5.json"
path = "/u/tchang85/dllm/eval/results/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9_inter3_SLO5.json"
path = "/u/tchang85/dllm/eval/results/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9_inter3_SLO5_1.json"
# path = "/u/tchang85/dllm/eval/results/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9_inter3_SLO5_4_4.json"
path = "/u/tchang85/dllm/eval/results/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"
path = "/u/tchang85/dllm/eval/results_1013/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"
# path = "/u/tchang85/dllm/eval/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32_conf0.9.json"
response_times = {}
# read in json
with open(path, "r") as f:
    data = json.load(f)
    
response_times = data["response_times"]
    
count = sum(1 for v in response_times.values() if v <= 5)
len_total = len(response_times)
print(count, "≤5 out of", len_total)
print("avg response time:", sum(response_times.values()) / len_total)
# plot histogram
import matplotlib.pyplot as plt
plt.hist(response_times.values(), bins=50)
plt.xlabel("Response Time (s)")
plt.ylabel("Frequency")
plt.title("Response Time Distribution")
plt.axvline(x=5, color='r', linestyle='--', label='SLO (5s)')
# x min 0, max 20
plt.xlim(0, 20)
plt.legend()
plt.savefig("response_time_histogram.png")
