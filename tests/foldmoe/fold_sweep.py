"""FoldMoE 1A1M：纯 prefill 下扫 micro-batch 数（max_tokens=1 隔离 decode）"""
import json, os, statistics, sys, time
# 输出目录，用 FOLDMOE_OUT 覆盖
OUT = os.environ.get("FOLDMOE_OUT", ".")
SEQ = int(os.environ.get("FOLD_SEQ", "65536"))
D = sys.argv[1]
TAG = os.environ.get("FOLD_TAG", "fold_" + D)
os.environ["FOLDMOE_1A1M"] = D
PROF = OUT + "/prof_" + TAG
from vllm import LLM, SamplingParams
# 超过模型原生 163840 时，把 YaRN 的 factor 调大，让 RoPE cache 覆盖到 SEQ+1
# （4096 是 original_max_position_embeddings）。不这么做位置索引会越界，
# 报 EE9999 Failed to submit record task。
EXTRA = {}
if SEQ + 1 > 163840:
    f = (SEQ + 4096) // 4096
    EXTRA["hf_overrides"] = {
        "max_position_embeddings": 4096 * f,
        "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": float(f),
                          "mscale": 0.707, "mscale_all_dim": 0.707,
                          "original_max_position_embeddings": 4096,
                          "type": "yarn", "rope_type": "yarn"},
    }
    print("[fold_sweep] RoPE 覆盖: factor=%d -> 位置上限 %d" % (f, 4096 * f), flush=True)
llm = LLM(model="/docker/models/DeepSeek-V2-Lite", tensor_parallel_size=2, **EXTRA,
          max_model_len=SEQ + 256, max_num_batched_tokens=SEQ,
          gpu_memory_utilization=0.75,
          trust_remote_code=True, enforce_eager=True,
          enable_expert_parallel=True, enable_prefix_caching=False, seed=0,
          profiler_config={"profiler": "torch", "torch_profiler_dir": PROF})
tok = llm.get_tokenizer()
ids = tok("混合专家模型通过路由机制分配token。", add_special_tokens=False).input_ids
while len(ids) < SEQ: ids = ids + ids
ids = ids[:SEQ]
req = [{"prompt_token_ids": ids}]
sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
llm.generate([{"prompt_token_ids": ids[:128]}], sp)
lat = []
for _ in range(3):
    t0 = time.perf_counter(); outs = llm.generate(req, sp); lat.append(time.perf_counter() - t0)
if os.environ.get("FOLD_PROF", "1") == "1":
    llm.start_profile(); llm.generate(req, sp); llm.stop_profile(); time.sleep(5)
r = {"d": D, "seq": SEQ, "med": round(statistics.median(lat), 3),
     "lat": [round(x, 3) for x in lat], "ids": list(outs[0].outputs[0].token_ids)}
json.dump(r, open(OUT + "/res_%s.json" % TAG, "w"))
print("[%s] med=%.3fs lat=%s seq=%d chunk=%d" % (TAG, r["med"], r["lat"], SEQ, SEQ // max(1,int(D))))
print("===== FOLD_OK =====")
