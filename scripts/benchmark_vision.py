"""Measure OpenAI CLIP on this machine; results are explicitly local hardware only."""
import argparse, csv, json, os, platform, sys, time, statistics, datetime
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vision.clip_engine import _load, ALL_LABELS, preprocess_image, prepare_cpu_inputs

def run(image_path: Path, passes: int = 3, warmup: int = 1):
    try:
        from PIL import Image
        import torch
    except ImportError as exc: raise SystemExit(f"Install the optional vision dependencies first: {exc}")
    try:
        import psutil
        process=psutil.Process(os.getpid()); before=process.memory_info().rss
    except ImportError: process=None; before=None
    os.environ.setdefault("HF_HOME", str(ROOT/"models"/"huggingface"))
    start=time.perf_counter(); torch,processor,model=_load("openai/clip-vit-base-patch32",True); load=time.perf_counter()-start
    image=Image.open(image_path).convert("RGB")
    start=time.perf_counter(); _,inputs=preprocess_image(image,processor); inputs=prepare_cpu_inputs(inputs); preprocess=time.perf_counter()-start
    with torch.inference_mode():
        for _ in range(warmup): model(**inputs)
    runs=[]
    for _ in range(passes):
        start=time.perf_counter()
        with torch.inference_mode(): model(**inputs)
        runs.append(time.perf_counter()-start)
    after=process.memory_info().rss if process else None
    result={"benchmark_type":"LOCAL INTEL CPU","label":"LOCAL INTEL CPU BENCHMARK","execution_device":"CPU","gpu":"NOT USED","torch_cuda_version":torch.version.cuda,"processor":platform.processor(),"platform":platform.platform(),"model":"openai/clip-vit-base-patch32","image":str(image_path),"model_load_time_seconds":round(load,4),"preprocessing_time_seconds":round(preprocess,4),"warmup_passes":warmup,"measured_passes":passes,"first_inference_time_seconds":round(runs[0],4),"warm_inference_time_seconds":[round(x,4) for x in runs[1:]],"average_inference_time_seconds":round(statistics.mean(runs),4),"minimum_inference_time_seconds":round(min(runs),4),"maximum_inference_time_seconds":round(max(runs),4),"measured_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"process_memory_delta_bytes":after-before if after is not None else None,"memory_note":"RSS delta from process start through inference; includes model weights" if after is not None else "Install psutil for process RSS measurement"}
    return result

def save_result(result, output: Path):
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    csv_path=output.with_suffix(".csv")
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=result);writer.writeheader();writer.writerow(result)
    return csv_path

def main():
    parser=argparse.ArgumentParser();parser.add_argument("image",type=Path);parser.add_argument("--output",type=Path,default=ROOT/"artifacts"/"local_clip_benchmark.json");parser.add_argument("--passes",type=int,default=3);parser.add_argument("--warmup",type=int,default=1);args=parser.parse_args()
    try: result=run(args.image,args.passes,args.warmup)
    except Exception as exc: raise SystemExit(f"Vision benchmark failed: {exc}")
    csv_path=save_result(result,args.output)
    print(json.dumps(result,indent=2)); print(f"Saved {args.output} and {csv_path}")
if __name__=="__main__": main()
