"""Download once, load locally, and execute one actual CPU CLIP pass."""
from pathlib import Path
import sys, time, json, datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vision.clip_engine import ClipVisionEngine, _load, MODEL_ID, ALL_LABELS

def main():
    import os
    os.environ.setdefault("HF_HOME",str(ROOT/"models"/"huggingface"))
    image=ROOT/"data"/"sample_images"/"sample_motor.png"
    if not image.is_file():
        from scripts.create_sample_motor_image import main as create_image
        create_image()
    start=time.perf_counter(); _load(MODEL_ID,False); model_load=time.perf_counter()-start
    start=time.perf_counter(); result=ClipVisionEngine(local_files_only=True).analyze(image.read_bytes()); total=time.perf_counter()-start
    if result.status!="Visual similarity signal":
        print(result.summary,file=sys.stderr); return 1
    timings=result.details["timings"]
    record={"model_name":MODEL_ID,"device":"CPU","gpu":"NOT USED","image_path":str(image.relative_to(ROOT)),"labels":ALL_LABELS,"top_results":result.details["observations"],"model_load_time_seconds":round(model_load,4),"preprocessing_time_seconds":timings["preprocessing_seconds"],"inference_time_seconds":timings["inference_seconds"],"total_analyze_call_seconds":round(total,4),"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"synthetic_image_notice":"Illustration verifies model execution only; it is not real-world accuracy evidence."}
    output=ROOT/"artifacts"/"local_clip_smoke_test.json";output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(f"CLIP model and processor loaded; actual CPU image/text inference passed. Model load={model_load:.2f}s, inference={timings['inference_seconds']:.2f}s.")
    print("Top CLIP similarity concepts:")
    for item in result.details["observations"]: print(f"  {item['label']}: {item['similarity']:.4f}")
    print(f"Saved actual results to {output}")
    print("DEVICE=CPU; GPU=NOT USED")
    return 0

if __name__=="__main__": raise SystemExit(main())
