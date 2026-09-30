"""Submit and record a real Qualcomm AI Hub profile job; never simulates success."""
from pathlib import Path
import argparse, json, os, sys, datetime

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("model",type=Path,nargs="?",default=Path(__file__).parent/"artifacts"/"edgefix_clip_embeddings.onnx");parser.add_argument("--device",help="Exact available device name from qai-hub list-devices");args=parser.parse_args()
    token=os.environ.get("QAI_HUB_API_TOKEN")
    if not token:
        print("Qualcomm AI Hub validation is optional and has not been configured.",file=sys.stderr);return 2
    if not args.device:
        parser.error("--device is required when QAI_HUB_API_TOKEN is configured")
    if not args.model.is_file():
        print(f"ONNX model not found: {args.model}. Run clip_export.py first.",file=sys.stderr);return 2
    try:
        import qai_hub as hub
    except ImportError:
        print("Install qai-hub in the profiling environment before submitting.",file=sys.stderr);return 2
    try:
        client=hub.Client(hub.ClientConfig(api_token=token))
        device=hub.Device(args.device)
        job=client.submit_profile_job(str(args.model),device=device,name="EdgeFix CLIP visual similarity")
        job.wait()
        profile=job.download_profile()
        def attr(*names):
            for name in names:
                value=getattr(job,name,None)
                if value is not None: return str(value)
            return None
        status=str(getattr(job,"status","unknown"))
        if "fail" in status.lower() or "error" in status.lower():
            raise RuntimeError(f"AI Hub returned job status {status}; no validated profile saved")
        record={"status":"AI Hub profile completed","validation":"REAL AI HUB PROFILE JOB","job_id":attr("job_id","id"),"job_status":status,"job_url":attr("url","web_url"),"device":args.device,"model":str(args.model.resolve()),"submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"profile":profile}
        output=Path(__file__).parent/"ai_hub_profile_result.json";output.write_text(json.dumps(record,indent=2),encoding="utf-8")
        print(json.dumps(record,indent=2));return 0
    except Exception as exc:
        print(f"AI Hub profile failed; no validation result was saved: {exc}",file=sys.stderr);return 1
if __name__=="__main__": raise SystemExit(main())
