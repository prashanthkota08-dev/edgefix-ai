"""Export the CLIP image/text embedding path to ONNX (CPU export only)."""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vision.clip_engine import _load, ALL_LABELS, MODEL_ID

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=Path(__file__).parent/"artifacts"/"edgefix_clip_embeddings.onnx");args=parser.parse_args()
    try:
        import torch
        from PIL import Image
    except ImportError as exc: raise SystemExit(f"Install vision requirements before export: {exc}")
    _,processor,model=_load(MODEL_ID,False)
    inputs=processor(text=[f"A photo of {label} in an industrial maintenance setting." for label in ALL_LABELS],images=Image.new("RGB",(224,224)),return_tensors="pt",padding="max_length",max_length=77,truncation=True)
    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, wrapped): super().__init__(); self.wrapped=wrapped
        def forward(self,pixel_values,input_ids,attention_mask):
            output=self.wrapped(pixel_values=pixel_values,input_ids=input_ids,attention_mask=attention_mask)
            return output.image_embeds,output.text_embeds
    wrapper=EmbeddingWrapper(model).eval();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(wrapper,(inputs["pixel_values"],inputs["input_ids"],inputs["attention_mask"]),str(args.output),input_names=["pixel_values","input_ids","attention_mask"],output_names=["image_embeds","text_embeds"],dynamic_axes={"input_ids":{0:"candidate_count"},"attention_mask":{0:"candidate_count"},"text_embeds":{0:"candidate_count"}},opset_version=17)
    metadata={"model":MODEL_ID,"export_format":"ONNX","file":str(args.output),"opset":17,"execution":"CPU export; target execution not verified","candidate_labels":ALL_LABELS}
    meta=args.output.with_suffix(".metadata.json");meta.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Exported ONNX model: {args.output}\nTarget execution remains unverified.")
if __name__=="__main__": main()
