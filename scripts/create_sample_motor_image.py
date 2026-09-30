"""Create a deterministic local illustration for inference-path checks only."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]

def main():
    output=ROOT/"data"/"sample_images"/"sample_motor.png"
    output.parent.mkdir(parents=True,exist_ok=True)
    image=Image.new("RGB",(960,640),(20,31,40)); draw=ImageDraw.Draw(image)
    # Base and motor body.
    draw.rounded_rectangle((190,235,690,470),radius=32,fill=(85,121,132),outline=(183,209,211),width=7)
    draw.rectangle((250,210,600,235),fill=(102,143,151),outline=(205,225,222),width=4)
    # Cooling fins and end cap.
    for x in range(285,580,35): draw.rounded_rectangle((x,255,x+12,445),radius=4,fill=(42,70,79),outline=(134,171,174),width=2)
    draw.ellipse((610,270,745,435),fill=(61,91,101),outline=(209,224,220),width=7)
    draw.ellipse((643,303,714,402),fill=(23,39,47),outline=(139,173,175),width=4)
    # Shaft, feet, terminal box, base.
    draw.rectangle((730,328,835,365),fill=(177,185,174),outline=(235,237,221),width=4)
    draw.rounded_rectangle((365,160,510,232),radius=10,fill=(43,68,77),outline=(215,224,218),width=4)
    draw.line((392,183,480,183),fill=(123,220,174),width=4)
    draw.rectangle((260,465,620,505),fill=(104,116,112),outline=(211,217,201),width=4)
    draw.rectangle((225,500,655,527),fill=(76,88,87))
    for x in (280,605): draw.ellipse((x-12,482,x+12,506),fill=(26,39,43),outline=(218,218,195),width=3)
    try: font=ImageFont.truetype("arial.ttf",24)
    except OSError: font=ImageFont.load_default()
    draw.text((36,34),"EDGEFIX · SYNTHETIC MOTOR ILLUSTRATION",font=font,fill=(104,226,173))
    draw.text((36,575),"Inference smoke test only · not a real motor photo or accuracy evidence",font=font,fill=(206,216,216))
    image.save(output,optimize=True)
    print(output)

if __name__=="__main__": main()
