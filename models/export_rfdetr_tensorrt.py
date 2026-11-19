import argparse
import subprocess
from pathlib import Path

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

MODEL_MAP: dict[str, type]= {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}

def export(size: str, precision: str):
    output_path = Path(__file__).parent.resolve() / "output"
    onnx_path = output_path / f"rfdetr-{size}.onnx"
    engine_path = output_path / f"rfdetr-{size}-{precision}.engine"

    if not onnx_path.is_file():
        model = MODEL_MAP[size]()
        model.export(output_dir=str(output_path), output_name= f"rfdetr-{size}")

    command = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]

    if precision == "fp16":
        command.append("--fp16")
    
    subprocess.run(command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the pretrained RF-DETR TensorRT engine of the desired size and precision.")
    parser.add_argument("-s", "--size", type=str, default="large", choices=list(MODEL_MAP), help="The size of the RF-DETR model to export")
    parser.add_argument("-p", "--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="The precision of the exported model. INT8 currently unsupported.")

    args = parser.parse_args()
    export(args.size, args.precision)
