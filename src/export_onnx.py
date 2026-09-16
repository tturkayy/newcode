import torch
from model import APSKDemodulator1D


def export():
    device = torch.device("cpu")
    model = APSKDemodulator1D(in_channels=1, symbol_len=24, num_classes=16)
    model.load_state_dict(torch.load("apsk_demodulator.pth", map_location=device, weights_only=True))
    model.eval()

    # Model giriş tensörü: (Batch, Channels, Length) -> (96, 1, 24)
    # 96 sembolü tek bir batch olarak aynı anda WebGPU'ya besleyeceğiz.
    dummy_input = torch.randn(96, 1, 24, dtype=torch.float32)

    onnx_path = "apsk_demodulator.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"[OK] ONNX modeli olusturuldu: {onnx_path}")


if __name__ == "__main__":
    export()