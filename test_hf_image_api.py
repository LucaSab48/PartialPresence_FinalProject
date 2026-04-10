import argparse
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import InferenceClient

from hf_auth import load_hf_token


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Hugging Face text-to-image generation with your HF_TOKEN."
    )
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Hugging Face model id to test",
    )
    parser.add_argument(
        "--prompt",
        default="an installation room with a projector",
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Requested output width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Requested output height",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_images",
        help="Folder where the test image will be saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = load_hf_token()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = InferenceClient(provider="hf-inference", api_key=token)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"hf_test_{timestamp}.png"

    print(f"Testing Hugging Face image generation with model: {args.model}")
    print(f"Prompt: {args.prompt}")

    try:
        image = client.text_to_image(
            args.prompt,
            model=args.model,
            width=args.width,
            height=args.height,
        )
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    image.save(output_path)
    print(f"Success. Image saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
