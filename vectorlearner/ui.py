from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch

from .checkpoint import load_vectorizer_checkpoint, newest_checkpoint
from .cuda_renderer import CudaSampledShapeRenderer
from .data import load_rgb, pil_to_tensor, resize_max_side
from .fit import refine_initialized
from .model import ShapeVectorizer
from .model_v2 import ShapeVectorizerV2
from .renderer import SoftShapeRenderer
from .svg import save_svg


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "div2k-model-quick-v2" / "checkpoint_0000400.pt"
V2_CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "v2-optimized"
BUNDLED_CHECKPOINT = PROJECT_ROOT / "models" / "shape_vectorizer_v2.pt"
DEFAULT_JOBS_DIR = PROJECT_ROOT / "runs" / "ui"
MAX_UPLOAD_BYTES = 75 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def preferred_default_checkpoint() -> Path:
    fallback = BUNDLED_CHECKPOINT if BUNDLED_CHECKPOINT.exists() else DEFAULT_CHECKPOINT
    return newest_checkpoint(V2_CHECKPOINT_DIR, fallback)


class FastVectorizer:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.available = checkpoint_path.exists()
        self.model: ShapeVectorizer | ShapeVectorizerV2 | None = None
        self.model_version: str | None = None
        self.renderer: SoftShapeRenderer | CudaSampledShapeRenderer | None = None
        self.correction_steps = 0
        self.trained_shapes = 0
        if self.available:
            loaded = load_vectorizer_checkpoint(checkpoint_path, device)
            if not isinstance(loaded.model, (ShapeVectorizer, ShapeVectorizerV2)):
                raise TypeError("Unsupported vectorizer model")
            self.model = loaded.model
            self.model_version = loaded.model_version
            self.trained_shapes = loaded.trained_shapes
            if isinstance(self.model, ShapeVectorizerV2):
                self.correction_steps = self.model.correction_steps
                if device.type == "cuda":
                    self.renderer = CudaSampledShapeRenderer(
                        min_size=self.model.min_size,
                        max_size=self.model.max_size,
                        allow_fallback=True,
                    ).to(device)
                else:
                    self.renderer = SoftShapeRenderer(
                        min_size=self.model.min_size,
                        max_size=self.model.max_size,
                        chunk_size=16,
                    ).to(device)

    def predict_parameters(
        self,
        source: Path,
        shape_count: int,
    ) -> tuple[object, object, torch.Tensor, torch.Tensor, float]:
        if self.model is None:
            raise RuntimeError("The fast checkpoint is missing; use Refined fit or supply --checkpoint")
        if not 1 <= shape_count <= self.trained_shapes:
            raise ValueError(f"Fast mode supports 1..{self.trained_shapes} trained shapes")

        original = load_rgb(source)
        encoder_side = 512 if isinstance(self.model, ShapeVectorizerV2) else 1024
        encoder_image = resize_max_side(original, encoder_side)
        image_tensor = pil_to_tensor(encoder_image)[None].to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            if isinstance(self.model, ShapeVectorizerV2):
                assert self.renderer is not None
                raw_shapes, raw_background = self.model(
                    image_tensor,
                    shape_count,
                    renderer=self.renderer,
                    correction_steps=self.correction_steps,
                )
            else:
                raw_shapes, raw_background = self.model(image_tensor, shape_count)
        elapsed = time.perf_counter() - started
        return original, encoder_image, raw_shapes, raw_background, elapsed

    def vectorize(self, source: Path, destination: Path, shape_count: int) -> dict[str, object]:
        original, encoder_image, raw_shapes, raw_background, elapsed = self.predict_parameters(
            source,
            shape_count,
        )
        use_presence = self.model_version == "v2"
        assert self.model is not None
        save_svg(
            destination,
            raw_shapes,
            raw_background,
            original.width,
            original.height,
            min_size=self.model.min_size,
            max_size=self.model.max_size,
            use_presence=use_presence,
            presence_threshold=0.25 if use_presence else 0.0,
            title=source.name,
        )
        return {
            "mode": "fast",
            "model_version": self.model_version,
            "checkpoint": str(self.checkpoint_path),
            "correction_steps": self.correction_steps,
            "shapes": shape_count,
            "native_size": [original.width, original.height],
            "encoder_size": list(encoder_image.size),
            "elapsed_seconds": elapsed,
        }

    def save_initializer(self, source: Path, destination: Path, shape_count: int) -> float:
        _, _, raw_shapes, raw_background, elapsed = self.predict_parameters(source, shape_count)
        torch.save(
            {
                "raw_shapes": raw_shapes.detach().cpu(),
                "raw_background": raw_background.detach().cpu(),
                "checkpoint": str(self.checkpoint_path),
            },
            destination,
        )
        return elapsed


def refined_settings(shape_count: int, quality: str) -> tuple[list[int], int, int, int]:
    stages = [value for value in (16, 32, 64, 128, 256, 450) if value <= shape_count]
    if not stages or stages[-1] != shape_count:
        stages.append(shape_count)
    presets = {
        "quick": (15, 8_192, 8),
        "balanced": (60, 16_384, 8),
        "detailed": (120, 16_384, 4),
    }
    if quality not in presets:
        raise ValueError("Unknown refined quality preset")
    steps, samples, error_refresh = presets[quality]
    return stages, steps, samples, error_refresh


def run_refined(
    source: Path,
    output_dir: Path,
    shape_count: int,
    quality: str,
    initializer_checkpoint: Path | None = None,
    initializer_parameters: Path | None = None,
) -> dict[str, object]:
    if shape_count not in (64, 128, 256, 450):
        raise ValueError("Refined mode supports 64, 128, 256, or 450 shapes")
    stages, steps, samples, error_refresh = refined_settings(shape_count, quality)
    if initializer_checkpoint is not None or initializer_parameters is not None:
        stages = [shape_count]
    command = [
        sys.executable,
        "-m",
        "vectorlearner.fit",
        str(source),
        "--output-dir",
        str(output_dir),
        "--max-shapes",
        str(shape_count),
        "--stages",
        ",".join(str(value) for value in stages),
        "--steps-per-stage",
        str(steps),
        "--sample-count",
        str(samples),
        "--error-refresh",
        str(error_refresh),
        "--preview-side",
        "768",
        "--log-every",
        str(steps),
    ]
    if initializer_checkpoint is not None:
        command.extend(("--initializer-checkpoint", str(initializer_checkpoint)))
    if initializer_parameters is not None:
        command.extend(("--initializer-parameters", str(initializer_parameters)))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Refined fit failed: {detail[-1500:]}")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["mode"] = "refined"
    summary["quality"] = quality
    return summary


def parse_form(headers: object, body: bytes) -> dict[str, str | tuple[str, bytes]]:
    content_type = headers.get("Content-Type", "")  # type: ignore[attr-defined]
    message = BytesParser(policy=email_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    values: dict[str, str | tuple[str, bytes]] = {}
    if not message.is_multipart():
        raise ValueError("Expected multipart form data")
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            values[name] = (filename, payload)
        else:
            values[name] = payload.decode(part.get_content_charset() or "utf-8")
    return values


class VectorUIHandler(BaseHTTPRequestHandler):
    server_version = "VectorShapeUI/0.1"

    @property
    def app(self) -> "VectorUIServer":
        return self.server  # type: ignore[return-value]

    def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._file(Path(__file__).with_name("ui.html"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._json(
                {
                    "device": str(self.app.device),
                    "gpu": torch.cuda.get_device_name(0) if self.app.device.type == "cuda" else None,
                    "fast_available": self.app.vectorizer.available,
                    "fast_trained_shapes": self.app.vectorizer.trained_shapes,
                    "fast_model_version": self.app.vectorizer.model_version,
                    "fast_correction_steps": self.app.vectorizer.correction_steps,
                }
            )
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if path.startswith("/jobs/"):
            relative = Path(path.removeprefix("/jobs/"))
            candidate = (self.app.jobs_dir / relative).resolve()
            jobs_root = self.app.jobs_dir.resolve()
            if jobs_root not in candidate.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._file(candidate)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/vectorize":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError("Image upload must be between 1 byte and 75 MB")
            form = parse_form(self.headers, self.rfile.read(length))
            upload = form.get("image")
            if not isinstance(upload, tuple):
                raise ValueError("Choose an image first")
            filename, data = upload
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise ValueError("Supported images: JPG, PNG, WebP, BMP, and TIFF")
            mode = str(form.get("mode", "fast"))
            shapes = int(str(form.get("shapes", "64")))
            quality = str(form.get("quality", "quick"))

            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            job_dir = self.app.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            source = job_dir / f"input{suffix}"
            source.write_bytes(data)
            image = load_rgb(source)
            if image.width * image.height > 100_000_000:
                raise ValueError("Image is over the 100-megapixel safety limit")

            with self.app.inference_lock:
                if mode == "fast":
                    metadata = self.app.vectorizer.vectorize(source, job_dir / "result.svg", shapes)
                elif mode == "refined":
                    initializer_seconds = 0.0
                    if (
                        self.app.vectorizer.available
                        and self.app.vectorizer.trained_shapes >= shapes
                    ):
                        _, _, initial_shapes, initial_background, initializer_seconds = (
                            self.app.vectorizer.predict_parameters(
                                source,
                                shapes,
                            )
                        )
                        _, steps, samples, error_refresh = refined_settings(shapes, quality)
                        assert self.app.vectorizer.model is not None
                        metadata = refine_initialized(
                            source,
                            job_dir,
                            initial_shapes,
                            initial_background,
                            steps=steps,
                            sample_count=samples,
                            error_refresh=error_refresh,
                            min_size=self.app.vectorizer.model.min_size,
                            max_size=self.app.vectorizer.model.max_size,
                        )
                    else:
                        metadata = run_refined(source, job_dir, shapes, quality)
                    metadata["mode"] = "refined"
                    metadata["quality"] = quality
                    metadata["initializer_inference_seconds"] = initializer_seconds
                    metadata["elapsed_seconds"] = (
                        float(metadata.get("elapsed_seconds", 0.0)) + initializer_seconds
                    )
                else:
                    raise ValueError("Unknown vectorization mode")
            (job_dir / "ui-result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self._json(
                {
                    "ok": True,
                    "job_id": job_id,
                    "svg_url": f"/jobs/{job_id}/result.svg",
                    "download_name": f"{Path(filename).stem}-vectorized.svg",
                    "metadata": metadata,
                }
            )
        except (ValueError, RuntimeError) as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._json({"ok": False, "error": f"Unexpected error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")


class VectorUIServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def __init__(
        self,
        address: tuple[str, int],
        jobs_dir: Path,
        vectorizer: FastVectorizer,
        device: torch.device,
    ) -> None:
        super().__init__(address, VectorUIHandler)
        self.jobs_dir = jobs_dir
        self.vectorizer = vectorizer
        self.device = device
        self.inference_lock = threading.Lock()

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local drag-and-drop image-to-SVG UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint override; defaults to the newest local V2 checkpoint or the bundled model",
    )
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see the GPU")
    device = torch.device(args.device)
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint if args.checkpoint is not None else preferred_default_checkpoint()
    vectorizer = FastVectorizer(checkpoint_path.resolve(), device)
    server = VectorUIServer((args.host, args.port), args.jobs_dir.resolve(), vectorizer, device)
    url = f"http://{args.host}:{args.port}"
    if args.open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    print(f"Vector Shape UI: {url}")
    if vectorizer.available:
        print(
            f"Fast checkpoint: {vectorizer.checkpoint_path} "
            f"({vectorizer.model_version}, {vectorizer.trained_shapes} shapes)"
        )
    print("Close this window or press Ctrl+C to stop it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
