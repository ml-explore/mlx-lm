# Copyright © 2026 Apple Inc.

"""Image inputs for `mlx_lm.server` chat completions.

The OpenAI chat API sends images as `{"type": "image_url", "image_url":
{"url": "data:image/png;base64,..."}}` content parts. Before this module the
server rejected any non-text part outright ("Only 'text' content type is
supported"), so any client that attached an image got a failed request.

Flow for a request carrying images (vision-capable models only):

1. `extract_images` (HTTP thread) pulls the raw image bytes out of the
   messages, in document order, and rewrites each image part to the
   `{"type": "image"}` form chat templates expect.
2. The chat template renders one image placeholder token per image.
3. `ImageInputs.build` (generation thread) preprocesses the images, expands
   each placeholder into the model's real soft-token span, and runs the
   model's own multimodal fusion once to get the full-prompt input
   embeddings. The server then hands those embeddings to the normal
   `stream_generate` path (it already supports `input_embeddings`, chunked in
   lock-step with the tokens), so chunked prefill can't split an image away
   from its placeholders.

Prompt caching: the server's prompt cache is keyed by token ids, and two
different images produce identical placeholder ids. `build` therefore also
returns a cache key where each image's positions hold a pseudo id derived
from a hash of that image's bytes -- the same image in a later turn reuses
the cache, a different image only matches the text before it.

Only Gemma 4 is wired up so far (its image preprocessing is HF's
`Gemma4ImageProcessorPil`, which needs Pillow but not torch).

Limits (requests are untrusted input): images come as `data:` URIs; http(s)
URLs are fetched only when the server runs with `--allow-image-urls` (else a
client could make the server request any address it can reach, loopback and
LAN included), then with a size cap and an overall deadline. Every image is
checked from its header -- format, pixel count -- in the HTTP thread, before
anything is decoded, and a request carries at most MAX_IMAGES images.
"""

import base64
import hashlib
import http.client
import io
import json
import math
import socket
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import numpy as np

# Pseudo token ids for image positions in prompt-cache keys only. Far above
# any real vocabulary so they can never collide with a real token.
_IMAGE_KEY_BASE = 1 << 40

# Set by the server from --allow-image-urls.
ALLOW_IMAGE_URLS = False
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 20 * 1024 * 1024
# 4000 x 4000 per image, 48 MP per request. Checked from the header: a tiny
# PNG can declare a huge canvas (a 20 KB file decoding to gigabytes). The
# processor resizes to at most max_soft_tokens patches anyway, so nothing
# larger is ever needed; decoded, 48 MP is about 1 GB at the peak.
MAX_IMAGE_PIXELS = 16_000_000
MAX_REQUEST_PIXELS = 48_000_000
URL_FETCH_SECONDS = 30
# Formats whose Image.open reads only the header (ICO, for one, decodes its
# embedded PNG right there -- the size check would come after the cost).
IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP"}
# Progressive JPEG decode time grows with its number of scans, not its size
# or pixels: 50 000 tiny scans in 1.6 MB took 30 s of the generation thread.
# Real encoders write about 10.
MAX_JPEG_SCANS = 100


class _HTTPOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith(("http://", "https://")):
            raise ValueError(f"Image URL redirected to an unsupported scheme: {newurl[:32]}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _RegisteringContext:
    """An SSL context whose sockets are registered before their handshake:
    wrapping moves the connection's descriptor into a new SSLSocket (the
    raw socket's shutdown then fails), and the handshake itself reads."""

    def __init__(self, context, register):
        self._context, self._register = context, register

    def wrap_socket(self, sock, **kwargs):
        wrapped = self._context.wrap_socket(sock, do_handshake_on_connect=False, **kwargs)
        self._register(wrapped)
        try:
            wrapped.do_handshake()
        except BaseException:
            wrapped.close()  # as the handshake inside wrap_socket would
            raise
        return wrapped

    def __getattr__(self, name):
        return getattr(self._context, name)


def _fetch(url: str) -> bytes:
    """An http(s) image, at most MAX_IMAGE_BYTES, within URL_FETCH_SECONDS
    overall. A deadline checked between reads can't stop a read already
    blocked (a header or chunk line dripped a byte at a time keeps each
    one inside the socket timeout): at the deadline the connections'
    sockets are shut down, which wakes it."""
    sockets, expired = [], threading.Event()
    deadline = time.monotonic() + URL_FETCH_SECONDS

    def shut(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def register(sock):
        sockets.append(sock)
        if expired.is_set():
            shut(sock)
        return sock

    def connect(address, timeout=None, source_address=None, **_):
        """socket.create_connection, each attempt within what's left of
        the deadline (each address could take the whole socket timeout),
        registered as soon as it exists: a proxy's CONNECT response is
        read inside connect(), before TLS."""
        # Resolved aside: getaddrinfo can't be interrupted, only waited for
        # up to the deadline (the resolver's own timeouts end it).
        resolved = {}

        def resolve():
            try:
                resolved["infos"] = socket.getaddrinfo(*address, 0, socket.SOCK_STREAM)
            except OSError as e:
                resolved["error"] = e

        resolver = threading.Thread(target=resolve, daemon=True)
        resolver.start()
        resolver.join(max(0.0, deadline - time.monotonic()))
        if "error" in resolved:
            raise resolved["error"]
        if "infos" not in resolved:
            raise TimeoutError(f"Resolving {address[0]} took too long.")
        error = None
        for family, kind, proto, _, addr in resolved["infos"]:
            left = deadline - time.monotonic()
            if left <= 0 or expired.is_set():
                break
            sock = None
            try:
                sock = register(socket.socket(family, kind, proto))
                sock.settimeout(min(timeout, left) if isinstance(timeout, (int, float)) else left)
                if source_address:
                    sock.bind(source_address)
                sock.connect(addr)
                sock.settimeout(timeout if isinstance(timeout, (int, float)) else None)
                return sock
            except OSError as e:
                error = e
                if sock is not None:
                    sock.close()
        raise error or TimeoutError(f"Connecting to {address[0]} took too long.")

    def tracked(base):
        class Connection(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._create_connection = connect
                if getattr(self, "_context", None) is not None:
                    self._context = _RegisteringContext(self._context, register)

        return Connection

    class HTTP(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(tracked(http.client.HTTPConnection), req)

    class HTTPS(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(tracked(http.client.HTTPSConnection), req, context=self._context)

    def expire():
        expired.set()
        for sock in list(sockets):
            shut(sock)

    timer = threading.Timer(URL_FETCH_SECONDS, expire)
    timer.daemon = True
    timer.start()
    request = urllib.request.Request(url, headers={"User-Agent": "mlx-lm-server"})
    opener = urllib.request.build_opener(_HTTPOnlyRedirects, HTTP, HTTPS)
    try:
        with opener.open(request, timeout=10) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None and length.isdigit() and int(length) > MAX_IMAGE_BYTES:
                raise ValueError(f"Image at {url[:64]} is larger than {MAX_IMAGE_BYTES} bytes.")
            chunks, total = [], 0
            while True:
                chunk = resp.read1(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image at {url[:64]} is larger than {MAX_IMAGE_BYTES} bytes.")
                chunks.append(chunk)
    except Exception:
        if expired.is_set() or time.monotonic() >= deadline:
            raise ValueError(f"Fetching {url[:64]} took longer than {URL_FETCH_SECONDS}s.")
        raise
    finally:
        timer.cancel()
    if expired.is_set():
        # Shut down between reads: what arrived may be cut short.
        raise ValueError(f"Fetching {url[:64]} took longer than {URL_FETCH_SECONDS}s.")
    return b"".join(chunks)


def _jpeg_scans(blob: bytes) -> int:
    """An upper bound on a JPEG's scans: every SOS marker (FF DA) counts,
    except those inside the metadata segments (APPn, COM) right after SOI,
    which every decoder skips by their length -- a comment full of FF DA
    rejected a valid image. Mirroring a decoder's resync over junk isn't
    attempted: past the metadata, every FF DA counts."""
    total, i, n = blob.count(b"\xff\xda"), 2, len(blob)
    while i + 4 <= n and blob[i] == 0xFF and (0xE0 <= blob[i + 1] <= 0xEF or blob[i + 1] == 0xFE):
        end = i + 2 + int.from_bytes(blob[i + 2 : i + 4], "big")
        if end > n or end < i + 4:
            break
        total -= blob.count(b"\xff\xda", i + 4, end)
        i = end
    return total


def _image_bytes_from_url(url: str) -> bytes:
    if url.startswith("data:"):
        try:
            _, b64 = url.split(",", 1)
        except ValueError:
            raise ValueError("Malformed image data URI.")
        if len(b64) > MAX_IMAGE_BYTES * 4 // 3 + 4096:
            raise ValueError(f"Image is larger than {MAX_IMAGE_BYTES} bytes.")
        # validate=True: the default silently drops non-alphabet characters,
        # turning garbage into b"" that only fails later, deep in the
        # generation thread.
        blob = base64.b64decode("".join(b64.split()), validate=True)
        if len(blob) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image is larger than {MAX_IMAGE_BYTES} bytes.")
        return blob
    if url.startswith(("http://", "https://")):
        if not ALLOW_IMAGE_URLS:
            raise ValueError(
                "Image URLs aren't fetched by this server: send the image as a "
                "data: URI (base64), or start the server with --allow-image-urls."
            )
        return _fetch(url)
    raise ValueError(f"Unsupported image URL scheme: {url[:32]}...")


def _check_image(blob: bytes) -> int:
    """Refuses what isn't an image in IMAGE_FORMATS, or declares more than
    MAX_IMAGE_PIXELS -- from the header only, nothing decoded. Returns its
    pixel count."""
    from PIL import Image

    import warnings

    try:
        # Pillow warns above its own limit; ours is lower and checked below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(blob), formats=sorted(IMAGE_FORMATS)) as im:
                width, height = im.size
    except Image.DecompressionBombError:
        raise ValueError(f"Image is larger than {MAX_IMAGE_PIXELS} pixels.")
    except Exception:
        raise ValueError(
            f"Unsupported or corrupt image data (accepted: {', '.join(sorted(IMAGE_FORMATS))})."
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is {width}x{height} pixels; at most {MAX_IMAGE_PIXELS} are accepted."
        )
    if blob[:2] == b"\xff\xd8" and _jpeg_scans(blob) > MAX_JPEG_SCANS:
        raise ValueError(f"JPEG has more than {MAX_JPEG_SCANS} scans.")
    return width * height


def extract_images(messages: List[Any]) -> List[bytes]:
    """Collect image bytes from OpenAI-style `image_url` content parts (in
    order) and rewrite those parts in place to `{"type": "image"}`. Raises
    ValueError (the server answers 400) for anything malformed."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list.")
    images = []
    pixels = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be an object.")
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if not all(isinstance(part, dict) and isinstance(part.get("type"), str) for part in content):
            raise ValueError("Each content part must be an object with a string type.")
        types = {part.get("type") for part in content}
        if "image" in types:
            # Only this function makes those: a client's would render an
            # image placeholder with no image behind it.
            raise ValueError("Send images as image_url parts.")
        if "image_url" in types and not types <= {"text", "image_url"}:
            # They'd reach the chat template as raw special tokens.
            raise ValueError("Only text and image_url parts are supported together with images.")
        for i, part in enumerate(content):
            if part.get("type") != "image_url":
                continue
            if len(images) >= MAX_IMAGES:
                raise ValueError(f"At most {MAX_IMAGES} images per request.")
            ref = part.get("image_url")
            url = ref.get("url") if isinstance(ref, dict) else ref
            if not isinstance(url, str):
                raise ValueError("image_url part is missing its url.")
            try:
                blob = _image_bytes_from_url(url)
            except ValueError:
                raise
            except Exception as e:  # network errors, bad base64, ...
                raise ValueError(f"Could not load image_url: {e}") from e
            pixels += _check_image(blob)
            if pixels > MAX_REQUEST_PIXELS:
                raise ValueError(f"The images add up to more than {MAX_REQUEST_PIXELS} pixels.")
            images.append(blob)
            content[i] = {"type": "image"}
    return images


def _decode(blob: bytes):
    """The RGB image, upright -- a phone photo's EXIF orientation applied, as
    HF's load_image does (its size was checked by extract_images)."""
    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(blob), formats=sorted(IMAGE_FORMATS))
    return ImageOps.exif_transpose(image).convert("RGB")


class UnifiedImageProcessor:
    """`Gemma4UnifiedImageProcessor` (gemma4_unified, the 12B) without torch:
    the image resized (aspect kept) to fit `max_soft_tokens` patches of
    (pooling_kernel_size * patch_size)^2 pixels, scaled to [0, 1], cut into
    those patches row by row, each flattened height x width x RGB, and padded
    to `max_soft_tokens` (positions (x, y), (-1, -1) for padding). HF's
    16 px patches merged 3x3 come out in exactly this order and layout."""

    def __init__(self, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3, rescale_factor=1 / 255, **_):
        self.side = patch_size * pooling_kernel_size
        self.max_soft_tokens = max_soft_tokens
        self.rescale = rescale_factor

    def size(self, height: int, width: int) -> Tuple[int, int]:
        """HF's get_aspect_ratio_preserving_size, in merged-patch units."""
        side, n = self.side, self.max_soft_tokens
        factor = math.sqrt(n * side * side / (height * width))
        h, w = int(math.floor(factor * height / side)), int(math.floor(factor * width / side))
        if h == 0 and w == 0:
            raise ValueError(f"image {width}x{height} is too small to patch")
        if h == 0:
            h, w = 1, min(int(math.floor(width / height)), n)
        elif w == 0:
            w, h = 1, min(int(math.floor(height / width)), n)
        return h * side, w * side

    def __call__(self, images, return_tensors="np"):
        from PIL import Image

        pixels, positions, counts = [], [], []
        for image in images:
            h, w = self.size(image.height, image.width)
            if (h, w) != (image.height, image.width):
                image = image.resize((w, h), Image.BICUBIC)
            a = np.asarray(image, dtype=np.float32) * self.rescale
            s = self.side
            gh, gw = h // s, w // s
            patches = a.reshape(gh, s, gw, s, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, s * s * 3)
            pos = np.stack(np.meshgrid(np.arange(gw), np.arange(gh), indexing="xy"), -1).reshape(-1, 2)
            pad = self.max_soft_tokens - len(patches)
            pixels.append(np.pad(patches, ((0, pad), (0, 0))))
            positions.append(np.pad(pos, ((0, pad), (0, 0)), constant_values=-1))
            counts.append(len(patches))
        return {
            "pixel_values": np.stack(pixels),
            "image_position_ids": np.stack(positions),
            "num_soft_tokens_per_image": counts,
        }


class ImageInputs:
    """Gemma 4 image preprocessing + prompt expansion + fusion."""

    def __init__(self, model, processor_config: dict):
        cfg = processor_config.get("image_processor", processor_config)
        kwargs = {
            k: cfg[k]
            for k in ("patch_size", "max_soft_tokens", "pooling_kernel_size")
            if k in cfg
        }
        if getattr(model, "unified", False):
            self.processor = UnifiedImageProcessor(rescale_factor=cfg.get("rescale_factor", 1 / 255), **kwargs)
        else:
            from transformers.models.gemma4.image_processing_pil_gemma4 import (
                Gemma4ImageProcessorPil,
            )

            self.processor = Gemma4ImageProcessorPil(**kwargs)
        self.model = model
        args = model.args
        self.image_token_id = args.image_token_id
        self.boi_token_id = args.boi_token_id
        self.eoi_token_id = args.eoi_token_id

    def build(
        self, prompt: List[int], images: List[bytes]
    ) -> Tuple[List[int], mx.array, List[int]]:
        """Returns (expanded token ids, input embeddings [L, H], cache key)."""
        n_placeholders = sum(1 for t in prompt if t == self.image_token_id)
        if n_placeholders != len(images):
            raise ValueError(
                f"Chat template produced {n_placeholders} image placeholder(s) "
                f"for {len(images)} image(s)."
            )

        pil_images = [_decode(b) for b in images]
        feats = self.processor(images=pil_images, return_tensors="np")
        n_soft = [int(n) for n in feats["num_soft_tokens_per_image"]]
        keys = [
            _IMAGE_KEY_BASE + int.from_bytes(hashlib.sha256(b).digest()[:5], "big")
            for b in images
        ]

        expanded, cache_key = [], []
        img = 0
        for t in prompt:
            if t == self.image_token_id:
                span = [self.image_token_id] * n_soft[img]
                expanded += [self.boi_token_id] + span + [self.eoi_token_id]
                cache_key += (
                    [self.boi_token_id] + [keys[img]] * n_soft[img] + [self.eoi_token_id]
                )
                img += 1
            else:
                expanded.append(t)
                cache_key.append(t)

        fused, _ = self.model._fuse_multimodal_inputs(
            mx.array(expanded)[None],
            mx.array(feats["pixel_values"]),
            mx.array(feats["image_position_ids"]),
            None,
            None,
        )
        embeddings = fused[0]
        mx.eval(embeddings)

        # The ids generation runs on must have image positions replaced by
        # pad, exactly like the fusion does internally: models with per-layer
        # embeddings (Gemma 4 E2B/E4B, hidden_size_per_layer_input > 0) derive
        # them from these ids, and HF computes them from the pad-substituted
        # ids. Passing the raw image-token ids here instead measurably
        # changed the output (last-token logits off by ~26) and the model
        # answered as if no image had been given.
        pad = self.model.language_model.model.config.pad_token_id
        gen_ids = [pad if t == self.image_token_id else t for t in expanded]
        return gen_ids, embeddings, cache_key


class Qwen35ImageInputs:
    """Qwen3.5 image preprocessing (as HF Qwen2VLImageProcessor), prompt
    expansion, fusion and mRoPE positions (as HF get_rope_index)."""

    # Lower than the checkpoint limit (16.7 MP) to keep prefill short.
    MAX_PIXELS = 1_048_576

    def __init__(self, model, processor_config: dict):
        cfg = processor_config.get("image_processor", processor_config)
        self.model = model
        self.patch = int(cfg.get("patch_size", 16))
        self.merge = int(cfg.get("merge_size", 2))
        self.temporal = int(cfg.get("temporal_patch_size", 2))
        size = cfg.get("size", {})
        self.min_pixels = int(size.get("shortest_edge", cfg.get("min_pixels", 65536)))
        self.max_pixels = min(
            int(size.get("longest_edge", cfg.get("max_pixels", self.MAX_PIXELS))),
            self.MAX_PIXELS,
        )
        self.mean = np.array(cfg.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.array(cfg.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.image_token_id = model.args.image_token_id

    def _resize(self, h: int, w: int) -> Tuple[int, int]:
        factor = self.patch * self.merge
        # Same limit as HF smart_resize. Thin images exceed the pixel cap.
        if max(h, w) / max(1, min(h, w)) > 200:
            raise ValueError(
                f"Image aspect ratio {max(h, w) / max(1, min(h, w)):.0f}:1 is over 200:1."
            )
        h_bar = max(factor, round(h / factor) * factor)
        w_bar = max(factor, round(w / factor) * factor)
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((h * w) / self.max_pixels)
            h_bar = max(factor, math.floor(h / beta / factor) * factor)
            w_bar = max(factor, math.floor(w / beta / factor) * factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (h * w))
            h_bar = math.ceil(h * beta / factor) * factor
            w_bar = math.ceil(w * beta / factor) * factor
        while h_bar * w_bar > self.max_pixels and max(h_bar, w_bar) > factor:
            if h_bar >= w_bar:
                h_bar -= factor
            else:
                w_bar -= factor
        return h_bar, w_bar

    def _patches(self, image) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        from PIL import Image

        h, w = self._resize(image.height, image.width)
        image = image.convert("RGB").resize((w, h), Image.BICUBIC)
        x = (
            np.asarray(image, dtype=np.float32) / 255.0 - self.mean
        ) / self.std  # [H, W, C]
        x = np.repeat(x.transpose(2, 0, 1)[None], self.temporal, axis=0)  # [T, C, H, W]
        gt, gh, gw = 1, h // self.patch, w // self.patch
        m, p, t, c = self.merge, self.patch, self.temporal, x.shape[1]
        x = x.reshape(gt, t, c, gh // m, m, p, gw // m, m, p)
        x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gt * gh * gw, c * t * p * p), (gt, gh, gw)

    def build(
        self, prompt: List[int], images: List[bytes]
    ) -> Tuple[List[int], mx.array, List[int]]:
        n_placeholders = sum(1 for t in prompt if t == self.image_token_id)
        if n_placeholders != len(images):
            raise ValueError(
                f"Chat template produced {n_placeholders} image placeholder(s) "
                f"for {len(images)} image(s)."
            )
        patches, grids = [], []
        for blob in images:
            pv, grid = self._patches(_decode(blob))
            patches.append(pv)
            grids.append(grid)
        keys = [
            _IMAGE_KEY_BASE + int.from_bytes(hashlib.sha256(b).digest()[:5], "big")
            for b in images
        ]

        expanded, cache_key, axes = [], [], [[], [], []]
        next_pos, img = 0, 0
        for tok in prompt:
            if tok != self.image_token_id:
                expanded.append(tok)
                cache_key.append(tok)
                for a in axes:
                    a.append(next_pos)
                next_pos += 1
                continue
            _, gh, gw = grids[img]
            lh, lw = gh // self.merge, gw // self.merge
            expanded += [self.image_token_id] * (lh * lw)
            cache_key += [keys[img]] * (lh * lw)
            for i in range(lh):
                for j in range(lw):
                    axes[0].append(next_pos)
                    axes[1].append(next_pos + i)
                    axes[2].append(next_pos + j)
            next_pos += max(lh, lw)
            img += 1

        embeddings = self.model.embed_with_images(
            mx.array(expanded)[None],
            mx.array(np.concatenate(patches)),
            mx.array(grids, dtype=mx.int32),
        )[0]
        mx.eval(embeddings)
        positions = mx.array(axes, dtype=mx.int32)
        self.media_positions = (positions, next_pos - len(expanded))
        return expanded, embeddings, cache_key


class Lfm2VlImageInputs:
    """LFM2-VL image preprocessing (as HF Lfm2VlImageProcessor): a large
    image is cut into tile_size tiles on the grid closest to its aspect ratio,
    plus a thumbnail; a small one is resized whole. Prompt expansion follows
    Lfm2VlProcessor, with the image special tokens."""

    def __init__(self, model, processor_config: dict, token_ids: dict):
        cfg = processor_config.get("image_processor", processor_config)
        self.model = model
        self.patch = int(cfg.get("encoder_patch_size", 16))
        self.factor = int(cfg.get("downsample_factor", 2))
        self.tile = int(cfg.get("tile_size", 512))
        self.min_tiles = int(cfg.get("min_tiles", 2))
        self.max_tiles = int(cfg.get("max_tiles", 10))
        if not cfg.get("do_image_splitting", True):
            self.min_tiles = self.max_tiles = 1
        self.thumbnail = bool(cfg.get("use_thumbnail", True))
        self.min_tokens = int(cfg.get("min_image_tokens", 64))
        self.max_tokens = int(cfg.get("max_image_tokens", 256))
        self.tolerance = float(cfg.get("max_pixels_tolerance", 2.0))
        self.mean = np.array(cfg.get("image_mean", [0.5] * 3), dtype=np.float32)
        self.std = np.array(cfg.get("image_std", [0.5] * 3), dtype=np.float32)
        self.image_token_id = model.args.image_token_id
        self.ids = token_ids  # special token name -> id
        self.ratios = sorted(
            {
                (w, h)
                for n in range(self.min_tiles, self.max_tiles + 1)
                for w in range(1, n + 1)
                for h in range(1, n + 1)
                if self.min_tiles <= w * h <= self.max_tiles
            },
            key=lambda r: r[0] * r[1],
        )

    def _smart_resize(self, h: int, w: int) -> Tuple[int, int]:
        f = self.patch * self.factor
        lo, hi = self.min_tokens * f * f, self.max_tokens * f * f
        h_bar, w_bar = max(f, round(h / f) * f), max(f, round(w / f) * f)
        if h_bar * w_bar > hi:
            beta = math.sqrt(h * w / hi)
            h_bar = max(f, math.floor(h / beta / f) * f)
            w_bar = max(f, math.floor(w / beta / f) * f)
        elif h_bar * w_bar < lo:
            beta = math.sqrt(lo / (h * w))
            h_bar, w_bar = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
        return h_bar, w_bar

    def _too_large(self, h: int, w: int) -> bool:
        f = self.patch * self.factor
        h_bar, w_bar = max(self.patch, round(h / f) * f), max(self.patch, round(w / f) * f)
        return h_bar * w_bar > self.max_tokens * f * f * self.tolerance

    def _grid(self, h: int, w: int) -> Tuple[int, int]:
        """(columns, rows): HF find_closest_aspect_ratio."""
        aspect, best, best_diff = w / h, (1, 1), float("inf")
        for cols, rows in self.ratios:
            diff = abs(aspect - cols / rows)
            if diff < best_diff:
                best, best_diff = (cols, rows), diff
            elif diff == best_diff and h * w > 0.5 * self.tile * self.tile * cols * rows:
                best = (cols, rows)
        return best

    def _patches(self, image) -> Tuple[np.ndarray, int, int]:
        x = (np.asarray(image, dtype=np.float32) / 255.0 - self.mean) / self.std
        p = self.patch
        gh, gw = x.shape[0] // p, x.shape[1] // p
        x = x.reshape(gh, p, gw, p, 3).transpose(0, 2, 1, 3, 4)
        return x.reshape(gh * gw, p * p * 3), gh, gw

    def crops(self, image) -> Tuple[list, int, int, int]:
        """The crops [(patches, h, w)], the grid rows and columns, and the
        image tokens of the whole-image crop (single or thumbnail)."""
        from PIL import Image

        H, W = image.height, image.width
        new_h, new_w = self._smart_resize(H, W)
        rows = cols = 1
        out = []
        if self._too_large(H, W) and not self.min_tiles == self.max_tiles == 1:
            cols, rows = self._grid(H, W)
            big = image.resize((cols * self.tile, rows * self.tile), Image.BICUBIC)
            for r in range(rows):
                for c in range(cols):
                    box = (c * self.tile, r * self.tile, (c + 1) * self.tile, (r + 1) * self.tile)
                    out.append(self._patches(big.crop(box)))
            if self.thumbnail and rows * cols > 1:
                out.append(self._patches(image.resize((new_w, new_h), Image.BICUBIC)))
        else:
            out.append(self._patches(image.resize((new_w, new_h), Image.BICUBIC)))
        n = math.ceil(new_h // self.patch / self.factor) * math.ceil(new_w // self.patch / self.factor)
        return out, rows, cols, n

    def build(
        self, prompt: List[int], images: List[bytes]
    ) -> Tuple[List[int], mx.array, List[int]]:
        n_placeholders = sum(1 for t in prompt if t == self.image_token_id)
        if n_placeholders != len(images):
            raise ValueError(
                f"Chat template produced {n_placeholders} image placeholder(s) "
                f"for {len(images)} image(s)."
            )
        keys = [
            _IMAGE_KEY_BASE + int.from_bytes(hashlib.sha256(b).digest()[:5], "big")
            for b in images
        ]
        per_tile = math.ceil(self.tile // self.patch / self.factor) ** 2
        expanded, cache_key, crops, img = [], [], [], 0
        for tok in prompt:
            if tok != self.image_token_id:
                expanded.append(tok)
                cache_key.append(tok)
                continue
            image_crops, rows, cols, n_image = self.crops(_decode(images[img]))
            crops += [(mx.array(p), h, w) for p, h, w in image_crops]
            spans = []  # (marker token or None, image tokens)
            if rows * cols > 1:
                for r in range(rows):
                    for c in range(cols):
                        spans.append((self.ids[f"<|img_row_{r + 1}_col_{c + 1}|>"], per_tile))
                if self.thumbnail:
                    spans.append((self.ids["<|img_thumbnail|>"], n_image))
            else:
                spans.append((None, n_image))
            seq = [self.ids["<|image_start|>"]]
            key = list(seq)
            for marker, n in spans:
                if marker is not None:
                    seq.append(marker)
                    key.append(marker)
                seq += [self.image_token_id] * n
                key += [keys[img]] * n
            seq.append(self.ids["<|image_end|>"])
            key.append(self.ids["<|image_end|>"])
            expanded += seq
            cache_key += key
            img += 1

        embeddings = self.model.embed_with_images(mx.array(expanded)[None], crops)[0]
        mx.eval(embeddings)
        return expanded, embeddings, cache_key


def _special_token_ids(model_path) -> dict:
    """Added-token name -> id, from the model's tokenizer.json."""
    path = Path(model_path) / "tokenizer.json"
    if not path.exists():
        return {}
    added = json.loads(path.read_text()).get("added_tokens", [])
    return {t["content"]: t["id"] for t in added}


def vision_spans(cache_key: List[int]) -> List[Tuple[int, int]]:
    """[start, end) of each image's soft tokens, from a cache key built by
    ImageInputs.build (its image positions hold pseudo ids >= _IMAGE_KEY_BASE;
    images are always separated by their begin / end tokens)."""
    spans, start = [], None
    for i, t in enumerate(list(cache_key) + [0]):
        if t >= _IMAGE_KEY_BASE and start is None:
            start = i
        elif t < _IMAGE_KEY_BASE and start is not None:
            spans.append((start, i))
            start = None
    return spans


def load_image_inputs(model, model_path) -> Optional[ImageInputs]:
    """ImageInputs for a vision-capable Gemma 4 or Qwen3.5 model, else None."""
    if getattr(model, "model_type", None) in ("qwen3_5", "qwen3_5_moe"):
        if getattr(model, "vision_tower", None) is None:
            return None
        from .utils import _download

        cfg_path = Path(_download(str(model_path))) / "preprocessor_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        return Qwen35ImageInputs(model, cfg)
    if getattr(model, "model_type", None) in ("lfm2_vl", "lfm2-vl"):
        if getattr(model, "vision_tower", None) is None:
            return None
        from .utils import _download

        path = Path(_download(str(model_path)))
        # The image settings: processor_config.json's image_processor, with
        # preprocessor_config.json over it (where older checkpoints keep them).
        cfg = {}
        for name in ("processor_config.json", "preprocessor_config.json"):
            if (path / name).exists():
                data = json.loads((path / name).read_text())
                cfg.update(data.get("image_processor", data))
        return Lfm2VlImageInputs(model, cfg, _special_token_ids(path))
    if getattr(model, "model_type", None) not in ("gemma4", "gemma4_unified"):
        return None
    if getattr(model, "embed_vision", None) is None:
        return None
    from .utils import _download

    cfg_path = Path(_download(str(model_path))) / "processor_config.json"
    if not cfg_path.exists():
        return None
    return ImageInputs(model, json.loads(cfg_path.read_text()))
