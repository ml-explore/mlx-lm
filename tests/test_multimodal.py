# Copyright © 2026 Apple Inc.

"""Unit tests for mlx_lm/multimodal.py (image inputs in mlx_lm.server).

No model weights needed: the fusion step is stubbed, so this checks the
parts the server owns -- extracting images from OpenAI-style content parts,
expanding the chat template's single image placeholder into the model's
soft-token span, the pad-substituted generation ids, and the image-hashed
prompt-cache key. End-to-end correctness against HF's Gemma4Processor and a
real model was verified separately (input_ids identical to HF's processor;
logits through the input_embeddings path identical to the direct
pixel_values path; "two cats" on the COCO cats photo via the server).
"""

import base64
import io
import types
import unittest

import mlx.core as mx
import numpy as np

try:
    from PIL import Image

    from mlx_lm import multimodal
    from mlx_lm.multimodal import _IMAGE_KEY_BASE, ImageInputs, extract_images
    from mlx_lm.server import process_message_content

    HAVE_DEPS = True
except ImportError:  # pillow / transformers' Gemma 4 PIL processor missing
    HAVE_DEPS = False

IMAGE, BOI, EOI, PAD = 258880, 255999, 258882, 0


def _png(h, w, seed):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(
        buf, format="PNG"
    )
    return buf.getvalue()


def _data_uri(blob):
    return "data:image/png;base64," + base64.b64encode(blob).decode()


def _stub_model():
    captured = {}

    def fuse(ids, pixel_values, position_ids, feats, mask):
        captured["n"] = ids.shape[1]
        return mx.zeros((1, ids.shape[1], 8)), None

    model = types.SimpleNamespace(
        args=types.SimpleNamespace(
            image_token_id=IMAGE, boi_token_id=BOI, eoi_token_id=EOI
        ),
        language_model=types.SimpleNamespace(
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(pad_token_id=PAD)
            )
        ),
        _fuse_multimodal_inputs=fuse,
    )
    return model, captured


PROCESSOR_CONFIG = {
    "image_processor": {"patch_size": 16, "max_soft_tokens": 280, "pooling_kernel_size": 3}
}


@unittest.skipUnless(HAVE_DEPS, "needs pillow + transformers")
class TestMultimodal(unittest.TestCase):
    def test_extract_images_rewrites_parts_in_order(self):
        a, b = _png(32, 48, 0), _png(40, 24, 1)
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image_url", "image_url": {"url": _data_uri(a)}},
                    {"type": "image_url", "image_url": _data_uri(b)},
                ],
            },
        ]
        self.assertEqual(extract_images(messages), [a, b])
        self.assertEqual(
            messages[1]["content"],
            [{"type": "text", "text": "compare"}, {"type": "image"}, {"type": "image"}],
        )
        # The server must now leave that list for the chat template instead
        # of rejecting non-text parts.
        process_message_content(messages)
        self.assertEqual(messages[1]["content"][1], {"type": "image"})
        self.assertEqual(messages[0]["content"], "sys")

    def test_bad_image_url_is_a_value_error(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}}
                ],
            }
        ]
        with self.assertRaises(ValueError):
            extract_images(messages)
        with self.assertRaises(ValueError):
            extract_images(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "ftp://x"}}]}]
            )

    def test_text_only_content_still_flattened(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        process_message_content(messages)
        self.assertEqual(messages[0]["content"], "ab")

    def test_build_expands_placeholders_and_keys_cache_by_image(self):
        model, captured = _stub_model()
        ii = ImageInputs(model, PROCESSOR_CONFIG)
        a, b = _png(480, 640, 0), _png(900, 300, 1)
        prompt = [2, 7, IMAGE, 9, IMAGE, 11]

        ids, emb, key = ii.build(prompt, [a, b])

        # boi + N soft tokens + eoi per image, text tokens untouched.
        self.assertEqual(ids[:2], [2, 7])
        self.assertEqual(ids[2], BOI)
        self.assertEqual(ids[-1], 11)
        self.assertEqual(ids.count(BOI), 2)
        self.assertEqual(ids.count(EOI), 2)
        # Generation ids carry pad (not the image id) at image positions --
        # required for per-layer-embedding models to match HF.
        self.assertNotIn(IMAGE, ids)
        self.assertEqual(len(ids), len(key))
        self.assertEqual(emb.shape[0], len(ids))
        self.assertEqual(captured["n"], len(ids))

        # Cache key: pseudo ids at image positions, one value per image.
        pseudo = [t for t in key if t >= _IMAGE_KEY_BASE]
        self.assertEqual(len(pseudo), ids.count(PAD))
        self.assertEqual(len(set(pseudo)), 2)

        # Same image -> same key; different image -> key diverges at the
        # first image position (only the text before it can be reused).
        _, _, key_same = ii.build(prompt, [a, b])
        self.assertEqual(key, key_same)
        _, _, key_other = ii.build(prompt, [_png(480, 640, 5), b])
        first_diff = next(i for i, (x, y) in enumerate(zip(key, key_other)) if x != y)
        self.assertEqual(first_diff, 3)  # right after [2, 7, BOI]

    def test_build_rejects_placeholder_count_mismatch(self):
        model, _ = _stub_model()
        ii = ImageInputs(model, PROCESSOR_CONFIG)
        with self.assertRaises(ValueError):
            ii.build([2, IMAGE, 3], [_png(64, 64, 0), _png(64, 64, 1)])


if __name__ == "__main__":
    unittest.main()


def _msg(*parts):
    return [{"role": "user", "content": list(parts)}]


def _img_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


@unittest.skipUnless(HAVE_DEPS, "needs pillow + transformers")
class TestImageInputLimits(unittest.TestCase):
    """Requests are untrusted: what extract_images refuses (-> HTTP 400)."""

    def setUp(self):
        self.saved = {k: getattr(multimodal, k) for k in
                      ("ALLOW_IMAGE_URLS", "MAX_IMAGES", "MAX_IMAGE_BYTES", "MAX_IMAGE_PIXELS",
                       "MAX_REQUEST_PIXELS", "URL_FETCH_SECONDS")}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(multimodal, k, v)

    def _serve(self, handler_body):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            hits = []

            def do_GET(self):
                Handler.hits.append(self.path)
                handler_body(self)

            def log_message(self, *a):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}", Handler.hits

    def test_urls_not_fetched_unless_allowed(self):
        png = _png(8, 8, 0)

        def body(h):
            h.send_response(200)
            h.end_headers()
            h.wfile.write(png)

        base, hits = self._serve(body)
        with self.assertRaisesRegex(ValueError, "allow-image-urls"):
            extract_images(_msg(_img_part(base + "/x")))
        self.assertEqual(hits, [], "the server must not even request it")
        multimodal.ALLOW_IMAGE_URLS = True
        self.assertEqual(extract_images(_msg(_img_part(base + "/x"))), [png])

    def test_url_size_and_time_bounded(self):
        multimodal.ALLOW_IMAGE_URLS = True
        multimodal.MAX_IMAGE_BYTES = 1000

        def big(h):
            h.send_response(200)
            h.end_headers()   # no Content-Length: only the read cap stops it
            h.wfile.write(b"x" * 5000)

        base, _ = self._serve(big)
        with self.assertRaisesRegex(ValueError, "larger than"):
            extract_images(_msg(_img_part(base + "/big")))

        import time as _t
        multimodal.MAX_IMAGE_BYTES = 10**6
        multimodal.URL_FETCH_SECONDS = 1

        def drip(h):
            h.send_response(200)
            h.end_headers()
            try:
                for _ in range(20):
                    h.wfile.write(b"x")
                    h.wfile.flush()
                    _t.sleep(0.3)
            except OSError:
                pass

        base, _ = self._serve(drip)
        start = _t.monotonic()
        with self.assertRaisesRegex(ValueError, "longer than"):
            extract_images(_msg(_img_part(base + "/drip")))
        self.assertLess(_t.monotonic() - start, 3)

    def test_url_deadline_holds_against_a_dripped_head_or_chunk_line(self):
        # Each byte comes inside the socket timeout, and a blocked read
        # never got back to a deadline check between reads.
        import time as _t
        multimodal.ALLOW_IMAGE_URLS = True
        multimodal.URL_FETCH_SECONDS = 1

        def dripped(prefix, endless):
            def body(h):
                try:
                    h.wfile.write(prefix)
                    h.wfile.flush()
                    for _ in range(40):
                        h.wfile.write(endless)
                        h.wfile.flush()
                        _t.sleep(0.2)
                except OSError:
                    pass
            return body

        cases = {
            "head": dripped(b"HTTP/1.1 200 OK\r\nX-Slow: ", b"a"),
            "chunk line": dripped(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n", b"0"),
        }
        for name, body in cases.items():
            with self.subTest(name):
                base, _ = self._serve(body)
                start = _t.monotonic()
                with self.assertRaisesRegex(ValueError, "longer than"):
                    extract_images(_msg(_img_part(base + "/slow")))
                self.assertLess(_t.monotonic() - start, 3)

    def test_url_deadline_holds_against_a_dripped_tls_handshake(self):
        import socket
        import threading
        import time as _t

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        self.addCleanup(listener.close)

        def server():
            conn, _ = listener.accept()
            with conn:
                try:
                    conn.sendall(b"\x16\x03\x03\x40\x00")   # a 16 KB handshake record...
                    for _ in range(40):                        # ...that never arrives
                        conn.sendall(b"\x02")
                        _t.sleep(0.2)
                except OSError:
                    pass

        threading.Thread(target=server, daemon=True).start()
        multimodal.ALLOW_IMAGE_URLS = True
        multimodal.URL_FETCH_SECONDS = 1
        start = _t.monotonic()
        with self.assertRaisesRegex(ValueError, "longer than"):
            extract_images(_msg(_img_part(f"https://127.0.0.1:{listener.getsockname()[1]}/x.png")))
        self.assertLess(_t.monotonic() - start, 3)

    def test_url_deadline_holds_across_addresses_that_time_out(self):
        # Each address could take the whole socket timeout to fail.
        import socket
        import time as _t
        from unittest import mock

        class Blackhole:
            def __init__(self, *a):
                self.timeout = None

            def settimeout(self, t):
                self.timeout = t

            def connect(self, addr):
                _t.sleep(self.timeout)
                raise TimeoutError("timed out")

            def shutdown(self, how):
                pass

            def close(self):
                pass

        addrs = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"10.0.0.{i}", 80)) for i in range(4)]
        multimodal.ALLOW_IMAGE_URLS = True
        multimodal.URL_FETCH_SECONDS = 1
        with mock.patch("socket.getaddrinfo", return_value=addrs), mock.patch("socket.socket", Blackhole):
            start = _t.monotonic()
            with self.assertRaisesRegex(ValueError, "longer than|Could not load"):
                extract_images(_msg(_img_part("http://many.invalid/x.png")))
            self.assertLess(_t.monotonic() - start, 2)

    def test_url_deadline_holds_against_a_stalled_resolver(self):
        import time as _t
        from unittest import mock

        multimodal.ALLOW_IMAGE_URLS = True
        multimodal.URL_FETCH_SECONDS = 1
        with mock.patch("socket.getaddrinfo", side_effect=lambda *a, **k: _t.sleep(5)):
            start = _t.monotonic()
            with self.assertRaisesRegex(ValueError, "longer than"):
                extract_images(_msg(_img_part("http://slow-dns.invalid/x.png")))
            self.assertLess(_t.monotonic() - start, 2)

    def test_connect_falls_back_past_an_unusable_address_family(self):
        import socket
        from unittest import mock

        png = _png(8, 8, 0)

        def body(h):
            h.send_response(200)
            h.end_headers()
            h.wfile.write(png)

        base, _ = self._serve(body)
        port = int(base.rsplit(":", 1)[1])
        real_socket = socket.socket

        def make(family=socket.AF_INET, *a, **kw):
            if family == socket.AF_INET6:
                raise OSError(47, "Address family not supported")
            return real_socket(family, *a, **kw)

        addrs = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port, 0, 0)),
                 (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        multimodal.ALLOW_IMAGE_URLS = True
        with mock.patch("socket.getaddrinfo", return_value=addrs), mock.patch("socket.socket", make):
            self.assertEqual(extract_images(_msg(_img_part(f"http://dual.invalid:{port}/x"))), [png])

    def test_failed_tls_handshake_closes_its_socket(self):
        import gc
        import os
        import warnings

        def plain(h):   # plain HTTP on a port fetched as https://
            h.send_response(200)
            h.end_headers()

        base, _ = self._serve(plain)
        multimodal.ALLOW_IMAGE_URLS = True
        url = base.replace("http://", "https://") + "/x"
        gc.disable()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                before = len(os.listdir("/dev/fd"))
                for _ in range(20):
                    with self.assertRaises(ValueError):
                        extract_images(_msg(_img_part(url)))
                after = len(os.listdir("/dev/fd"))
        finally:
            gc.enable()
        self.assertLess(after - before, 5, "sockets left open")
        self.assertFalse([w for w in caught if "SSLSocket" in str(w.message)])

    def test_url_deadline_holds_through_a_proxy_connect(self):
        # An HTTPS fetch through a proxy that drips its CONNECT response:
        # read inside connect(), before the socket used to be registered.
        import os
        import socket
        import threading
        import time as _t

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        self.addCleanup(listener.close)

        def proxy():
            conn, _ = listener.accept()
            with conn:
                conn.recv(4096)
                try:
                    for byte in b"HTTP/1.1 200 Connection established" * 10:
                        conn.sendall(bytes([byte]))
                        _t.sleep(0.2)
                except OSError:
                    pass

        threading.Thread(target=proxy, daemon=True).start()
        multimodal.ALLOW_IMAGE_URLS = True
        multimodal.URL_FETCH_SECONDS = 1
        env = {"https_proxy": f"http://127.0.0.1:{listener.getsockname()[1]}", "no_proxy": ""}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            start = _t.monotonic()
            with self.assertRaisesRegex(ValueError, "longer than"):
                extract_images(_msg(_img_part("https://example.invalid/x.png")))
            self.assertLess(_t.monotonic() - start, 3)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_data_uri_size_and_pixel_caps(self):
        multimodal.MAX_IMAGE_BYTES = 100
        with self.assertRaisesRegex(ValueError, "larger than"):
            extract_images(_msg(_img_part(_data_uri(_png(64, 64, 0)))))
        # Checked decoded too: the encoded length's allowance let a padded
        # image a little over the cap through.
        png = _png(8, 8, 0)
        multimodal.MAX_IMAGE_BYTES = 1000
        padded = png + b"\0" * (1001 - len(png))
        with self.assertRaisesRegex(ValueError, "larger than"):
            extract_images(_msg(_img_part(_data_uri(padded))))
        self.assertEqual(len(extract_images(_msg(_img_part(_data_uri(padded[:1000]))))), 1)
        multimodal.MAX_IMAGE_BYTES = 20 * 1024 * 1024
        # A decompression bomb: a small file declaring a huge canvas.
        buf = io.BytesIO()
        Image.new("1", (13000, 13000)).save(buf, format="PNG")
        self.assertLess(len(buf.getvalue()), 100_000)
        with self.assertRaisesRegex(ValueError, "pixels"):
            extract_images(_msg(_img_part(_data_uri(buf.getvalue()))))

    def test_formats_whose_open_decodes_are_refused(self):
        # ICO decodes its embedded PNG inside Image.open: refused by format,
        # before any of that.
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="ICO")
        with self.assertRaisesRegex(ValueError, "accepted: BMP, GIF, JPEG, PNG, WEBP"):
            extract_images(_msg(_img_part(_data_uri(buf.getvalue()))))
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="TIFF")
        with self.assertRaises(ValueError):
            extract_images(_msg(_img_part(_data_uri(buf.getvalue()))))
        for fmt in ("PNG", "JPEG", "WEBP", "GIF", "BMP"):
            buf = io.BytesIO()
            Image.new("RGB", (16, 16)).save(buf, format=fmt)
            self.assertEqual(len(extract_images(_msg(_img_part(_data_uri(buf.getvalue()))))), 1, fmt)

    def test_request_pixel_budget(self):
        multimodal.MAX_REQUEST_PIXELS = 3000
        uri = _data_uri(_png(40, 40, 0))   # 1600 px each
        with self.assertRaisesRegex(ValueError, "add up"):
            extract_images(_msg(_img_part(uri), _img_part(uri)))

    def test_unhashable_part_type(self):
        with self.assertRaises(ValueError):
            extract_images(_msg({"type": ["text"], "text": "x"}))

    def test_redirect_to_other_scheme_refused(self):
        multimodal.ALLOW_IMAGE_URLS = True

        def body(h):
            h.send_response(302)
            h.send_header("Location", "ftp://127.0.0.1/x")
            h.end_headers()

        base, _ = self._serve(body)
        with self.assertRaisesRegex(ValueError, "scheme"):
            extract_images(_msg(_img_part(base + "/r")))

    def test_jpeg_scan_count_capped(self):
        import struct

        # A valid progressive-style JPEG with extra (empty) scans spliced in
        # before EOI: decoding time grows with the scan count.
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="JPEG", progressive=True)
        jpg = buf.getvalue()
        normal_scans = jpg.count(b"\xff\xda")
        self.assertLess(normal_scans, 20)
        self.assertEqual(len(extract_images(_msg(_img_part(_data_uri(jpg))))), 1)
        sos = b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00"
        bomb = jpg[:-2] + sos * 200 + b"\xff\xd9"
        with self.assertRaisesRegex(ValueError, "scans"):
            extract_images(_msg(_img_part(_data_uri(bomb))))
        # FF DA bytes in a comment aren't scans (a byte count took them).
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="JPEG")
        comment = b"\xff\xda" * 150
        com = b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment
        jpg = buf.getvalue()
        with_comment = jpg[:2] + com + jpg[2:]
        Image.open(io.BytesIO(with_comment)).load()
        self.assertEqual(len(extract_images(_msg(_img_part(_data_uri(with_comment))))), 1)
        # A junk byte between segments: the decoder skips it, so the scans
        # after it still count.
        app0_end = 4 + int.from_bytes(bomb[4:6], "big")
        junk = bomb[:app0_end] + b"x" + bomb[app0_end:]
        with self.assertRaisesRegex(ValueError, "scans"):
            extract_images(_msg(_img_part(_data_uri(junk))))
        # A padded marker a walker would misread as a segment's length.
        padded = jpg[:-2] + b"\xff\xff\x00\x7f\x7f" + sos * 200 + b"\xff\xd9"
        with self.assertRaisesRegex(ValueError, "scans"):
            extract_images(_msg(_img_part(_data_uri(padded))))

    def test_not_an_image(self):
        uri = "data:image/png;base64," + base64.b64encode(b"hello, not an image").decode()
        with self.assertRaisesRegex(ValueError, "image data"):
            extract_images(_msg(_img_part(uri)))

    def test_image_count_cap(self):
        multimodal.MAX_IMAGES = 2
        uri = _data_uri(_png(8, 8, 0))
        with self.assertRaisesRegex(ValueError, "At most 2"):
            extract_images(_msg(_img_part(uri), _img_part(uri), _img_part(uri)))

    def test_malformed_structure(self):
        uri = _data_uri(_png(8, 8, 0))
        for messages in (
            _msg("hi", _img_part(uri)),                       # a bare string part
            ["hello"],                                        # a bare string message
            _msg({"type": "image"}),                          # a placeholder without an image
            _msg(_img_part(uri), {"type": "input_audio", "input_audio": {}}),
            _msg(_img_part(uri), {"type": "video"}),
            "not a list",
        ):
            with self.assertRaises(ValueError, msg=repr(messages)[:80]):
                extract_images(messages)
        # Text next to images is fine.
        self.assertEqual(len(extract_images(_msg({"type": "text", "text": "a"}, _img_part(uri)))), 1)

    def test_exif_orientation_applied(self):
        # Stored 40 wide x 20 high, EXIF says "rotate 90" (Orientation=6):
        # upright it's 20 x 40.
        im = Image.new("RGB", (40, 20))
        exif = im.getexif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        im.save(buf, format="JPEG", exif=exif.tobytes())
        self.assertEqual(multimodal._decode(buf.getvalue()).size, (20, 40))


AUDIO, BOA, EOA = 258881, 256000, 258883
AUDIO_CONFIG = {
    **PROCESSOR_CONFIG,
    "feature_extractor": {
        "feature_extractor_type": "Gemma4AudioFeatureExtractor", "dither": 0.0, "feature_size": 128,
        "fft_length": 512, "fft_overdrive": False, "frame_length": 320, "hop_length": 160,
        "input_scale_factor": 1.0, "max_frequency": 8000.0, "mel_floor": 0.001, "min_frequency": 0.0,
        "padding_side": "right", "padding_value": 0.0, "per_bin_mean": None, "per_bin_stddev": None,
        "preemphasis": 0.0, "preemphasis_htk_flavor": True, "return_attention_mask": True,
        "sampling_rate": 16000,
    },
}


def _wav(seconds, rate=16000, channels=1, freq=440.0):
    import wave

    t = np.arange(int(seconds * rate)) / rate
    x = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    if channels > 1:
        x = np.repeat(x[:, None], channels, axis=1).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def _audio_part(blob):
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(blob).decode(), "format": "wav"}}


def _audio_model(unified=False):
    model, captured = _stub_model()

    def fuse(ids, pixel_values, position_ids, feats, mask):
        captured["n"] = ids.shape[1]
        captured["feats"] = None if feats is None else feats.shape
        captured["pixels"] = pixel_values is not None
        return mx.zeros((1, ids.shape[1], 8)), None

    model._fuse_multimodal_inputs = fuse
    model.args.audio_token_id = AUDIO
    model.args.boa_token_id = BOA
    model.args.eoa_token_index = EOA
    model.embed_audio = object()
    model.unified = unified
    return model, captured


@unittest.skipUnless(HAVE_DEPS, "needs pillow + transformers")
class TestAudioInputs(unittest.TestCase):
    def test_extract_media_decodes_and_rewrites(self):
        from mlx_lm.multimodal import extract_media

        messages = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                                  _audio_part(_wav(1.5, rate=48000, channels=2))]}]
        images, audios = extract_media(messages)
        self.assertEqual(images, [])
        self.assertEqual(len(audios), 1)
        self.assertEqual(audios[0].dtype, np.float32)
        self.assertAlmostEqual(len(audios[0]) / 16000, 1.5, places=2)  # 48 kHz stereo -> 16 kHz mono
        self.assertLess(np.abs(audios[0]).max(), 0.6)
        self.assertEqual(messages[0]["content"][1], {"type": "audio"})
        process_message_content(messages)
        self.assertEqual(messages[0]["content"][1], {"type": "audio"}, "left for the chat template")

    def test_bad_audio_is_a_value_error(self):
        from mlx_lm.multimodal import extract_media

        bad = [
            [{"type": "input_audio", "input_audio": {"data": base64.b64encode(b"not a wav").decode(), "format": "wav"}}],
            [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}}],
            [{"type": "input_audio", "input_audio": {}}],
            [{"type": "audio"}],  # only extract_media makes those
        ]
        for content in bad:
            with self.assertRaises(ValueError, msg=str(content)[:60]):
                extract_media([{"role": "user", "content": content}])

    def test_build_expands_audio_as_hf_counts(self):
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

        model, captured = _audio_model()
        ii = ImageInputs(model, AUDIO_CONFIG)
        self.assertTrue(ii.accepts_audio)
        clips = [np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32) * 0.1,
                 np.zeros(16000 // 2, dtype=np.float32)]
        ids, _, key = ii.build([2, AUDIO, 9, AUDIO, 11], [], clips)
        # HF's processor: the mask through two stride-2 convolutions.
        fe = Gemma4AudioFeatureExtractor(**{k: v for k, v in AUDIO_CONFIG["feature_extractor"].items()
                                            if k != "feature_extractor_type"})
        feats = fe(clips, sampling_rate=16000, return_tensors="np")
        expected = []
        for mask in feats["input_features_mask"]:
            for _ in range(2):
                t = len(mask)
                mask = mask[::2][: (t + 2 - 3) // 2 + 1]
            expected.append(int(mask.sum()))
        self.assertEqual(ids.count(BOA), 2)
        self.assertEqual(ids.count(EOA), 2)
        self.assertEqual(len(ids), 3 + sum(expected) + 4)
        self.assertNotIn(AUDIO, ids, "pad at the soft-token positions, as the fusion")
        self.assertEqual(len(key), len(ids))
        self.assertEqual(captured["feats"][0], 2)
        self.assertFalse(captured["pixels"])
        # 3 s is 75 soft tokens at 40 ms each (HF's audio_ms_per_token).
        self.assertEqual(expected[0], 75)

    def test_unified_counts_frames(self):
        model, _ = _audio_model(unified=True)
        cfg = {**PROCESSOR_CONFIG, "feature_extractor": {"feature_extractor_type": "Gemma4UnifiedAudioFeatureExtractor"}}
        ii = ImageInputs(model, cfg)
        self.assertEqual(ii.audio_soft_tokens(np.ones(50, dtype=bool)), 50)

    def test_placeholders_must_match_and_a_model_without_audio_refuses(self):
        model, _ = _audio_model()
        ii = ImageInputs(model, AUDIO_CONFIG)
        with self.assertRaises(ValueError):
            ii.build([2, AUDIO, AUDIO], [], [np.zeros(16000, dtype=np.float32)])
        plain, _ = _stub_model()
        no_audio = ImageInputs(plain, PROCESSOR_CONFIG)
        self.assertFalse(no_audio.accepts_audio)
        with self.assertRaises(ValueError):
            no_audio.build([2, 11], [], [np.zeros(16000, dtype=np.float32)])
