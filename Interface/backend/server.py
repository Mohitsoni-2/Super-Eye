"""
Super Eye v3.1 — backend/server.py  (ROOT CAUSE FIXED)

ROOT CAUSE OF DETECTION FAILURE:
  The backend was calling cv2.VideoCapture(0) to open webcam directly.
  But the browser ALSO had the webcam open via getUserMedia.
  On Windows, two processes cannot share the same webcam simultaneously.
  So the backend was getting empty/black frames → no detections.

THE FIX:
  For LIVE mode: browser captures frames from <video> element and sends
  them as base64 JPEG via WebSocket ('live_frame' messages).
  Backend receives those frames and runs face detection on them.
  Backend never opens its own camera for live mode.

  For RECORDED mode: same as before — browser sends frames from <video>.
"""

import asyncio, base64, json, sys, time
import cv2, numpy as np, face_recognition, websockets
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

HOST            = "localhost"
PORT            = 8765
THIS_DIR        = Path(__file__).resolve().parent
ROOT_DIR        = THIS_DIR.parent
PEOPLE_DIR      = ROOT_DIR / "lost_people_images"
REPORT_FILE     = ROOT_DIR / "found_people_report.csv"

SCALE           = 0.5    # downscale factor before detection
MATCH_THRESHOLD = 0.6    # face distance threshold (lower = stricter)
NUM_JITTERS     = 1      # jitter for encoding (1=fast, 2=more accurate)
MODEL           = "hog"  # "hog"=CPU fast, "cnn"=GPU best
COOLDOWN_SEC    = 5.0    # min seconds between same-person detections


def preprocess_rgb(img_bgr):
    """CLAHE contrast enhancement then convert to RGB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_encoding(img_bgr, num_jitters=NUM_JITTERS):
    """Extract face encoding from an image file. Tries multiple strategies."""
    strategies = [
        lambda i: preprocess_rgb(i),
        lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2RGB),
        lambda i: cv2.cvtColor(cv2.resize(i, (0,0), fx=2, fy=2), cv2.COLOR_BGR2RGB),
    ]
    for fn in strategies:
        try:
            rgb  = fn(img_bgr)
            locs = face_recognition.face_locations(rgb, model=MODEL)
            if locs:
                largest = max(locs, key=lambda l: (l[2]-l[0])*(l[1]-l[3]))
                encs = face_recognition.face_encodings(rgb, [largest], num_jitters=num_jitters)
                if encs:
                    return encs[0]
        except Exception:
            pass
    return None


class SuperEyeBackend:

    def __init__(self):
        self.known_encodings = []
        self.known_names     = []
        self.clients         = set()
        self._load_people()

    def _load_people(self):
        print(f"\n[INFO] People folder : {PEOPLE_DIR}")
        if not PEOPLE_DIR.exists():
            print(f"[WARN] Folder not found — create it and add photos.")
            return
        images = [f for f in sorted(PEOPLE_DIR.iterdir())
                  if f.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp')]
        print(f"[INFO] Found {len(images)} images\n")
        loaded = 0
        for f in images:
            print(f"  {f.name} ... ", end="", flush=True)
            img = cv2.imread(str(f))
            if img is None:
                print("FAILED (cannot read)")
                continue
            enc = load_encoding(img)
            if enc is not None:
                self.known_encodings.append(enc)
                self.known_names.append(f.stem)
                loaded += 1
                print("OK")
            else:
                print("SKIP (no face found)")
        print(f"\n[INFO] {loaded}/{len(images)} people loaded\n")

    def add_person_b64(self, name, b64):
        try:
            raw = base64.b64decode(b64.split(",")[-1])
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None: return False
            enc = load_encoding(img, num_jitters=2)
            if enc is None:
                print(f"  [!] No face in photo for: {name}")
                return False
            if name in self.known_names:
                self.known_encodings[self.known_names.index(name)] = enc
                print(f"  [~] Updated : {name}")
            else:
                self.known_encodings.append(enc)
                self.known_names.append(name)
                print(f"  [+] Added   : {name}")
            return True
        except Exception as e:
            print(f"  [ERR] add_person: {e}")
            return False

    def detect_in_frame(self, frame_bgr):
        """Run face detection. Returns list of {name, confidence, bbox}."""
        if not self.known_encodings:
            return []

        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (int(w * SCALE), int(h * SCALE)))
        rgb   = preprocess_rgb(small)

        locs = face_recognition.face_locations(rgb, model=MODEL)
        if not locs:
            return []

        encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
        inv  = 1.0 / SCALE
        out  = []

        for enc, loc in zip(encs, locs):
            dists = face_recognition.face_distance(self.known_encodings, enc)
            idx   = int(np.argmin(dists))
            dist  = float(dists[idx])
            conf  = round(max(0.0, 1.0 - dist), 4)
            name  = self.known_names[idx] if dist <= MATCH_THRESHOLD else "Unknown"
            top, right, bottom, left = loc
            bbox  = [int(top*inv), int(right*inv), int(bottom*inv), int(left*inv)]
            out.append({"name": name, "confidence": conf, "bbox": bbox})
        return out

    def log_csv(self, name, source, video_ts=None):
        try:
            is_new = not REPORT_FILE.exists()
            if source == "live":
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elif video_ts is not None:
                h,m,s = int(video_ts//3600), int((video_ts%3600)//60), int(video_ts%60)
                ts = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                ts = ""
            with open(REPORT_FILE, "a") as f:
                if is_new: f.write("NAME,TIMESTAMP,SOURCE\n")
                f.write(f'"{name}","{ts}","{source}"\n')
        except Exception as e:
            print(f"[ERR] log_csv: {e}")

    async def _process_frame_b64(self, b64, source, timestamp, cooldowns, send_fn):
        """Decode base64 frame, detect faces, send results."""
        try:
            raw   = base64.b64decode(b64.split(",")[-1])
            arr   = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            loop = asyncio.get_event_loop()
            dets = await loop.run_in_executor(None, self.detect_in_frame, frame)
            now  = time.time()

            for d in dets:
                if d["name"] == "Unknown":
                    continue
                key = d["name"] + "_" + source
                if now - cooldowns.get(key, 0) < COOLDOWN_SEC:
                    continue
                cooldowns[key] = now
                self.log_csv(d["name"], source, timestamp)
                print(f"  [DET] {d['name']}  conf={d['confidence']:.2f}  source={source}")
                await send_fn({
                    "type":       "detection",
                    "name":       d["name"],
                    "confidence": d["confidence"],
                    "bbox":       d["bbox"],
                    "timestamp":  timestamp,
                    "source":     source,
                })
        except Exception as e:
            print(f"  [ERR] _process_frame: {e}")
            import traceback; traceback.print_exc()

    async def handle(self, ws):
        self.clients.add(ws)
        print(f"\n[WS] Browser connected  (total clients: {len(self.clients)})")
        cooldowns = {}

        async def send(obj):
            try: await ws.send(json.dumps(obj))
            except Exception: pass

        try:
            await send({"type": "ready", "people": self.known_names})
            print(f"[WS] Sent ready — people: {self.known_names}")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                t = msg.get("type", "")

                # ── Live frame from browser webcam ─────────────────────
                # This is the KEY fix — browser sends frames, we detect here
                if t == "live_frame":
                    b64 = msg.get("data", "")
                    await self._process_frame_b64(b64, "live", None, cooldowns, send)

                # ── Recorded video frame ───────────────────────────────
                elif t == "frame":
                    b64 = msg.get("data", "")
                    ts  = msg.get("timestamp", 0.0)
                    await self._process_frame_b64(b64, "recorded", ts, cooldowns, send)

                # ── Add person from UI ─────────────────────────────────
                elif t == "add_person":
                    name = msg.get("name", "")
                    ok   = self.add_person_b64(name, msg.get("image", ""))
                    await send({"type": "add_result", "name": name, "success": ok})

                # ── Stop (no-op now, browser just stops sending frames) ─
                elif t == "stop":
                    print("[WS] Stop received")

                # ── Legacy start_live (ignore — no longer open camera) ─
                elif t == "start_live":
                    print("[WS] start_live received (ignored — browser sends frames now)")

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[ERR] handle: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.clients.discard(ws)
            print(f"[WS] Disconnected  (remaining: {len(self.clients)})\n")


async def main():
    be = SuperEyeBackend()
    print("╔══════════════════════════════════════════╗")
    print("║   Super Eye Backend  v3.1  (ROOT FIXED)  ║")
    print(f"║   ws://{HOST}:{PORT}                    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  People  : {len(be.known_names)}")
    print(f"  Names   : {be.known_names}")
    print(f"  Model   : {MODEL}  |  Threshold: {MATCH_THRESHOLD}  |  Scale: {SCALE}")
    print("\nWaiting for browser to connect...\n")
    async with websockets.serve(be.handle, HOST, PORT,
                                ping_interval=20, ping_timeout=30,
                                max_size=10 * 1024 * 1024):  # 10MB max frame size
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
