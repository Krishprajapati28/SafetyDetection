#!/usr/bin/env python3
"""
Final PPE Safety Detection System — improved matching & association

Key fixes:
- Robust label normalization (handles "no gloves" vs "no-gloves" vs "no_gloves")
- Token-based matching (avoids matching "gloves" inside "no gloves")
- Associate PPE -> person by PPE-box center inside person bbox (more robust)
- Resolve conflicts by choosing the detection (positive vs negative) with the higher confidence
- Optionally treat "no detection" as missing (see `infer_missing_if_no_positive`)
"""

from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os
import re

class PPEDetector:
    def __init__(self,
                 person_model_path="newsafe.pt",
                 ppe_model_path="PPE.pt",
                 pos_conf_thresh=0.30,
                 neg_conf_thresh=0.30,
                 infer_missing_if_no_positive=True,
                 debug=False):
        """Initialize the PPE detection system."""
        self.debug = debug
        print("Loading models...")
        self.person_model = YOLO(person_model_path)
        self.ppe_model = YOLO(ppe_model_path)

        print(f"Person model classes: {self.person_model.names}")
        print(f"PPE model classes: {self.ppe_model.names}")

        # PPE canonical items (change to match your model's intended items)
        self.ppe_items = ['helmet', 'vest', 'gloves', 'shoes']

        # thresholds (tune if necessary)
        self.pos_conf_thresh = pos_conf_thresh
        self.neg_conf_thresh = neg_conf_thresh

        # If True, if no positive or negative detection exists for an item we mark it "missing".
        # Set False if you prefer "unknown" instead of assuming missing.
        self.infer_missing_if_no_positive = infer_missing_if_no_positive

        # Colors (BGR)
        self.colors = {
            'safe': (0, 255, 0),      # Green
            'unsafe': (0, 0, 255),    # Red
            'text': (255, 255, 255)   # White
        }

        # Determine the class id that corresponds to "person" in the person model
        self.person_class_id = self._get_class_id_by_name(self.person_model, 'person')
        if self.person_class_id is None:
            print("Warning: could not find 'person' class id in person model names — "
                  "falling back to common IDs (0 or 6). If detections are wrong, update model or class mapping.")

    # -------------------------
    # Helper utilities
    # -------------------------
    def _get_class_id_by_name(self, model, target_name):
        """Return class id for target_name (case-insensitive) or None."""
        try:
            names = model.names  # dict-like: id -> name
        except Exception:
            return None
        for k, v in names.items():
            try:
                if str(v).lower() == target_name.lower():
                    return int(k)
            except Exception:
                continue
        # fallback: find a name that contains the word
        for k, v in names.items():
            try:
                if target_name.lower() in str(v).lower():
                    return int(k)
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_label(label):
        """Normalize model label to lowercase, remove punctuation, and unify separators."""
        if label is None:
            return ""
        s = str(label).lower()
        s = s.replace('-', ' ').replace('_', ' ')
        s = re.sub(r'[^a-z0-9\s]', '', s)  # remove other punctuation
        s = ' '.join(s.split())
        return s

    def _parse_ppe_label(self, label):
        """
        Parse normalized label into (item, is_negative)
        Returns (item_name or None, is_negative: bool).
        """
        nl = self._normalize_label(label)
        if nl == "":
            return None, False

        words = nl.split()
        is_negative = False
        target = nl

        # check for negative prefix
        if words[0] in ('no', 'not', 'none', 'without', 'absent'):
            is_negative = True
            target = ' '.join(words[1:]).strip()
        # also accept labels like "nohelmet" (rare)
        # Now match canonical items
        for item in self.ppe_items:
            if item in target.split() or item in target:
                return item, is_negative

        # nothing matched
        return None, is_negative

    # -------------------------
    # Detection helpers
    # -------------------------
    def detect_persons(self, frame, conf_threshold=0.5):
        """Detect persons in the frame and return list of {'box':(x1,y1,x2,y2), 'conf':conf}."""
        results = self.person_model(frame, conf=conf_threshold, verbose=False)
        persons = []

        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                except Exception:
                    # fallback: if box.cls not present, skip
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_person = False
                if self.person_class_id is not None:
                    is_person = (cls_id == self.person_class_id)
                else:
                    # fallback: match label text if available
                    label = str(self.person_model.names.get(cls_id, '')).lower() if hasattr(self.person_model, 'names') else ''
                    if 'person' in label or cls_id in (0, 6):
                        is_person = True

                if is_person:
                    persons.append({'box': (x1, y1, x2, y2), 'conf': conf})

        return persons

    def detect_ppe_items(self, frame, conf_threshold=0.3):
        """Detect PPE items in the frame and return list of detections with labels."""
        results = self.ppe_model(frame, conf=conf_threshold, verbose=False)
        ppe_detections = []

        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                except Exception:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = str(self.ppe_model.names.get(cls_id, str(cls_id))).lower() if hasattr(self.ppe_model, 'names') else str(cls_id)
                ppe_detections.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'label': label,
                    'cls_id': cls_id
                })

        return ppe_detections

    def check_ppe_coverage(self, person_box, ppe_detections):
        """
        Check PPE coverage for a specific person.
        Strategy:
          - For each PPE detection whose center falls inside the person bbox:
              - parse label into (item, polarity)
              - keep best positive and best negative detection for each item (by confidence)
          - Decide final status per item by comparing positive vs negative best confidences
          - If nothing found: optionally infer missing
        """
        px1, py1, px2, py2 = person_box

        # initialize best holders
        best_pos = {item: {'conf': 0.0, 'box': None, 'label': None} for item in self.ppe_items}
        best_neg = {item: {'conf': 0.0, 'box': None, 'label': None} for item in self.ppe_items}

        # scan detections
        for detection in ppe_detections:
            label_raw = detection['label']
            label_norm = self._normalize_label(label_raw)
            hx1, hy1, hx2, hy2 = detection['box']
            conf = detection['conf']

            # associate detection -> person by center point
            cx = (hx1 + hx2) / 2.0
            cy = (hy1 + hy2) / 2.0
            if not (px1 <= cx <= px2 and py1 <= cy <= py2):
                continue  # not associated to this person

            item, is_negative = self._parse_ppe_label(label_norm)
            if item is None:
                if self.debug:
                    print("Unrecognized PPE label (ignored):", label_raw)
                continue

            if is_negative:
                if conf > best_neg[item]['conf']:
                    best_neg[item].update({'conf': conf, 'box': detection['box'], 'label': label_raw})
            else:
                if conf > best_pos[item]['conf']:
                    best_pos[item].update({'conf': conf, 'box': detection['box'], 'label': label_raw})

        # build status
        ppe_status = {}
        for item in self.ppe_items:
            ppe_status[item] = {
                'detected': False,
                'missing': False,
                'box': None,
                'conf': 0.0,
                'label': None
            }
            posc = best_pos[item]['conf']
            pos_box = best_pos[item]['box']
            pos_label = best_pos[item]['label']

            negc = best_neg[item]['conf']
            neg_box = best_neg[item]['box']
            neg_label = best_neg[item]['label']

            # Choose winner by confidence (prefer positive if equal)
            if posc >= self.pos_conf_thresh and posc >= negc:
                ppe_status[item]['detected'] = True
                ppe_status[item]['box'] = pos_box
                ppe_status[item]['conf'] = posc
                ppe_status[item]['label'] = pos_label or item
            elif negc >= self.neg_conf_thresh and negc > posc:
                ppe_status[item]['missing'] = True
                ppe_status[item]['label'] = neg_label or f"no {item}"
                ppe_status[item]['conf'] = negc
                ppe_status[item]['box'] = neg_box
            else:
                # No strong detection either way
                if self.infer_missing_if_no_positive:
                    ppe_status[item]['missing'] = True
                    ppe_status[item]['label'] = f"no {item} (inferred)"
                else:
                    # Leave as unknown (both False)
                    ppe_status[item]['label'] = f"{item} (unknown)"

        return ppe_status

    def draw_detection(self, frame, person, ppe_status, person_id):
        """Draw person and PPE detection results and return whether person is safe."""
        px1, py1, px2, py2 = person['box']

        # Count detected items
        detected_count = sum(1 for s in ppe_status.values() if s['detected'])

        # Determine overall safety
        is_safe = (detected_count == len(self.ppe_items))
        safety_color = self.colors['safe'] if is_safe else self.colors['unsafe']

        # Draw person bounding box (green if safe, red if unsafe)
        cv2.rectangle(frame, (px1, py1), (px2, py2), safety_color, 2)
        cv2.putText(frame, f"Person {person_id}", (px1, max(py1-10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, safety_color, 2)

        # Draw per-item status lines inside/near person bbox
        y_offset = max(py1 - 10, 12)  # start from just above the person box (clamped)

        for item, status in ppe_status.items():
            if status['detected']:
                # Draw a small box around the PPE (if provided)
                if status['box']:
                    hx1, hy1, hx2, hy2 = status['box']
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), self.colors['safe'], 2)
                    cv2.putText(frame, f"{status['label']} ({status['conf']:.2f})",
                                (hx1, max(hy1-6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.colors['safe'], 1)
                # status text
                cv2.putText(frame, f"✓ {item}", (px1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['safe'], 2)
            elif status['missing']:
                # mark missing
                cv2.putText(frame, f"✗ {status['label']}", (px1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['unsafe'], 2)
                # optionally draw negative detection box (if model predicted explicit "no" box)
                if status.get('box'):
                    hx1, hy1, hx2, hy2 = status['box']
                    # Draw red dashed-ish rectangle: here we draw solid red with thickness 1
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), self.colors['unsafe'], 1)

            y_offset -= 18

        # Draw overall status
        cv2.putText(frame, f"Status: {'SAFE' if is_safe else 'UNSAFE'}", (px1, py2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, safety_color, 2)

        return is_safe

    # -------------------------
    # Process routines
    # -------------------------
    def process_image(self, input_path, output_path=None):
        """Process a single image."""
        print(f"Processing image: {input_path}")

        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Could not load image {input_path}")
            return False

        print(f"Image loaded: {frame.shape}")

        persons = self.detect_persons(frame)
        ppe_detections = self.detect_ppe_items(frame)

        if self.debug:
            print("Detected persons:", persons)
            print("Detected PPE items (raw):")
            for d in ppe_detections:
                print(d)

        safe_persons = 0
        for i, person in enumerate(persons):
            ppe_status = self.check_ppe_coverage(person['box'], ppe_detections)
            is_safe = self.draw_detection(frame, person, ppe_status, i + 1)
            if is_safe:
                safe_persons += 1

        cv2.putText(frame, f"Safe: {safe_persons}/{len(persons)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)

        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"Result saved to: {output_path}")

        return True

    def process_video(self, input_path, output_path=None):
        """Process video file."""
        print(f"Processing video: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            persons = self.detect_persons(frame)
            ppe_detections = self.detect_ppe_items(frame)

            safe_persons = 0
            for i, person in enumerate(persons):
                ppe_status = self.check_ppe_coverage(person['box'], ppe_detections)
                is_safe = self.draw_detection(frame, person, ppe_status, i + 1)
                if is_safe:
                    safe_persons += 1

            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            cv2.putText(frame, f"Safe: {safe_persons}/{len(persons)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)

            if out:
                out.write(frame)

            if frame_count % 50 == 0:
                print(f"Processed frame {frame_count}")

        cap.release()
        if out:
            out.release()

        print(f"Video processing complete. Processed {frame_count} frames.")
        if output_path:
            print(f"Output saved to: {output_path}")

        return True

# -----------------------------------------------------------------------------
# CLI helper (same behavior as your original main)
# -----------------------------------------------------------------------------
def main():
    detector = PPEDetector(debug=False)

    if len(sys.argv) == 1:
        print("\n=== PPE Safety Detection System ===")
        print("1. Process image")
        print("2. Process video")
        print("3. Test with safety.jpg")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            input_path = input("Enter image path: ").strip()
            output_path = input("Enter output path (optional): ").strip() or None
            detector.process_image(input_path, output_path)

        elif choice == "2":
            input_path = input("Enter video path: ").strip()
            output_path = input("Enter output path (optional): ").strip() or None
            detector.process_video(input_path, output_path)

        elif choice == "3":
            detector.process_image("safety.jpg", "ppe_result.jpg")

        else:
            print("Invalid choice")

    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            detector.process_image(input_path, f"ppe_result_{os.path.basename(input_path)}")
        else:
            detector.process_video(input_path, f"ppe_result_{os.path.basename(input_path)}")

    elif len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            detector.process_image(input_path, output_path)
        else:
            detector.process_video(input_path, output_path)
    else:
        print("Usage: python final_ppe_detector.py [input_file] [output_file]")

if __name__ == "__main__":
    main()
