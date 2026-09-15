"""Minimal end-to-end vHIT acquisition pipeline.

This prototype focuses on the requested camera -> head tracking -> pupil
tracking -> synchronized time-series -> CSV path and keeps both tracker
implementations modular and untouched.

The classifier is intentionally left behind the repository-verified contract gate.
That gate asks us to validate the training signal format before touching the CNN.
"""

import argparse
import csv
import json
import os
import time
import datetime as dt
import numpy as np
import cv2

from head_tracker import HeadTracker
from eye_tracker import EyeTracker
from config import OUTPUT_DIR, DEFAULT_SOURCE, DEFAULT_DURATION


class HITPipeline:
    def __init__(self, source=DEFAULT_SOURCE, source_label='webcam', duration=DEFAULT_DURATION, display=True):
        self.source = source
        self.source_label = source_label
        self.duration = duration
        self.display = display
        self.cap = None
        self.frame_index = 0
        self.start_time = time.time()
        self.records = []
        self.head_tracker = HeadTracker()
        self.eye_tracker = EyeTracker()
        self.face_detected = 0
        self.pupil_detected = 0
        self.total_frames = 0
        self.width = None
        self.height = None
        self.fps = None

    def open_source(self):
        if isinstance(self.source, str):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(int(self.source))
        if not self.cap.isOpened():
            raise RuntimeError(f'Camera/video source unavailable: {self.source}')
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        return self.cap

    def run(self, max_seconds=None):
        self.open_source()
        if max_seconds is None:
            max_seconds = self.duration
        end_time = time.time() + max_seconds

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break

                if frame.shape[:2] != (self.height, self.width):
                    self.height, self.width = frame.shape[:2]

                self.total_frames += 1
                now = time.time()
                timestamp = now - self.start_time
                frame_index = self.frame_index
                self.frame_index += 1

                pose = self.head_tracker.process(frame)
                pupil = self.eye_tracker.process(frame)

                if pose is not None and pose.get('success', False):
                    self.face_detected += 1
                if pupil is not None and pupil.get('success', False):
                    self.pupil_detected += 1

                record = {
                    'timestamp': timestamp,
                    'frame_index': frame_index,
                    'pupil_x': float(pupil.get('pupil_x', np.nan)) if pupil else float('nan'),
                    'pupil_y': float(pupil.get('pupil_y', np.nan)) if pupil else float('nan'),
                    'pupil_radius': float(pupil.get('pupil_radius', np.nan)) if pupil else float('nan'),
                    'head_yaw': float(pose.get('yaw', np.nan)) if pose else float('nan'),
                    'head_pitch': float(pose.get('pitch', np.nan)) if pose else float('nan'),
                    'head_roll': float(pose.get('roll', np.nan)) if pose else float('nan'),
                    'head_tx': float(pose.get('tx', np.nan)) if pose else float('nan'),
                    'head_ty': float(pose.get('ty', np.nan)) if pose else float('nan'),
                    'head_tz': float(pose.get('tz', np.nan)) if pose else float('nan'),
                }
                self.records.append(record)

                if self.display:
                    frame = self.visualize(frame, pose, pupil, frame_index, timestamp)
                    cv2.imshow('HITPipeline', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break

                if time.time() >= end_time:
                    break

            return self.records
        finally:
            self.close()

    def visualize(self, frame, pose, pupil, frame_index, timestamp):
        if pose and pose.get('success'):
            head_status = 'HEAD OK'
            head_color = (0, 255, 0)
        else:
            head_status = 'FACE LOST'
            head_color = (0, 0, 255)

        if pupil and pupil.get('success'):
            pupil_status = 'PUPIL OK'
            pupil_color = (0, 255, 0)
        else:
            pupil_status = 'PUPIL LOST'
            pupil_color = (0, 0, 255)

        cv2.putText(frame, f'Frame {frame_index}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Time {timestamp:.3f}s', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, head_status, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)
        cv2.putText(frame, pupil_status, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pupil_color, 2)

        if pose:
            cv2.putText(frame, f"Yaw {pose.get('yaw', np.nan):.1f}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch {pose.get('pitch', np.nan):.1f}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll {pose.get('roll', np.nan):.1f}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if pupil and pupil.get('success'):
            cv2.putText(frame, f"Pupil {pupil.get('pupil_x', np.nan):.1f}, {pupil.get('pupil_y', np.nan):.1f}", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.head_tracker:
            self.head_tracker.close()
        if self.eye_tracker:
            self.eye_tracker.close()

    def save_records(self, out_path=None):
        if out_path is None:
            out_path = os.path.join(OUTPUT_DIR, f"session_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        keys = ['timestamp', 'frame_index', 'pupil_x', 'pupil_y', 'pupil_radius', 'head_yaw', 'head_pitch', 'head_roll', 'head_tx', 'head_ty', 'head_tz']
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.records:
                writer.writerow(row)
        return out_path

    def save_metadata(self, out_path, fps=None, width=None, height=None):
        meta_path = out_path.replace('.csv', '_metadata.json')
        meta = {
            'fps': fps,
            'frame_count': self.total_frames,
            'video_resolution': f'{width}x{height}' if width and height else 'unknown',
            'timestamp': dt.datetime.utcnow().isoformat(),
            'tracking_success_rates': {
                'face_detection_rate': round(self.face_detected / max(1, self.total_frames), 4) if self.total_frames else 0.0,
                'pupil_detection_rate': round(self.pupil_detected / max(1, self.total_frames), 4) if self.total_frames else 0.0,
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return meta_path

    def run_test_mode(self):
        self.duration = 10.0
        self.run(max_seconds=10.0)
        out_path = self.save_records()
        meta_path = self.save_metadata(out_path, fps=self.fps, width=self.width, height=self.height)
        return {
            'total_frames': self.total_frames,
            'average_fps': self.total_frames / max(1.0, time.time() - self.start_time),
            'face_detection_rate': self.face_detected / max(1, self.total_frames),
            'pupil_detection_rate': self.pupil_detected / max(1, self.total_frames),
            'synchronized_samples': len(self.records),
            'output_csv_path': out_path,
            'output_metadata_path': meta_path,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='HIT vHIT raw acquisition prototype')
    parser.add_argument('source', nargs='?', default='0', help='webcam index or video path')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION, help='seconds to record')
    parser.add_argument('--no-display', action='store_true', help='disable OpenCV visualization window')
    parser.add_argument('--test-mode', action='store_true', help='run for about 10 seconds and print summary')
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source
    if source.isdigit() or source == '0':
        source_int = int(source)
    else:
        source_int = source

    display = not args.no_display
    pipeline = HITPipeline(source=source_int, display=display, duration=args.duration)

    if args.test_mode:
        result = pipeline.run_test_mode()
        print(json.dumps(result, indent=2))
        return

    records = pipeline.run(max_seconds=args.duration)
    out_path = pipeline.save_records()
    pipeline.save_metadata(out_path, fps=pipeline.fps, width=pipeline.width, height=pipeline.height)
    print(f'Recording saved to: {out_path}')
    print(f'Metadata saved to: {out_path.replace(".csv", "_metadata.json")}')


if __name__ == '__main__':
    main()
