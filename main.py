#!/usr/bin/env python3
"""
Прототип: YOLO (люди) + зона столика многоугольником по кликам.

Логика: различаем момент «подошёл» и «сел» по времени пребывания в выбранной зоне.

События в логе:
  - empty — в выбранной зоне никого (эпизод закончился)
  - approach — человек вошёл в зону, но ушёл до `min_seat_time_sec`
  - sat — человек провёл в зоне >= `min_seat_time_sec`

Запуск:
  python main.py --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


PERSON_CLASS = 0

WIN_POLY = "Table zone — LMB points, ENTER close, Z undo, ESC cancel"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Стол пусто / за столом сидят (YOLO + полигон)")
    p.add_argument("--video", type=str, required=True, help="Входное видео")
    p.add_argument("--output", type=str, default="output.mp4", help="Выходное видео")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Веса YOLO")
    p.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Порог детекции человека (сидящих часто ниже — попробуйте 0.25–0.35)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство YOLO: cpu, mps, cuda:0 …",
    )
    p.add_argument(
        "--min_seat_time_sec",
        type=float,
        default=5.0,
        help="Если человек был в зоне >= этого времени — считаем, что сел.",
    )
    p.add_argument(
        "--min_approach_time_sec",
        type=float,
        default=1.0,
        help="Минимальное время (сек) в зоне, чтобы засчитывать детекцию подходом, а не шумом.",
    )
    p.add_argument(
        "--leave_grace_sec",
        type=float,
        default=1.0,
        help="Сколько секунд можно «временно пропасть» из зоны детектора, прежде чем считать эпизод законченным.\n"
        "Полезно при перекрытиях/фликере детекции (например, когда новый человек закрывает сидящего).",
    )
    p.add_argument(
        "--draw_points",
        action="store_true",
        help="Рисовать контрольные точки bbox (низ/центр/бёдра--) и отмечать, попадают ли они в полигон.",
    )
    return p.parse_args()


def select_polygon_zone(frame: np.ndarray) -> np.ndarray:
    """
    Многоугольник по точкам: ЛКМ — вершина, Enter — замкнуть (≥3 точек).
    Z — убрать последнюю точку, Esc — выход из программы.
    Возвращает (N, 2) точек.
    """
    points: List[Tuple[int, int]] = []
    state = {"img": frame.copy()}

    def redraw() -> None:
        img = frame.copy()
        h, w = img.shape[:2]
        cv2.putText(
            img,
            "LMB=vertex  ENTER=finish  Z=undo  ESC=quit",
            (10, min(28, h - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i + 1], (0, 255, 0), 2)
        for p in points:
            cv2.circle(img, p, 5, (0, 255, 255), -1)
        state["img"] = img
        cv2.imshow(WIN_POLY, img)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(WIN_POLY, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_POLY, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Отменено.")
        if key in (ord("z"), ord("Z")):
            if points:
                points.pop()
                redraw()
        if key in (13, 10):
            break

    cv2.destroyAllWindows()

    if len(points) < 3:
        raise SystemExit("Нужно минимум 3 точки многоугольника. Запустите снова.")

    return np.array(points, dtype=np.int32)


def point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """poly: (N,2). Внутри или на границе — True."""
    cnt = poly.astype(np.float32).reshape(-1, 1, 2)
    r = cv2.pointPolygonTest(cnt, (float(px), float(py)), False)
    return r >= 0.0


def person_in_seat_zone(xyxy: np.ndarray, poly: np.ndarray) -> bool:
    """
    Человек «рядом со столом», «за столом», 
    если внутри зоны оказывается хотя бы одна опорная точка bbox:
    низ (ноги), центр, зона бёдер — чтобы сидящий не пропадал, когда ноги вне кадра/полигона.
    """
    x1, y1, x2, y2 = xyxy.astype(float)
    if x2 <= x1 or y2 <= y1:
        return False
    cx = (x1 + x2) / 2.0
    foot_y = y2
    mid_y = (y1 + y2) / 2.0
    hip_y = y1 + 0.62 * (y2 - y1)
    for px, py in ((cx, foot_y), (cx, mid_y), (cx, hip_y)):
        if point_in_polygon(px, py, poly):
            return True
    return False


def draw_zone(frame: np.ndarray, poly: np.ndarray, status: str) -> None:
    """
    status: 'empty' | 'approach' | 'sat'
    """
    if status == "sat":
        color = (0, 0, 255)
    elif status == "approach":
        color = (0, 220, 220)
    else:
        color = (0, 200, 0)
    pts = poly.reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)
    x0 = int(np.min(poly[:, 0]))
    y0 = int(np.min(poly[:, 1]))
    label = status
    cv2.putText(
        frame,
        label,
        (max(x0, 8), max(y0 - 8, 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def run() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Файл не найден: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise SystemExit("Не удалось прочитать первый кадр.")

    print("Загрузка YOLO…", flush=True)
    model = YOLO(args.model)
    pred_kw: dict = dict(classes=[PERSON_CLASS], conf=args.conf, verbose=False)
    if args.device:
        pred_kw["device"] = args.device

    print(
        "Прогрев модели (первый кадр, подождите)…",
        flush=True,
    )
    model.predict(first, **pred_kw)
    print("Модель готова.", flush=True)

    print(
        "Обведите зону места за столом многоугольником.",
        flush=True,
    )
    poly = select_polygon_zone(first)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(args.output)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    prev_effective_in_zone: Optional[bool] = None
    episode_start_time: Optional[float] = None
    approach_candidate_time: Optional[float] = None
    approach_candidate_frame_idx: Optional[int] = None
    approach_logged: bool = False
    sat_reached: bool = False
    absence_start_time: Optional[float] = None
    rows: List[dict] = []

    def append_event(frame_idx: int, time_sec: float, event: str) -> None:
        rows.append(
            {
                "frame": frame_idx,
                "time_sec": round(time_sec, 4),
                "event": event,
            }
        )

    def finalize_episode(end_frame_idx: int, end_time_sec: float, append_empty_event: bool) -> None:
        nonlocal episode_start_time, approach_candidate_time, approach_candidate_frame_idx, approach_logged, sat_reached, absence_start_time
        if episode_start_time is None:
            return

        min_seat_time_sec = float(args.min_seat_time_sec)
        min_approach_time_sec = float(args.min_approach_time_sec)
        duration = end_time_sec - episode_start_time

        if (not approach_logged) and approach_candidate_time is not None and duration >= min_approach_time_sec:
            fi = approach_candidate_frame_idx if approach_candidate_frame_idx is not None else end_frame_idx
            append_event(fi, approach_candidate_time, "approach")
            approach_logged = True

        if (not sat_reached) and duration >= min_seat_time_sec:
            append_event(end_frame_idx, end_time_sec, "sat")
            sat_reached = True

        if append_empty_event:
            append_event(end_frame_idx, end_time_sec, "empty")

        episode_start_time = None
        approach_candidate_time = None
        approach_candidate_frame_idx = None
        approach_logged = False
        sat_reached = False
        if append_empty_event:
            absence_start_time = end_time_sec

    def process_frame(
        frame: np.ndarray,
        frame_idx: int,
        time_sec: float,
    ) -> bool:
        nonlocal prev_effective_in_zone, episode_start_time, approach_candidate_time, approach_candidate_frame_idx, approach_logged, sat_reached, absence_start_time
        results = model.predict(frame, **pred_kw)
        boxes = results[0].boxes
        raw_in_zone = False
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            for b in xyxy:
                if args.draw_points:
                    x1, y1, x2, y2 = b.astype(float)
                    cx = (x1 + x2) / 2.0
                    foot_y = y2
                    mid_y = (y1 + y2) / 2.0
                    hip_y = y1 + 0.62 * (y2 - y1)
                    for px, py in (
                        (cx, foot_y),
                        (cx, mid_y),
                        (cx, hip_y),
                    ):
                        inside = point_in_polygon(px, py, poly)
                        color = (0, 255, 0) if inside else (0, 0, 255)
                        cv2.circle(frame, (int(px), int(py)), 4, color, -1)

                if person_in_seat_zone(b, poly):
                    raw_in_zone = True

        # Учитываем небольшую «потерю» детекции, чтобы эпизод не рвался из-за 1-2 кадров
        if raw_in_zone:
            absence_start_time = None
        else:
            if absence_start_time is None:
                absence_start_time = time_sec

        leave_grace_sec = float(args.leave_grace_sec)
        effective_in_zone = raw_in_zone or (
            absence_start_time is not None and (time_sec - absence_start_time) < leave_grace_sec
        )

        min_seat_time_sec = float(args.min_seat_time_sec)
        min_approach_time_sec = float(args.min_approach_time_sec)

        if effective_in_zone and episode_start_time is not None:
            dur = time_sec - episode_start_time
            if dur >= min_seat_time_sec:
                status = "sat"
            elif dur >= min_approach_time_sec:
                status = "approach"
            else:
                status = "empty"
        else:
            status = "empty"

        if prev_effective_in_zone is None:
            prev_effective_in_zone = effective_in_zone
            if effective_in_zone:
                episode_start_time = time_sec
                approach_candidate_time = time_sec
                approach_candidate_frame_idx = frame_idx
                approach_logged = False
                sat_reached = False
        else:
            if (not prev_effective_in_zone) and effective_in_zone:
                episode_start_time = time_sec
                approach_candidate_time = time_sec
                approach_candidate_frame_idx = frame_idx
                approach_logged = False
                sat_reached = False

            elif prev_effective_in_zone and effective_in_zone:
                if episode_start_time is not None:
                    duration = time_sec - episode_start_time
                    if (not approach_logged) and approach_candidate_time is not None and duration >= min_approach_time_sec:
                        fi = approach_candidate_frame_idx if approach_candidate_frame_idx is not None else frame_idx
                        append_event(fi, approach_candidate_time, "approach")
                        approach_logged = True
                    if (not sat_reached) and duration >= min_seat_time_sec:
                        append_event(frame_idx, time_sec, "sat")
                        sat_reached = True

            elif prev_effective_in_zone and (not effective_in_zone):
                finalize_episode(frame_idx, time_sec, append_empty_event=True)

            prev_effective_in_zone = effective_in_zone

        draw_zone(frame, poly, status)
        cv2.putText(
            frame,
            f"t={time_sec:.2f}s  f={frame_idx}/{total_frames}",
            (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        return effective_in_zone

    print(f"Запись: {out_path} (~{total_frames} кадров)…", flush=True)
    process_frame(first, 0, 0.0)

    frame_idx = 1
    progress_every = max(1, total_frames // 20) if total_frames > 0 else 30
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        time_sec = frame_idx / fps if fps > 0 else 0.0
        process_frame(frame, frame_idx, time_sec)
        if frame_idx % progress_every == 0:
            print(f"  кадров: {frame_idx}/{total_frames}", flush=True)
        frame_idx += 1

    if prev_effective_in_zone:
        last_frame_idx = max(0, frame_idx - 1)
        last_time_sec = (last_frame_idx / fps) if fps > 0 else 0.0
        finalize_episode(last_frame_idx, last_time_sec, append_empty_event=False)
        prev_effective_in_zone = False

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    events_csv = Path("events.csv")
    df.to_csv(events_csv, index=False)

    # Статистика:
    #  - empty -> approach
    #  - empty -> sat
    delays_to_approach: List[float] = []
    delays_to_sat: List[float] = []
    last_empty_time: Optional[float] = None
    approach_recorded_for_episode = False
    sat_recorded_for_episode = False
    if not df.empty:
        df_sorted = df.sort_values("time_sec")
        for _, row in df_sorted.iterrows():
            ev = str(row["event"])
            t = float(row["time_sec"])
            if ev == "empty":
                last_empty_time = t
                approach_recorded_for_episode = False
                sat_recorded_for_episode = False
            elif ev == "approach" and last_empty_time is not None and (not approach_recorded_for_episode):
                delays_to_approach.append(t - last_empty_time)
                approach_recorded_for_episode = True
            elif ev == "sat" and last_empty_time is not None and (not sat_recorded_for_episode):
                delays_to_sat.append(t - last_empty_time)
                sat_recorded_for_episode = True

    mean_to_approach = float(np.mean(delays_to_approach)) if delays_to_approach else float("nan")
    mean_to_sat = float(np.mean(delays_to_sat)) if delays_to_sat else float("nan")

    report_lines = [
        "Отчёт",
        f"Видео: {video_path}",
        f"Кадров обработано: {frame_idx}",
        f"Событий в логе: {len(df)}",
        f"Пар «empty -> approach»: {len(delays_to_approach)}",
        f"Среднее время empty -> approach (сек, min_seat_time_sec={args.min_seat_time_sec}): "
        f"{mean_to_approach:.3f}" if not np.isnan(mean_to_approach) else "Среднее: нет пар empty -> approach",
        f"Пар «empty -> sat»: {len(delays_to_sat)}",
        f"Среднее время empty -> sat (сек, min_seat_time_sec={args.min_seat_time_sec}): "
        f"{mean_to_sat:.3f}" if not np.isnan(mean_to_sat) else "Среднее: нет пар empty -> sat",
        f"События: {events_csv.resolve()}",
        f"Видео: {out_path.resolve()}",
    ]
    report_text = "\n".join(report_lines)
    print(report_text)
    Path("report.txt").write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
