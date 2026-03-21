import argparse
import base64
import html
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.algorithm_visualizer import AlgorithmVisualizer
from backend.core.batch_processor import _resolve_diameter_method
from src.analysis.feature_extractor import FeatureExtractor

DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "xr_feature_visual_reports"


def read_grayscale_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img

    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def prepare_visualization_image(img_gray: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img_gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img_gray

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_base64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def build_waviness_overlay(extractor: FeatureExtractor, img_gray: np.ndarray):
    roi = extractor.extract_roi(img_gray)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    _, thresh = extractor.calculate_density(processed)

    diameter_method = extractor.diameter_method
    if diameter_method == "enhanced":
        _, skel = extractor.calculate_diameter_enhanced(thresh)
    else:
        _, skel = extractor.calculate_diameter(thresh)

    waviness = extractor.calculate_waviness(skel)
    overlay = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    overlay[skel > 0] = (0, 255, 255)

    ratio = waviness.get("waviness_ratio")
    height_nm = waviness.get("waviness_height_nm")
    wavelength_nm = waviness.get("waviness_wavelength_nm")
    branches = waviness.get("waviness_branches")

    text_lines = [
        f"waviness ratio = {ratio:.4f}" if ratio is not None else "waviness ratio = N/A",
        f"height = {height_nm:.2f} nm" if height_nm is not None else "height = N/A",
        f"wavelength = {wavelength_nm:.2f} nm" if wavelength_nm is not None else "wavelength = N/A",
        f"branches = {branches}",
    ]
    y = 28
    for line in text_lines:
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28

    description = (
        "黄色骨架为用于 waviness 的中心线。算法先对骨架做主轴投影，再统计横向偏移的峰谷，"
        "以波高/波长得到 waviness ratio。"
    )
    return {
        "name": "波浪度 (Waviness)",
        "image": image_to_base64(overlay),
        "description": description,
    }


def collect_feature_timings(image_path: str, magnification, source: str, requested_method: str):
    img = read_grayscale_image(image_path)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    effective_method, fallback_reason = _resolve_diameter_method(source, magnification, requested_method)
    extractor = FeatureExtractor(
        magnification=int(magnification) if magnification else None,
        diameter_method=effective_method,
    )

    steps = []

    def on_progress(step_name: str, elapsed_s: float, payload: dict):
        steps.append({
            "step": step_name,
            "elapsed_s": elapsed_s,
            "payload": payload or {},
        })

    result = extractor.extract_all(img, progress_callback=on_progress)
    result["effective_diameter_method"] = effective_method
    result["fallback_reason"] = fallback_reason
    return img, result, steps


def query_images(limit: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, sample_id, file_path, magnification, source, processed
        FROM images
        WHERE source = 'XR'
          AND COALESCE(is_deleted, 0) = 0
          AND COALESCE(magnification, 0) >= 20000
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(entries, output_path: Path):
    cards = []
    for entry in entries:
        features = entry["features"]
        timing_rows = "".join(
            f"<tr><td>{html.escape(step['step'])}</td><td>{step['elapsed_s']:.3f}s</td><td>{html.escape(str(step['payload']))}</td></tr>"
            for step in entry["timings"]
        )
        step_blocks = "".join(
            f"""
            <div class="step-card">
              <h4>{html.escape(step['name'])}</h4>
              <img src="data:image/jpeg;base64,{step['image']}" alt="{html.escape(step['name'])}">
              <p>{html.escape(step['description'])}</p>
            </div>
            """
            for step in entry["visual_steps"]
        )
        cards.append(
            f"""
            <section class="sample-card">
              <div class="sample-header">
                <div>
                  <h2>{html.escape(entry['sample_label'])}</h2>
                  <div class="meta">image_id={entry['image_id']} | mag={entry['magnification']} | processed={entry['processed']}</div>
                  <div class="meta path">{html.escape(entry['file_path'])}</div>
                </div>
                <div class="method">
                  <div>requested method: {html.escape(entry['requested_method'])}</div>
                  <div>effective method: {html.escape(features['effective_diameter_method'])}</div>
                  <div>fallback: {html.escape(features['fallback_reason'] or 'none')}</div>
                </div>
              </div>

              <div class="feature-grid">
                <div><span>Density</span><strong>{fmt(features['density'], 2)} %</strong></div>
                <div><span>Diameter</span><strong>{fmt(features['diameter'], 2)} nm</strong></div>
                <div><span>Alignment</span><strong>{fmt(features['alignment'], 4)}</strong></div>
                <div><span>Curvature</span><strong>{fmt(features['curvature'])}</strong></div>
                <div><span>Waviness</span><strong>{fmt(features['waviness_ratio'], 4)}</strong></div>
                <div><span>Wave H / λ</span><strong>{fmt(features['waviness_height_nm'], 2)} / {fmt(features['waviness_wavelength_nm'], 2)} nm</strong></div>
              </div>

              <div class="formula-box">
                <div><strong>五个特征的当前解释</strong></div>
                <div>Density: 二值前景占比</div>
                <div>Diameter: 骨架/距离变换估计的代表管径</div>
                <div>Alignment: Herman 取向因子</div>
                <div>Curvature: 骨架三点法曲率分类</div>
                <div>Waviness: 骨架中心线波高 / 波长</div>
              </div>

              <table class="timing-table">
                <thead><tr><th>Step</th><th>Elapsed</th><th>Payload</th></tr></thead>
                <tbody>{timing_rows}</tbody>
              </table>

              <div class="steps-grid">{step_blocks}</div>
            </section>
            """
        )

    report_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8">
      <title>XR 特征可视化报告</title>
      <style>
        body {{
          font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
          margin: 0;
          background: #f3f6fb;
          color: #1f2937;
        }}
        .wrap {{
          max-width: 1680px;
          margin: 0 auto;
          padding: 24px;
        }}
        .hero {{
          background: linear-gradient(135deg, #0f172a, #1d4ed8);
          color: white;
          border-radius: 18px;
          padding: 24px 28px;
          margin-bottom: 24px;
        }}
        .hero h1 {{
          margin: 0 0 10px;
          font-size: 30px;
        }}
        .hero p {{
          margin: 4px 0;
          opacity: 0.92;
        }}
        .sample-card {{
          background: white;
          border-radius: 18px;
          padding: 22px;
          margin-bottom: 28px;
          box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
        }}
        .sample-header {{
          display: flex;
          justify-content: space-between;
          gap: 20px;
          margin-bottom: 18px;
        }}
        .sample-header h2 {{
          margin: 0 0 8px;
        }}
        .meta {{
          font-size: 13px;
          color: #64748b;
          margin-bottom: 4px;
        }}
        .path {{
          word-break: break-all;
        }}
        .method {{
          min-width: 260px;
          font-size: 13px;
          background: #eff6ff;
          color: #1d4ed8;
          border-radius: 14px;
          padding: 12px 14px;
        }}
        .feature-grid {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin-bottom: 18px;
        }}
        .feature-grid > div {{
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          padding: 12px 14px;
        }}
        .feature-grid span {{
          display: block;
          font-size: 12px;
          color: #64748b;
          margin-bottom: 6px;
        }}
        .feature-grid strong {{
          font-size: 20px;
        }}
        .formula-box {{
          background: #fff7ed;
          border: 1px solid #fed7aa;
          border-radius: 12px;
          padding: 12px 14px;
          margin-bottom: 18px;
          line-height: 1.7;
        }}
        .timing-table {{
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 20px;
          font-size: 13px;
        }}
        .timing-table th, .timing-table td {{
          padding: 10px 12px;
          border-bottom: 1px solid #e5e7eb;
          text-align: left;
          vertical-align: top;
        }}
        .timing-table th {{
          background: #f8fafc;
        }}
        .steps-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }}
        .step-card {{
          background: #f8fafc;
          border-radius: 14px;
          padding: 14px;
          border: 1px solid #e5e7eb;
        }}
        .step-card h4 {{
          margin: 0 0 10px;
          font-size: 16px;
        }}
        .step-card img {{
          width: 100%;
          border-radius: 10px;
          display: block;
          margin-bottom: 10px;
        }}
        .step-card p {{
          margin: 0;
          font-size: 13px;
          line-height: 1.65;
          color: #475569;
        }}
        @media (max-width: 1100px) {{
          .feature-grid, .steps-grid {{
            grid-template-columns: 1fr;
          }}
          .sample-header {{
            flex-direction: column;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="hero">
          <h1>XR 图像五特征可视化报告</h1>
          <p>生成时间：{html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
          <p>样本数：{len(entries)} 张</p>
          <p>说明：特征值来自真实 FeatureExtractor；步骤图用于展示“每一步怎么算出来的”。</p>
        </div>
        {''.join(cards)}
      </div>
    </body>
    </html>
    """

    output_path.write_text(report_html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate XR feature visual report")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--diameter-method", choices=["standard", "enhanced"], default="enhanced")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_ROOT / f"xr_feature_visual_report_{timestamp}.html"

    rows = query_images(args.limit)
    if not rows:
        raise SystemExit("No eligible XR images found")

    entries = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] analyzing image_id={row['id']} sample={row['sample_id']}")
        img_gray, features, timings = collect_feature_timings(
            image_path=row["file_path"],
            magnification=row["magnification"],
            source=row["source"],
            requested_method=args.diameter_method,
        )

        visualizer = AlgorithmVisualizer(magnification=row["magnification"] or 50000)
        visual_img = prepare_visualization_image(img_gray, max_side=1280)
        visual_steps = visualizer.visualize_extraction(visual_img)

        # Add an explicit waviness visualization step so the fifth feature is visible.
        waviness_extractor = FeatureExtractor(
            magnification=int(row["magnification"]) if row["magnification"] else None,
            diameter_method=features["effective_diameter_method"],
        )
        visual_steps.append(build_waviness_overlay(waviness_extractor, visual_img))

        entries.append({
            "image_id": row["id"],
            "sample_label": row["sample_id"] or f"ID {row['id']}",
            "magnification": row["magnification"],
            "processed": row["processed"],
            "file_path": row["file_path"],
            "requested_method": args.diameter_method,
            "features": features,
            "timings": timings,
            "visual_steps": visual_steps,
        })

    render_report(entries, output_path)
    print(f"REPORT_PATH={output_path}")


if __name__ == "__main__":
    main()
