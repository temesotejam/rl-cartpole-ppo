from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static Cart-Pole PPO training dashboard.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-url", default="")
    return parser.parse_args()


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def copy_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    with (source / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    copy_dir(source / "videos", output / "videos")
    copy_dir(source / "plots", output / "plots")
    shutil.copy2(source / "metrics.csv", output / "metrics.csv")
    shutil.copy2(source / "metadata.json", output / "metadata.json")
    shutil.copy2(source / "summary.md", output / "summary.md")

    best = max(rows[1:] if len(rows) > 1 else rows, key=lambda row: number(row, "mean_return"))
    final = rows[-1]
    videos = sorted((output / "videos").glob("*.mp4"))
    plot_names = [
        ("learning_curve.png", "Mean return"),
        ("angle_error.png", "Pole angle"),
        ("cart_position.png", "Cart position"),
        ("success_rate.png", "10 s completion rate"),
        ("motor_force.png", "Motor force"),
    ]

    def metric_card(label: str, value: str, note: str = "") -> str:
        return f'''<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>'''

    cards = "".join(
        [
            metric_card("Best return", f"{number(best, 'mean_return'):.1f}", best["stage"]),
            metric_card("Final success", f"{100 * number(final, 'success_rate'):.1f}%", "10 s completion"),
            metric_card("Final RMS angle", f"{number(final, 'rms_angle_deg'):.2f}°", "true state"),
            metric_card("Final RMS cart x", f"{number(final, 'rms_cart_position_m'):.3f} m", "center = 0"),
            metric_card("Final RMS force", f"{number(final, 'rms_force_n'):.2f} N", "actual motor force"),
        ]
    )

    table_rows = "".join(
        f'''<tr><td>{html.escape(row['stage'])}</td><td>{int(float(row['timesteps'])):,}</td><td>{number(row, 'mean_return'):.1f}</td><td>{number(row, 'mean_survival_time_s'):.2f}s</td><td>{100 * number(row, 'success_rate'):.1f}%</td><td>{number(row, 'rms_angle_deg'):.2f}°</td><td>{number(row, 'rms_cart_position_m'):.3f}m</td><td>{number(row, 'rms_force_n'):.2f}N</td></tr>'''
        for row in rows
    )

    video_buttons = "".join(
        f'''<button class="video-tab{' active' if i == 0 else ''}" data-src="videos/{html.escape(video.name)}">{html.escape(video.stem.replace('_', ' '))}</button>'''
        for i, video in enumerate(videos)
    )
    first_video = f"videos/{videos[0].name}" if videos else ""

    plots = "".join(
        f'''<figure><img src="plots/{name}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'''
        for name, label in plot_names
        if (output / "plots" / name).exists()
    )

    sensor = metadata["sensor_noise"]
    physics = metadata["physics"]
    run_link = (
        f'<a class="button" href="{html.escape(args.run_url)}">Open source GitHub Actions run</a>'
        if args.run_url
        else ""
    )

    document = f'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rl-cartpole-ppo · Training Dashboard</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1018; --panel:#131b27; --line:#263447; --text:#edf3fb; --muted:#9caec2; --accent:#69a8ff; --good:#6ee7a8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(180deg,#0a0f17,#0d1420 45%,#0b1018); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:min(1180px,92vw); margin:0 auto; padding:48px 0 80px; }}
h1 {{ font-size:clamp(2rem,5vw,4.4rem); margin:0; letter-spacing:-.05em; }}
h2 {{ margin:48px 0 18px; font-size:1.55rem; }}
p {{ color:var(--muted); line-height:1.75; }}
.hero {{ padding:38px; border:1px solid var(--line); border-radius:24px; background:rgba(19,27,39,.82); box-shadow:0 24px 80px rgba(0,0,0,.25); }}
.eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.8rem; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-top:28px; }}
.metric {{ padding:18px; border:1px solid var(--line); border-radius:16px; background:#0f1723; }}
.metric .label,.metric .note {{ color:var(--muted); font-size:.82rem; }}
.metric .value {{ font-size:1.65rem; font-weight:750; margin:5px 0; }}
.video-shell {{ border:1px solid var(--line); background:#05080d; border-radius:20px; overflow:hidden; }}
video {{ display:block; width:100%; max-height:650px; background:#05080d; }}
.tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:14px 0; }}
.video-tab,.button {{ color:var(--text); background:#182435; border:1px solid #31435b; border-radius:999px; padding:10px 14px; cursor:pointer; text-decoration:none; font:inherit; }}
.video-tab.active {{ background:var(--accent); color:#06101f; border-color:var(--accent); font-weight:700; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
figure {{ margin:0; padding:12px; background:var(--panel); border:1px solid var(--line); border-radius:18px; }}
figure img {{ width:100%; border-radius:10px; background:white; }}
figcaption {{ color:var(--muted); padding:9px 4px 2px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:16px; }}
table {{ width:100%; border-collapse:collapse; min-width:800px; background:var(--panel); }}
th,td {{ padding:13px 14px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ color:var(--muted); font-size:.8rem; }}
th:first-child,td:first-child {{ text-align:left; }}
.details {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
.detail {{ padding:18px; border:1px solid var(--line); border-radius:16px; background:var(--panel); }}
code {{ color:#b7d5ff; }}
footer {{ color:var(--muted); margin-top:55px; font-size:.9rem; }}
</style>
</head>
<body><main>
<section class="hero">
<div class="eyebrow">Reinforcement Learning · Continuous Cart-Pole</div>
<h1>rl-cartpole-ppo</h1>
<p>連続的なモータ力で台車を動かし、センサノイズとモータ応答遅れがある状態でPPOが倒立振子を安定化する過程を可視化します。AIが見るのはノイズ付き観測だけで、下の評価値はシミュレータ真値から計算しています。</p>
<div class="metrics">{cards}</div>
<p>{run_link}</p>
</section>

<h2>学習の進み方を動画で比較</h2>
<p>同じ評価seedを使った random / 25% / 50% / 75% / 100% を切り替えられます。台車位置、振り子角度、実際のモータ力も動画左上に表示します。</p>
<div class="tabs">{video_buttons}</div>
<div class="video-shell"><video id="player" src="{html.escape(first_video)}" controls loop playsinline></video></div>

<h2>評価値</h2>
<div class="table-wrap"><table><thead><tr><th>Stage</th><th>Steps</th><th>Mean return</th><th>Survival</th><th>Success</th><th>RMS angle</th><th>RMS cart x</th><th>RMS force</th></tr></thead><tbody>{table_rows}</tbody></table></div>

<h2>学習曲線</h2>
<div class="grid">{plots}</div>

<h2>実験条件</h2>
<div class="details">
<div class="detail"><strong>Physics</strong><p>Cart <code>{physics['cart_mass_kg']} kg</code><br>Pole <code>{physics['pole_mass_kg']} kg</code><br>COM length <code>{physics['pole_com_length_m']} m</code><br>Force <code>±{physics['max_force_n']} N</code><br>Control <code>{1/physics['dt_s']:.0f} Hz</code><br>Motor τ <code>{physics['motor_time_constant_s']*1000:.0f} ms</code><br>Track <code>±{physics['track_limit_m']} m</code></p></div>
<div class="detail"><strong>Sensor model</strong><p>Position noise <code>{sensor['position_noise_std_m']*1000:.1f} mm</code><br>Position bias σ <code>{sensor['position_bias_std_m']*1000:.1f} mm</code><br>Velocity noise <code>{sensor['velocity_noise_std_mps']:.3f} m/s</code><br>Angle noise <code>{sensor['angle_noise_std_deg']}°</code><br>Angle bias σ <code>{sensor['angle_bias_std_deg']}°</code><br>Gyro noise <code>{sensor['gyro_noise_std_dps']}°/s</code></p></div>
<div class="detail"><strong>Training</strong><p>Algorithm <code>{html.escape(metadata['algorithm'])}</code><br>Preset <code>{html.escape(metadata['preset'])}</code><br>Seed <code>{metadata['seed']}</code><br>Requested steps <code>{metadata['requested_total_timesteps']:,}</code><br>Actual steps <code>{metadata['actual_final_timesteps']:,}</code><br>Environment <code>{html.escape(metadata['environment'])}</code></p></div>
</div>

<footer>Generated automatically from the latest successful <code>Train RL Agent</code> GitHub Actions artifact.</footer>
</main>
<script>
const player=document.getElementById('player');
document.querySelectorAll('.video-tab').forEach(btn=>btn.addEventListener('click',()=>{{
  document.querySelectorAll('.video-tab').forEach(x=>x.classList.remove('active'));
  btn.classList.add('active'); player.src=btn.dataset.src; player.play().catch(()=>{{}});
}}));
</script>
</body></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
