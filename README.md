# rl-cartpole-ppo

[![CI](https://github.com/temesotejam/rl-cartpole-ppo/actions/workflows/ci.yml/badge.svg)](https://github.com/temesotejam/rl-cartpole-ppo/actions/workflows/ci.yml)
[![Pages](https://github.com/temesotejam/rl-cartpole-ppo/actions/workflows/pages.yml/badge.svg)](https://github.com/temesotejam/rl-cartpole-ppo/actions/workflows/pages.yml)

**連続的なモータ力で台車を左右へ動かし、PPO が倒立振子を立て続ける制御を学習する物理シミュレーション実験です。**

GitHub Pages: **https://temesotejam.github.io/rl-cartpole-ppo/**

このリポジトリは `rl-<environment>-<algorithm>` シリーズの第2号です。

- 第1号: `rl-pendulum-ppo` — 回転軸へ直接トルクを与える1自由度振り子
- 第2号: `rl-cartpole-ppo` — 台車を動かして振り子を間接的に制御

今回は Gymnasium の標準 `CartPole-v1` をそのまま使用せず、**連続力入力・台車摩擦・モータ応答遅れ・民生センサ相当の観測ノイズ**を持つ独自環境を実装しています。

---

## 何を見る実験か

AI には倒立振子の制御則を与えません。

```text
センサ観測
    ↓
PPO Agent
    ↓
-1.0 ～ +1.0
    ↓
-10 N ～ +10 N のモータ指令
    ↓
台車が加速
    ↓
振り子が動く
    ↓
新しいセンサ観測 + reward
    └────────────→ PPO
```

最初はランダムな力しか出せず、振り子をすぐ倒したり、台車をレール端へ追いやったりします。

学習が進むにつれて、

1. 振り子が倒れそうな方向を読む
2. 台車を適切な方向へ加速する
3. 角速度を止める
4. 台車自身も中央付近へ戻す
5. 必要以上に大きなモータ力を使わない

という複数の条件を同時に満たす方策を学習します。

---

# 学習進行を動画で確認

1回の学習で同一評価条件に対して次の5段階を動画化します。

```text
00_random.mp4
01_25_percent.mp4
02_50_percent.mp4
03_75_percent.mp4
04_100_percent.mp4
```

GitHub Pagesではこの5本をブラウザ上で切り替えて再生できます。

動画左上には、

```text
x       台車位置 [m]
theta   真の振り子角度 [deg]
force   実際に発生しているモータ力 [N]
t       時間 [s]
```

を表示します。

PPO が受け取るのはセンサノイズ付き観測ですが、**動画表示と評価値はシミュレータ内部の真値**を使います。

---

# 物理モデル

## 構成

```text
                    Pole mass
                       ●
                      /
                     /
                    O  pivot
             ┌──────────────┐
             │     Cart     │
─────────────┴──────────────┴─────────────
                  ←  x  →

             ← F             F →
```

状態は、

\[
\mathbf{x}=
\begin{bmatrix}
x \\
\dot{x} \\
\theta \\
\dot{\theta}
\end{bmatrix}
\]

です。

- `x` : 台車位置
- `x_dot` : 台車速度
- `theta` : 振り子角度。0 rad が真上
- `theta_dot` : 振り子角速度

## 初期物理パラメータ

| 項目 | 値 |
|---|---:|
| 台車質量 | 1.0 kg |
| 振り子質量 | 0.1 kg |
| 支点→振り子重心 | 0.5 m |
| 重力加速度 | 9.81 m/s² |
| 最大モータ力 | ±10 N |
| 台車粘性摩擦係数 | 0.10 N/(m/s) |
| モータ時定数 | 0.05 s |
| 制御周期 | 0.02 s = 50 Hz |
| レール範囲 | ±2.4 m |
| 許容角度 | ±30° |
| 1 episode | 最大10 s |

設定は `configs/*.yaml` に集約しています。

---

# 標準 CartPole-v1 との違い

Gymnasium の標準 `CartPole-v1` は教材として非常に優れていますが、本実験では制御系としてもう少し現実寄りにしています。

## 標準環境

```text
Action = 0 : 左へ固定力
Action = 1 : 右へ固定力
```

## この環境

```text
Action = -1.0 ... +1.0
        ↓
Motor command
        ↓
-10 N ... +10 N
```

つまり「左右どちらへ動かすか」だけではなく、**どれくらいの力を出すか**までPPOが決めます。

さらに、モータ指令は瞬時に実際の力になりません。

---

# モータ応答

モータ・ドライバを単純な一次遅れとして模擬します。

\[
F_{actual}(t+\Delta t)
=
F_{actual}(t)
+\alpha\left(F_{cmd}-F_{actual}(t)\right)
\]

ここで、

```text
motor_time_constant_s = 0.05 s
```

です。

したがって、PPO が突然 +10 N を要求しても、その瞬間に +10 N が出るわけではありません。

この遅れは小さいですが、理想シミュレーションより制御問題を少し現実に近づけます。

---

# 台車摩擦

台車には速度に比例する粘性摩擦を入れています。

\[
F_f=-b\dot{x}
\]

初期値は、

```text
b = 0.10 N/(m/s)
```

です。

---

# AI が見るセンサ値

PPO にシミュレータの真値を直接渡しません。

真の状態から民生用センサを模擬した観測値を作ります。

PPO への入力は、

```text
x_measured
x_dot_measured
cos(theta_measured)
sin(theta_measured)
theta_dot_measured
```

です。

角度を `theta` のまま渡さず `cos` と `sin` にすることで、角度の ±π 境界を連続的に扱えます。

---

# センサノイズ

## 台車位置

```text
白色ノイズ σ = 1 mm
episode bias σ = 2 mm
```

## 台車速度

```text
白色ノイズ σ = 0.01 m/s
episode bias σ = 0.01 m/s
```

## 振り子角度

```text
白色ノイズ σ = 0.25°
episode bias σ = 1.0°
```

## ジャイロ

```text
白色ノイズ σ = 0.10°/s
episode bias σ = 0.30°/s
```

これは特定の1製品を厳密に再現するものではなく、**民生用MEMS IMU + 位置センサを意識した単純化モデル**です。

ノイズには2種類あります。

```text
白色ノイズ
  毎サンプル変化

bias
  episode開始時に決まり、そのepisode中は固定
```

したがってPPOは、単純な瞬間ノイズだけでなく、少しずれたセンサを使って制御することになります。

---

# 真値と観測値の分離

重要な設計です。

```text
              Physics simulator
                    │
          ┌─────────┴─────────┐
          │                   │
          ↓                   ↓
       true state        Sensor model
          │                   │
          │                   ↓
          │             noisy observation
          │                   │
          │                   ↓
          │                PPO Agent
          │                   │
          │                   ↓
          │                action
          │                   │
          └────→ evaluation ←─┘
```

PPOは真値を知りません。

評価だけが真値を使います。

これにより、センサがたまたま良い値を出しただけで制御性能が高く見えることを防ぎます。

---

# Reward

各stepで基本報酬を `+1` とし、状態が悪くなるほど減点します。

主に、

- 振り子角度
- 台車位置
- 振り子角速度
- 台車速度
- モータ力

をペナルティにします。

概念的には、

\[
r = 1
-w_\theta\theta^2
-w_xx^2
-w_{\dot\theta}\dot\theta^2
-w_{\dot x}\dot x^2
-w_FF^2
\]

です。

特に角度を最も重くしています。

ただし、角度だけを良くすればよいわけではありません。

```text
振り子を立てるために
台車を一方向へ走らせ続ける
        ↓
レール端へ到達
        ↓
失敗
```

となるため、PPOは台車位置も考える必要があります。

---

# Episode終了条件

次のどちらかで失敗終了します。

```text
|cart position| > 2.4 m

または

|pole angle| > 30°
```

10秒間これを避ければepisode完走です。

---

# 評価指標

| 指標 | 意味 |
|---|---|
| Mean return | rewardの総合結果。大きいほど良い |
| Survival time | 平均して何秒倒れずにいられたか |
| Success rate | 10秒完走したepisodeの割合 |
| RMS pole angle | 振り子がどれだけ真上付近にいたか |
| RMS cart position | 台車が中央付近にいたか |
| RMS cart velocity | 不要に走り回っていないか |
| RMS motor force | 制御入力の大きさ |
| Max abs angle | 大きく傾いたか |

最終的には、

```text
Success rate → 100%
RMS angle → 小さい
RMS cart position → 小さい
RMS force → 必要以上に大きくない
```

状態を目指します。

---

# PPO

[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) のPPOを使用します。

初期設定は、

```text
MlpPolicy
4 parallel environments
CPU
n_steps = 1024
batch_size = 64
gamma = 0.99
gae_lambda = 0.95
learning_rate = 3e-4
use_sde = true
```

です。

この問題は小さなMLPで扱えるため、GitHub ActionsのCPU runnerだけで学習します。

---

# Training preset

## quick

```text
20,000 requested timesteps
3 evaluation episodes
```

CIと動作確認用です。

## normal

```text
100,000 requested timesteps
10 evaluation episodes
```

通常実験です。

## long

```text
300,000 requested timesteps
20 evaluation episodes
```

学習量を増やしたい場合に使用します。

PPOは複数環境からrollout単位でデータを集めるため、実際の最終timestepsは指定値より少し多くなる場合があります。

---

# GitHub Actions

`Actions` → `Train RL Agent` → `Run workflow`

から、

```text
preset
  quick
  normal
  long

seed
  42
  123
  ...
```

を指定できます。

処理は、

```text
GitHub-hosted Ubuntu runner
        ↓
Python + CPU PyTorch
        ↓
Continuous Cart-Pole × 4 environments
        ↓
PPO training
        ↓
random / 25 / 50 / 75 / 100% evaluation
        ↓
videos
models
plots
metrics.csv
metadata.json
summary.md
        ↓
Actions Artifact
```

となります。

---

# 生成物

```text
results/
├── videos/
│   ├── 00_random.mp4
│   ├── 01_25_percent.mp4
│   ├── 02_50_percent.mp4
│   ├── 03_75_percent.mp4
│   └── 04_100_percent.mp4
│
├── models/
│   ├── 25_percent.zip
│   ├── 50_percent.zip
│   ├── 75_percent.zip
│   └── 100_percent.zip
│
├── plots/
│   ├── learning_curve.png
│   ├── angle_error.png
│   ├── cart_position.png
│   ├── success_rate.png
│   └── motor_force.png
│
├── metrics.csv
├── metadata.json
└── summary.md
```

---

# GitHub Pages

学習が成功すると `Publish Training Dashboard` workflow がそのArtifactを取得し、最新の結果をGitHub Pagesへ自動公開します。

**https://temesotejam.github.io/rl-cartpole-ppo/**

表示内容は、

- 5段階動画
- best / final評価値
- 全checkpoint比較表
- 学習曲線
- RMS角度
- RMS台車位置
- 10秒完走率
- RMSモータ力
- 物理条件
- センサ条件
- 元のActions run

です。

新しい学習が成功すれば、Pagesはその最新結果へ更新されます。

---

# ローカル実行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --preset quick --seed 42 --output-dir results
```

Windows PowerShellの場合は仮想環境の有効化方法を環境に合わせて変更してください。

学習済みcheckpointだけを再評価する場合は、

```bash
python -m src.evaluate results/models/100_percent.zip --preset normal --seed 42
```

です。

---

# リポジトリ構造

```text
rl-cartpole-ppo/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── train.yml
│   │   └── pages.yml
│   └── training-trigger/
├── configs/
│   ├── quick.yaml
│   ├── normal.yaml
│   └── long.yaml
├── src/
│   ├── environment.py
│   ├── train.py
│   ├── evaluation.py
│   ├── evaluate.py
│   └── reporting.py
├── tests/
│   └── test_smoke.py
├── web/
│   └── build_pages.py
├── requirements.txt
└── README.md
```

---

# 今回はまだやらないこと

最初の目標は**倒立状態付近からのBalance**です。

したがって、初期角度はおよそ±10°以内です。

まだ、

```text
     ●
     │
     O

からではなく

     O
     │
     ●

下向きから振り上げる Swing-up
```

は扱いません。

Balanceが安定した後、同じ物理モデルを使って、

1. 初期角度範囲を広げる
2. Swing-up用rewardへ変更
3. 下向きから振り上げる
4. Swing-up後にそのままBalanceへ移行

という順で難しくできます。

---

# 今後の候補

- 複数seedでPPOの再現性を評価
- モータ力を±5 Nまで低下
- モータ遅れを増加
- 摩擦係数ランダム化
- 質量ランダム化
- センサノイズ増加
- センサ遅延追加
- actuator saturation
- motor deadband
- SAC / TD3との比較
- Swing-up + balance
- Domain Randomization
- 実機Cart-Poleへのsim-to-real

特に、物理パラメータをepisodeごとに少しずつ変えるDomain Randomizationを入れると、単一の理想モデルだけに適応した方策から、多少条件が変化しても倒立を維持できる方策へ発展させられます。

---

# 設計上のポイント

このリポジトリでは、単に「CartPoleをPPOで解いた」という結果だけではなく、

- **何をAIが見ているか**
- **何をAIには隠しているか**
- **どの物理条件で学習したか**
- **どのセンサ誤差を入れたか**
- **学習途中でどう動きが変わったか**
- **制御性能が数値としてどう改善したか**

を再現可能な形で残すことを目的にしています。
