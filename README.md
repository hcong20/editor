# 电影一键浓缩系统 MVP

这是一个面向“电影自动解说/自动浓缩”的 Python 骨架项目，围绕下面这条主链路组织：

1. 场景切分
2. ASR 转字幕
3. 场景语义理解
4. 故事结构抽取
5. 解说脚本生成
6. 关键片段选择
7. 输出剪辑计划与多版本产物规划
8. 自动交付（可选）：剪辑 + 配音 + 字幕 + 成片合成

当前版本优先解决两件事：

- 把整个系统的数据流和模块边界搭起来
- 提供一个无需外部模型也能跑通的 heuristic/stub MVP
- 为 `PySceneDetect` 和 `faster-whisper` 预留真实 provider，并支持 `auto` 回退
- 已接入 `GPT-4o` / `Gemini` / `Ollama` 做 scene 理解和 B 站风格脚本生成

后续你可以逐步把 `stub` provider 换成真实组件，比如：

- `PySceneDetect` / `TransNetV2` / `FFmpeg` 做镜头切分
- `faster-whisper` / `Whisper` 做 ASR
- `GPT-4o` / `Gemini` / `Ollama` / `LLaVA` 做 scene 级理解与脚本生成
- `FFmpeg` / `MoviePy` 做自动剪辑
- `Edge TTS` / `ElevenLabs` / `Azure TTS` 做自动配音
- `Whisper` / `faster-whisper` 做自动字幕

## 快速开始

```bash
python3 main.py \
  --input /path/to/movie.mp4 \
  --output /Users/eden/Documents/projects/editor/runs/demo \
  --config /Users/eden/Documents/projects/editor/configs/pipeline.example.toml
```

如果你暂时没有 ASR 模型，MVP 也能跑：

- 它会尝试读取同名字幕 sidecar 文件，如 `.srt` / `.vtt` / `.txt`
- 如果没有字幕，会自动生成占位 transcript，便于先验证全流程

## 真实 Provider 模式

安装可选依赖：

```bash
python3 -m pip install -e ".[video,delivery]"
```

使用真实 provider：

```bash
python3 main.py \
  --input /path/to/movie.mp4 \
  --output /Users/eden/Documents/projects/editor/runs/real \
  --config /Users/eden/Documents/projects/editor/configs/pipeline.real.example.toml
```

或者使用自动回退模式，在配置里写：

```toml
[providers]
shot_detector = "auto"
asr = "auto"
```

`auto` 的行为：

- 如果检测到 `scenedetect`，就用 `PySceneDetect`
- 如果检测到 `faster_whisper`，就用 `faster-whisper`
- 如果存在同名 `.srt/.vtt/.txt`，ASR 会优先读 sidecar 字幕
- 如果依赖没装，则自动回退到 `stub`

## 自动交付模式（剪辑 + 配音 + 字幕）

你可以在 pipeline 末端直接输出“解说版成片”：

- 视频裁剪: `FFmpeg` / `MoviePy` / `CapCut API`（未配置 API 时自动回退到本地剪辑）
- 自动配音: `Edge TTS` / `ElevenLabs` / `Azure TTS` / `stub`
- 自动字幕: `Whisper(faster-whisper)` / `heuristic`

命令行直接开启交付：

```bash
python3 main.py \
  --input /path/to/movie.mp4 \
  --output /Users/eden/Documents/projects/editor/runs/demo_delivery \
  --config /Users/eden/Documents/projects/editor/configs/pipeline.example.toml \
  --deliver \
  --variant commentary_10m
```

也可以在配置中开启：

```toml
[delivery]
enabled = true
variant = "commentary_10m"
burn_subtitles = true

[providers]
video_editor = "ffmpeg"   # ffmpeg | moviepy | capcut-api | auto
tts = "auto"              # edge-tts | elevenlabs | azure | stub | auto
subtitles = "auto"        # whisper | heuristic | auto
```

## LLM Provider 模式

### OpenAI GPT-4o

```bash
export OPENAI_API_KEY=your_key
python3 main.py \
  --input /path/to/movie.mp4 \
  --output /Users/eden/Documents/projects/editor/runs/openai \
  --config /Users/eden/Documents/projects/editor/configs/pipeline.openai.example.toml
```

### Gemini

```bash
export GEMINI_API_KEY=your_key
python3 main.py \
  --input /path/to/movie.mp4 \
  --output /Users/eden/Documents/projects/editor/runs/gemini \
  --config /Users/eden/Documents/projects/editor/configs/pipeline.gemini.example.toml
```

### Ollama（本地模型）

先安装 Ollama Python SDK（可选，但推荐）：

```bash
python3 -m pip install -e ".[llm]"
```

先启动 Ollama 服务并准备模型（示例）：

```bash
ollama serve
ollama pull gemma4:31b
ollama pull gemma4:31b
```

再运行 pipeline：

```bash
python3 main.py \
  --input /path/to/movie.mp4 \
  --output /Users/eden/Documents/projects/editor/runs/ollama \
  --config /Users/eden/Documents/projects/editor/configs/pipeline.ollama.example.toml
```

这三个 provider 都会：

- 用 `ffmpeg` 从每个 scene 抽取关键帧
- 把关键帧和 transcript_excerpt 一起发给模型
- 输出结构化 scene 分析
- 输出 story beat、高潮锚点和关键因果链
- 生成更接近 B 站电影解说风格的中文口播稿

如果你只想自动选用可用的大模型，也可以把配置写成：

```toml
[providers]
vision = "auto"
story = "auto"
script = "auto"
```

`auto` 会优先用 `OPENAI_API_KEY`，其次尝试 `GEMINI_API_KEY`，再尝试本地 `Ollama`，都不可用时退回 heuristic/template（story 阶段会退回 heuristic）。

`Ollama` provider 在检测到 `ollama-python` 时会优先使用 SDK，请求失败时会自动回退到 HTTP 调用。

## 目录结构

```text
configs/
  pipeline.example.toml
  pipeline.ollama.example.toml
docs/
  architecture.zh-CN.md
src/movie_brief/
  cli.py
  config.py
  models.py
  pipeline.py
  prompts.py
  utils.py
  stages/
tests/
main.py
```

## 当前已实现

- 可运行的 CLI 入口
- 数据模型与阶段化 pipeline
- 镜头切分 stub
- ASR sidecar/stub/faster-whisper 扩展点
- `PySceneDetect` / `faster-whisper` 真实 provider 接口
- `auto` provider 自动回退
- `GPT-4o` / `Gemini` / `Ollama` 的多模态 scene 理解
- `GPT-4o` / `Gemini` / `Ollama` 的故事结构抽取（高潮识别 + 因果链）
- B 站风格 JSON 结构化脚本生成
- 每个 scene 的关键帧抽取与落盘
- 场景聚合与启发式重要性打分
- 四段式故事结构建模
- 中文解说脚本模板生成
- 多版本剪辑计划输出
- 自动交付执行（可选）：导出片段、拼接视频、TTS 配音、SRT 字幕、最终成片
- `unittest` smoke test

## 输出文件

每次运行会在输出目录生成：

- `01_shots.json`
- `02_transcript.json`
- `03_scenes.json`
- `04_story_beats.json`（包含 `beats` 与 `cross_beat_causal_graph`，edge 含 `causal_strength`）
- `05_script.json`
- `06_selected_scenes.json`
- `07_render_plan.json`
- `08_delivery.json`（当 `delivery.enabled=true` 时）
- `scene_frames/`
- `summary.md`

## 推荐的下一步

1. 在 `story_modeling.py` 里加入 LLM 版故事结构抽取，把三幕/高潮定位做得更稳
2. 为 `CapCut API` 增加稳定的上传/任务轮询/下载实现
3. 为 `PySceneDetect` 增加 `TransNetV2` 备选 provider
4. 给 ASR 增加说话人分离和角色对齐
5. 增加 prompt cache / scene cache，避免重复花费 LLM 成本

## 风险提醒

- 真实产品必须正面处理版权问题
- “高潮识别”不能只看音量或情绪词，最好结合剧情因果链
- 多人物交叉叙事需要 scene graph 或角色状态跟踪，单纯摘要很容易丢线
