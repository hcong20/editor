# 系统架构说明

## 1. 总体目标

输入一部电影，输出三类产物：

- 10 分钟左右的完整解说版
- 3 分钟的短视频版
- 1 分钟的爆点版

并同时产出：

- 解说脚本
- 剪辑片段清单
- 故事结构摘要

## 2. 推荐主链路

```text
电影视频
  -> 镜头切分
  -> ASR 转录
  -> scene 级语义理解
  -> 故事结构建模
  -> 解说脚本生成
  -> 片段选择与重组
  -> 剪辑 / 配音 / 字幕
```

## 3. 当前代码里的模块映射

- `shot_detection.py`
  - 输入：原始视频
  - 输出：shot 列表
- `asr.py`
  - 输入：视频
  - 输出：带时间戳的 utterance 列表
- `scene_understanding.py`
  - 输入：shots + transcript
  - 输出：scene 列表与重要性分数
- `story_modeling.py`
  - 输入：scene 列表
  - 输出：四段式故事 beat
- `script_generation.py`
  - 输入：story beat + scene
  - 输出：中文解说稿
- `selection.py`
  - 输入：scene + story beat
  - 输出：入选片段
- `render_plan.py`
  - 输入：入选片段 + 解说稿
  - 输出：多版本剪辑计划

## 4. 数据粒度设计

建议坚持三层粒度：

1. `Shot`
   - 画面切分最小单位
   - 适合做镜头检测和基础时间轴
2. `Scene`
   - 叙事理解单位
   - 适合做人物、冲突、情绪和剧情推进分析
3. `StoryBeat`
   - 结构重组单位
   - 适合生成解说脚本和 10 分钟浓缩版

## 5. 重要性评分建议

当前代码里使用的核心思路是：

```text
Importance Score
= 情绪强度
+ 角色核心度
+ 冲突程度
+ 剧情推进度
```

后面建议把它扩展成：

```text
Importance
= w1 * emotion
+ w2 * conflict
+ w3 * protagonist_relevance
+ w4 * causal_impact
+ w5 * narrative_uniqueness
```

其中最难的不是情绪，而是 `causal_impact`：

- 这一段是否真正改变了人物状态
- 是否引出后续关键事件
- 是否是冲突升级节点

## 6. 为什么不能只按时间裁剪

好的电影解说通常不是简单顺剪，而是“结构重组”：

- 背景介绍
- 冲突爆发
- 高潮对抗
- 结局收束

所以这个项目默认会保留 story beat 层，而不是只做时间轴摘要。

## 7. 下一阶段推荐接法

### 镜头切分

- MVP：`PySceneDetect`
- 高精度：`TransNetV2`

当前代码已接入：

- `stub`
- `pyscenedetect`
- `auto`

### ASR

- 本地优先：`faster-whisper`
- 云端优先：`OpenAI Whisper API`

当前代码已接入：

- `stub`
- `faster-whisper`
- `auto`

并支持优先读取 sidecar 字幕：

- `.srt`
- `.vtt`
- `.txt`

### 场景理解

- 先抽关键帧
- 再把每个 scene 的关键帧 + transcript 喂给多模态模型

当前代码已接入：

- `openai`
- `gemini`
- `auto`

产出字段包括：

- `summary`
- `events`
- `characters`
- `visual_cues`
- 情绪/冲突/剧情推进分数

### 文案生成

- 可以用一轮结构抽取 prompt
- 再用一轮“B 站电影解说风格”脚本 prompt

当前代码已接入：

- `openai`
- `gemini`
- `auto`

输出仍然是结构化 `ScriptSegment`，方便后面继续做：

- 自动配音
- 旁白对齐
- 按段落挑片

## 8. 商业与合规风险

- 影视版权
- 模型生成内容的误判和事实偏差
- 不同国家/平台对剪辑搬运的政策不同

如果你要做创业级产品，建议尽早把：

- 内容来源授权
- 审核策略
- 水印与追溯
- 模板化配音与标题生成

一起纳入系统设计。
