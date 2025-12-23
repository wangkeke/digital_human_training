# 数字人训练数据生成使用指南

## 📦 文件说明

1. **training_script.json** - 完整训练文案数据
2. **tts_batch_generator.py** - TTS批量生成脚本

## 🚀 快速开始

### 步骤1：准备环境

```bash
# 创建项目目录
mkdir digital_human_training
cd digital_human_training

# 安装依赖
pip install soundfile numpy

# 根据你选择的TTS安装对应的库
# ChatTTS
pip install ChatTTS

# 或 Kokoro
pip install kokoro-tts

# 或 CosyVoice
pip install cosyvoice
```

### 步骤2：准备文件

```bash
# 将两个artifact保存为文件
# 1. 复制 "完整训练文案JSON数据" 保存为 training_script.json
# 2. 复制 "TTS批量生成脚本" 保存为 tts_batch_generator.py

# 目录结构应该是：
digital_human_training/
├── training_script.json
└── tts_batch_generator.py
```

### 步骤3：配置TTS模型

编辑 `tts_batch_generator.py`，根据你的TTS修改以下部分：

#### 如果使用 ChatTTS（推荐）

```python
from ChatTTS import ChatTTS

class TrainingDataGenerator:
    def __init__(self, output_dir="training_data"):
        # ... 其他代码 ...

        # 初始化ChatTTS
        self.tts = ChatTTS.Chat()
        self.tts.load_models()

    def generate_audio(self, text: str, emotion: str = "neutral"):
        # 情绪映射
        emotion_params = {
            'neutral': {'temperature': 0.3},
            'happy': {'temperature': 0.5, 'top_p': 0.7},
            'sad': {'temperature': 0.2, 'top_p': 0.8},
            'angry': {'temperature': 0.6, 'top_p': 0.6},
            'surprised': {'temperature': 0.5, 'top_p': 0.7}
        }

        params = emotion_params.get(emotion, emotion_params['neutral'])

        # 生成音频
        wavs = self.tts.infer(
            text,
            params_infer_code=params,
            use_decoder=True
        )

        # 返回numpy array
        return wavs[0]
```

#### 如果使用 Kokoro

```python
from kokoro import generate

class TrainingDataGenerator:
    def __init__(self, output_dir="training_data"):
        # ... 其他代码 ...
        # Kokoro不需要初始化
        pass

    def generate_audio(self, text: str, emotion: str = "neutral"):
        # Kokoro通过语速和音调模拟情绪
        speed_map = {
            'neutral': 1.0,
            'happy': 1.1,
            'sad': 0.9,
            'angry': 1.05,
            'surprised': 1.15
        }

        speed = speed_map.get(emotion, 1.0)

        # 生成音频
        audio = generate(
            text,
            voice='af_sky',  # 或其他可用声音
            speed=speed
        )

        return audio
```

#### 如果使用 CosyVoice

```python
from cosyvoice.cli.cosyvoice import CosyVoice

class TrainingDataGenerator:
    def __init__(self, output_dir="training_data"):
        # ... 其他代码 ...

        # 初始化CosyVoice
        self.tts = CosyVoice('pretrained_models/CosyVoice-300M')

    def generate_audio(self, text: str, emotion: str = "neutral"):
        # 情绪描述
        emotion_instruct = {
            'neutral': '平静地说',
            'happy': '开心地说',
            'sad': '悲伤地说',
            'angry': '愤怒地说',
            'surprised': '惊讶地说',
            'thoughtful': '思考着说',
            'fearful': '害怕地说',
            'tired': '疲惫地说',
            'gentle': '温柔地说',
            'confident': '自信地说',
            'professional': '专业地说',
            'casual': '随意地说',
            'curious': '好奇地问',
            'storytelling': '讲故事般说'
        }

        instruct = emotion_instruct.get(emotion, '平静地说')

        # 生成音频
        output = self.tts.inference_instruct(
            text,
            sft_dropdown='中文女',
            instruct_text=instruct
        )

        # 提取音频数据
        audio = output['tts_speech'].numpy()
        return audio
```

### 步骤4：运行生成

```bash
python tts_batch_generator.py
```

### 步骤5：查看结果

生成完成后，目录结构：

```
digital_human_training/
├── training_script.json
├── tts_batch_generator.py
└── digital_human_training_data/
    ├── audio/
    │   ├── audio_0001.wav  # 大家好啊，花花...
    │   ├── audio_0002.wav  # 一七西瓜...
    │   ├── audio_0003.wav  # 五路不图书...
    │   └── ... (约80+个文件)
    └── metadata/
        └── index.json      # 所有音频的索引
```

## 📊 数据统计

完整文案包含：
- **43个section**（音素和场景分类）
- **约85-90句话**（取决于句子拆分）
- **总时长**：约20-30分钟音频
- **覆盖内容**：
  - ✅ 21个中文声母
  - ✅ 39个韵母
  - ✅ 四声调
  - ✅ 10种情绪
  - ✅ 英文基础音素

## 🔧 高级配置

### 1. 自定义文案

如果你想添加自己的句子，编辑 `training_script.json`：

```json
[
  {
    "section": "自定义场景-问候语",
    "emotion": "happy",
    "sentences": [
      "你好，欢迎光临！",
      "很高兴见到你！"
    ]
  }
]
```

### 2. 多版本生成（数据增强）

修改 `generate_training_set()` 方法：

```python
# 同一句话生成多个版本
for sentence in section['sentences']:
    for speed in [0.9, 1.0, 1.1]:  # 三种语速
        audio_data = self.generate_audio(
            text=sentence,
            emotion=section['emotion'],
            speed=speed  # 需要TTS支持
        )

        audio_filename = f"audio_{audio_id:04d}_speed{speed}.wav"
        self.save_audio(audio_data, audio_path)
```

### 3. 批量并行生成

```python
from multiprocessing import Pool

def generate_one_audio(args):
    text, emotion, output_path = args
    # ... 生成逻辑

# 在 generate_training_set() 中：
tasks = []
for section in script_sections:
    for sentence in section['sentences']:
        tasks.append((sentence, section['emotion'], output_path))

# 8进程并行
with Pool(8) as pool:
    pool.map(generate_one_audio, tasks)
```

## ⚠️ 常见问题

### Q1: 生成的音频质量不好？

**A:** 检查以下几点：
1. TTS模型是否正确加载
2. 情绪参数是否设置合理
3. 采样率是否一致（建议22050或24000）
4. 音频是否有噪音（检查降噪设置）

### Q2: 某些情绪效果不明显？

**A:**
1. 不同TTS对情绪支持不同，ChatTTS和CosyVoice效果较好
2. 可以手动调整情绪参数（temperature, top_p等）
3. 或者后期用音频处理调整（pitch shift等）

### Q3: 生成速度太慢？

**A:**
1. 使用GPU加速（如果TTS支持）
2. 使用多进程并行生成
3. 先生成一小部分测试，确认无误后再批量生成

### Q4: 如何验证生成的音频是否正确？

**A:** 添加质量检查代码：

```python
def check_audio_quality(audio_path):
    import librosa
    audio, sr = librosa.load(audio_path)

    # 检查时长
    duration = len(audio) / sr
    if duration < 0.5 or duration > 10:
        print(f"⚠️ 时长异常: {audio_path}")

    # 检查音量
    volume = np.abs(audio).mean()
    if volume < 0.01:
        print(f"⚠️ 音量过低: {audio_path}")

    return duration, volume

# 在生成后调用
for meta in all_metadata:
    check_audio_quality(meta['filename'])
```

## 📝 下一步：生成训练视频

生成音频后，需要配合视频生成运动参数：

1. **方法A - 真人视频录制**
   - 找一个人按顺序读完所有句子
   - 用LivePortrait提取运动参数

2. **方法B - AI生成视频**
   - 使用JoyVASA从音频生成视频
   - 用LivePortrait提取运动参数

3. **方法C - 使用现成数据集**
   - MEAD数据集（多情绪）
   - VoxCeleb数据集

详见后续教程：《Audio2Motion模型训练指南》

## 📚 相关资源

- [LivePortrait官方仓库](https://github.com/KwaiVGI/LivePortrait)
- [ChatTTS文档](https://github.com/2noise/ChatTTS)
- [CosyVoice文档](https://github.com/FunAudioLLM/CosyVoice)
- [数字人完整训练流程](链接到你的文档)

## 💡 提示

- 建议先用10句话测试完整流程
- 确认音质、情绪、时长都符合预期
- 再批量生成全部数据
- 保留原始音频（不要压缩），后续训练时可能需要重新处理
