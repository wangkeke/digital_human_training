"""
数字人训练数据生成器 - CosyVoice3版本
使用阿里开源的 Fun-CosyVoice3-0.5B 批量生成音频

使用方法:
1. 安装依赖: pip install modelscope torchaudio
2. 下载模型: 自动从modelscope下载 Fun-CosyVoice3-0.5B
3. 准备参考音频: 将"小O"的音频样本放在 ./reference_audio/prompt.wav
4. 运行: python tts_batch_generator.py

模式选择:
- instruct2 (推荐): 通过指令控制情绪、语速，需要参考音频
- zero_shot: 纯音色克隆，需要参考音频
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import torchaudio
from modelscope import AutoModel

class CosyVoice3Generator:
    def __init__(self,
                 output_dir="training_data",
                 model_dir='pretrained_models/Fun-CosyVoice3-0.5B',
                 prompt_audio='./reference_audio/prompt.wav',
                 mode='instruct2'):
        """
        初始化CosyVoice3生成器

        Args:
            output_dir: 输出目录
            model_dir: 模型路径（会自动下载）
            prompt_audio: 参考音频路径（"小O"的音频样本）
            mode: 生成模式 ('instruct2' 或 'zero_shot')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 创建子目录
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        self.prompt_audio = prompt_audio
        self.mode = mode

        # 检查参考音频是否存在
        if not os.path.exists(prompt_audio):
            print(f"\n⚠️  警告: 参考音频不存在: {prompt_audio}")
            print("请准备一段'小O'的音频样本（10-30秒），保存为:")
            print(f"  {prompt_audio}")
            print("\n如果没有参考音频，将使用默认音色")
            self.prompt_audio = None

        # 初始化模型
        print(f"\n正在加载 CosyVoice3 模型...")
        print(f"模型路径: {model_dir}")
        print(f"如果是首次运行，将自动从 ModelScope 下载模型（约500MB）")

        self.cosyvoice = AutoModel(model_dir=model_dir)
        self.sample_rate = self.cosyvoice.sample_rate

        print(f"✅ 模型加载成功！采样率: {self.sample_rate}Hz")
        print(f"✅ 使用模式: {mode}")

        # 情绪映射到指令
        self.emotion_to_instruct = {
            'neutral': '请用平静自然的语气说话。',
            'happy': '请用开心、愉快、兴奋的语气说话。',
            'sad': '请用悲伤、低落、难过的语气说话。',
            'angry': '请用愤怒、生气、激动的语气说话。',
            'surprised': '请用惊讶、吃惊、不可思议的语气说话。',
            'thoughtful': '请用思考、沉思、犹豫的语气说话。',
            'fearful': '请用害怕、紧张、担心的语气说话。',
            'tired': '请用疲惫、无奈、懒散的语气说话。',
            'gentle': '请用温柔、柔和、轻声细语的语气说话。',
            'confident': '请用自信、坚定、有力的语气说话。',
            'professional': '请用专业、正式、严肃的语气说话。',
            'casual': '请用随意、轻松、聊天的语气说话。',
            'curious': '请用好奇、疑问、询问的语气说话。',
            'storytelling': '请用讲故事般生动、有趣、抓人的语气说话。'
        }

    def generate_training_set(self, script_json_path="training_script.json"):
        """生成完整训练集"""

        # 从JSON文件加载文案
        with open(script_json_path, 'r', encoding='utf-8') as f:
            script_sections = json.load(f)

        # 生成音频
        all_metadata = []
        audio_id = 0

        print(f"\n开始生成音频...")
        print(f"总section数: {len(script_sections)}")

        for section_idx, section in enumerate(script_sections, 1):
            print(f"\n[{section_idx}/{len(script_sections)}] 生成 [{section['section']}]")

            for sentence in section['sentences']:
                audio_id += 1

                # 生成音频文件名
                audio_filename = f"audio_{audio_id:04d}.wav"
                audio_path = self.output_dir / "audio" / audio_filename

                # 生成音频
                print(f"  {audio_id:04d}. {sentence[:30]}{'...' if len(sentence) > 30 else ''}")

                try:
                    audio_tensor = self.generate_audio(
                        text=sentence,
                        emotion=section['emotion']
                    )

                    # 保存音频
                    torchaudio.save(str(audio_path), audio_tensor, self.sample_rate)

                    # 计算时长
                    duration = audio_tensor.shape[1] / self.sample_rate

                    # 记录元数据
                    metadata = {
                        "id": audio_id,
                        "filename": audio_filename,
                        "text": sentence,
                        "section": section['section'],
                        "emotion": section['emotion'],
                        "duration": round(duration, 2),
                        "sample_rate": self.sample_rate
                    }
                    all_metadata.append(metadata)

                except Exception as e:
                    print(f"    ⚠️  生成失败: {e}")
                    continue

        # 保存元数据索引
        metadata_path = self.output_dir / "metadata" / "index.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

        # 统计信息
        total_duration = sum(m['duration'] for m in all_metadata)

        print(f"\n{'='*60}")
        print(f"✅ 完成！")
        print(f"{'='*60}")
        print(f"生成音频数: {len(all_metadata)}")
        print(f"总时长: {total_duration/60:.1f} 分钟")
        print(f"输出目录: {self.output_dir}")
        print(f"元数据: {metadata_path}")

        return all_metadata

    def generate_audio(self, text: str, emotion: str = "neutral"):
        """
        使用CosyVoice3生成单句音频

        Args:
            text: 要生成的文本
            emotion: 情绪标签

        Returns:
            audio_tensor: torch.Tensor, shape (1, samples)
        """
        # 构建指令
        instruct = self.emotion_to_instruct.get(emotion, self.emotion_to_instruct['neutral'])
        prompt_text = f"You are a helpful assistant. {instruct}<|endofprompt|>"

        if self.mode == 'instruct2':
            # 使用instruct2模式（推荐）
            output = None
            for i, j in enumerate(self.cosyvoice.inference_instruct2(
                text,
                prompt_text,
                self.prompt_audio if self.prompt_audio else './asset/zero_shot_prompt.wav',
                stream=False
            )):
                output = j['tts_speech']
                break  # 只取第一个输出

            return output

        elif self.mode == 'zero_shot':
            # 使用zero_shot模式
            output = None
            for i, j in enumerate(self.cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                self.prompt_audio if self.prompt_audio else './asset/zero_shot_prompt.wav',
                stream=False
            )):
                output = j['tts_speech']
                break

            return output

        else:
            raise ValueError(f"不支持的模式: {self.mode}，请使用 'instruct2' 或 'zero_shot'")


def prepare_reference_audio():
    """准备参考音频的辅助函数"""
    ref_dir = Path('./reference_audio')
    ref_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("准备参考音频")
    print("="*60)
    print("\n为了生成具有'小O'音色的音频，你需要：")
    print("\n1. 录制一段'小O'的音频（10-30秒）")
    print("   - 内容：随便说几句话，自然即可")
    print("   - 质量：清晰、无噪音、无背景音乐")
    print("   - 格式：WAV 或 MP3")
    print("\n2. 将音频保存为: ./reference_audio/prompt.wav")
    print("\n3. 示例录音内容:")
    print("   '你好，我是小O。希望你以后能够做得比我还好呦。'")
    print("\n如果没有参考音频，将使用默认音色。")
    print("="*60 + "\n")


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='使用CosyVoice3批量生成训练音频')
    parser.add_argument('--script', default='./data/training_complete.json',
                       help='训练文案JSON文件路径')
    parser.add_argument('--output', default='digital_human_training_data',
                       help='输出目录')
    parser.add_argument('--model', default='pretrained_models/Fun-CosyVoice3-0.5B',
                       help='模型路径')
    parser.add_argument('--prompt', default='./reference_audio/prompt.wav',
                       help='参考音频路径')
    parser.add_argument('--mode', default='instruct2', choices=['instruct2', 'zero_shot'],
                       help='生成模式: instruct2(推荐) 或 zero_shot')
    parser.add_argument('--prepare-ref', action='store_true',
                       help='显示如何准备参考音频的说明')

    args = parser.parse_args()

    # 如果用户请求准备参考音频的说明
    if args.prepare_ref:
        prepare_reference_audio()
        exit(0)

    # 创建生成器并运行
    generator = CosyVoice3Generator(
        output_dir=args.output,
        model_dir=args.model,
        prompt_audio=args.prompt,
        mode=args.mode
    )

    metadata = generator.generate_training_set(script_json_path=args.script)

    print("\n📊 生成统计:")
    print(f"   总句数: {len(metadata)}")

    # 按section统计
    from collections import Counter
    sections = Counter([m['section'] for m in metadata])
    print(f"   Section数: {len(sections)}")
    print(f"\n前10个section:")
    for section, count in list(sections.items())[:10]:
        print(f"     {section}: {count}句")
