"""
数字人训练数据生成器 - CosyVoice3版本（修正版）
使用阿里开源的 Fun-CosyVoice3-0.5B 批量生成音频

使用方法:
1. 克隆CosyVoice仓库: git clone https://github.com/FunAudioLLM/CosyVoice.git
2. 安装依赖: cd CosyVoice && pip install -r requirements.txt
3. 将此脚本放在 CosyVoice 目录下
4. 准备参考音频: ./asset/zero_shot_prompt.wav (或自己的参考音频)
5. 运行: python tts_batch_generator.py

模式选择:
- instruct2 (推荐): 通过指令控制情绪、语速
- zero_shot: 纯音色克隆
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict

# 添加CosyVoice路径
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio


class CosyVoice3Generator:
    def __init__(self,
                 output_dir="training_data",
                 model_dir='pretrained_models/Fun-CosyVoice3-0.5B',
                 prompt_audio='./asset/zero_shot_prompt.wav',
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
            print("\n如果使用默认参考音频，请确保:")
            print("  ./asset/zero_shot_prompt.wav 存在")
            raise FileNotFoundError(f"参考音频不存在: {prompt_audio}")

        # 初始化模型
        print(f"\n正在加载 CosyVoice3 模型...")
        print(f"模型路径: {model_dir}")
        print(f"如果是首次运行，将自动下载模型（约500MB）")

        try:
            self.cosyvoice = AutoModel(model_dir=model_dir)
            self.sample_rate = self.cosyvoice.sample_rate

            print(f"✅ 模型加载成功！采样率: {self.sample_rate}Hz")
            print(f"✅ 使用模式: {mode}")
            print(f"✅ 参考音频: {prompt_audio}")
        except Exception as e:
            print(f"\n❌ 模型加载失败: {e}")
            print("\n请确保:")
            print("  1. 已克隆CosyVoice仓库")
            print("  2. 已安装所有依赖: pip install -r requirements.txt")
            print("  3. 在CosyVoice目录下运行此脚本")
            raise

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
        print(f"预计生成句子数: {sum(len(s['sentences']) for s in script_sections)}")

        for section_idx, section in enumerate(script_sections, 1):
            print(f"\n{'='*60}")
            print(f"[{section_idx}/{len(script_sections)}] {section['section']}")
            print(f"{'='*60}")

            for sentence_idx, sentence in enumerate(section['sentences'], 1):
                audio_id += 1

                # 生成音频文件名
                audio_filename = f"audio_{audio_id:04d}.wav"
                audio_path = self.output_dir / "audio" / audio_filename

                # 显示进度
                print(f"  [{sentence_idx}/{len(section['sentences'])}] 正在生成...")
                print(f"  文本: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
                print(f"  情绪: {section['emotion']}")

                try:
                    # 生成音频
                    audio_tensor = self.generate_audio(
                        text=sentence,
                        emotion=section['emotion']
                    )

                    # 保存音频
                    torchaudio.save(str(audio_path), audio_tensor, self.sample_rate)

                    # 计算时长
                    duration = audio_tensor.shape[1] / self.sample_rate

                    print(f"  ✅ 成功: {audio_filename} ({duration:.2f}s)")

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
                    print(f"  ❌ 生成失败: {e}")
                    import traceback
                    traceback.print_exc()
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
        print(f"成功生成: {len(all_metadata)} 个音频")
        print(f"失败数量: {audio_id - len(all_metadata)}")
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
            # 根据官方示例: inference_instruct2(text, prompt_text, prompt_audio, stream=False)
            output = None
            for i, j in enumerate(self.cosyvoice.inference_instruct2(
                text,
                prompt_text,
                self.prompt_audio,
                stream=False
            )):
                output = j['tts_speech']
                break  # 只取第一个输出

            if output is None:
                raise RuntimeError("生成失败：未返回音频数据")

            return output

        elif self.mode == 'zero_shot':
            # 使用zero_shot模式
            # 根据官方示例: inference_zero_shot(text, prompt_text, prompt_audio, stream=False)
            output = None
            for i, j in enumerate(self.cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                self.prompt_audio,
                stream=False
            )):
                output = j['tts_speech']
                break

            if output is None:
                raise RuntimeError("生成失败：未返回音频数据")

            return output

        else:
            raise ValueError(f"不支持的模式: {self.mode}，请使用 'instruct2' 或 'zero_shot'")


def test_cosyvoice_setup():
    """测试CosyVoice环境是否正确配置"""
    print("\n" + "="*60)
    print("测试CosyVoice环境")
    print("="*60)

    try:
        # 测试导入
        print("\n1. 测试导入...")
        from cosyvoice.cli.cosyvoice import AutoModel
        print("   ✅ 导入成功")

        # 测试模型加载
        print("\n2. 测试模型加载...")
        model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'
        if not os.path.exists(model_dir):
            print(f"   ⚠️  模型目录不存在: {model_dir}")
            print("   将在首次运行时自动下载")
        else:
            print(f"   ✅ 模型目录存在: {model_dir}")

        # 测试参考音频
        print("\n3. 测试参考音频...")
        prompt_audio = './asset/zero_shot_prompt.wav'
        if os.path.exists(prompt_audio):
            print(f"   ✅ 参考音频存在: {prompt_audio}")
        else:
            print(f"   ⚠️  参考音频不存在: {prompt_audio}")
            print("   请准备参考音频或使用自己的音频")

        print("\n" + "="*60)
        print("环境检查完成！可以开始生成")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 环境检查失败: {e}")
        print("\n请确保:")
        print("  1. 在CosyVoice目录下运行")
        print("  2. 已安装依赖: pip install -r requirements.txt")
        print("  3. 已添加路径: sys.path.append('third_party/Matcha-TTS')")
        return False


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='使用CosyVoice3批量生成训练音频')
    parser.add_argument('--script', default='training_script.json',
                       help='训练文案JSON文件路径')
    parser.add_argument('--output', default='digital_human_training_data',
                       help='输出目录')
    parser.add_argument('--model', default='pretrained_models/Fun-CosyVoice3-0.5B',
                       help='模型路径')
    parser.add_argument('--prompt', default='./asset/zero_shot_prompt.wav',
                       help='参考音频路径')
    parser.add_argument('--mode', default='instruct2', choices=['instruct2', 'zero_shot'],
                       help='生成模式: instruct2(推荐) 或 zero_shot')
    parser.add_argument('--test', action='store_true',
                       help='测试环境配置')

    args = parser.parse_args()

    # 如果是测试模式
    if args.test:
        test_cosyvoice_setup()
        exit(0)

    # 创建生成器并运行
    try:
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

        # 按情绪统计
        emotions = Counter([m['emotion'] for m in metadata])
        print(f"\n按情绪统计:")
        for emotion, count in emotions.items():
            print(f"   {emotion}: {count}句")

    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n请运行测试模式检查环境:")
        print("  python tts_batch_generator.py --test")
