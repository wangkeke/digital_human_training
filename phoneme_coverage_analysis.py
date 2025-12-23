"""
音素覆盖度分析工具（正确版本）
使用pypinyin自动转换拼音，精确分析音素覆盖
"""

import json
from typing import Set, List, Dict, Tuple
import re
from collections import Counter

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("⚠️  警告: 未安装pypinyin库，请运行: pip install pypinyin")

try:
    import eng_to_ipa  # 英文音素转换
    ENG_IPA_AVAILABLE = True
except ImportError:
    ENG_IPA_AVAILABLE = False
    print("⚠️  提示: 未安装eng_to_ipa，英文分析将使用简化方法")
    print("   安装命令: pip install eng-to-ipa")


class PhonemeAnalyzer:
    def __init__(self):
        # 中文声母（21个）+ 零声母
        self.chinese_initials = {
            'b', 'p', 'm', 'f',      # 双唇音、唇齿音
            'd', 't', 'n', 'l',      # 舌尖音
            'g', 'k', 'h',           # 舌根音
            'j', 'q', 'x',           # 舌面音
            'zh', 'ch', 'sh', 'r',   # 翘舌音
            'z', 'c', 's',           # 平舌音
            ''                        # 零声母（如"啊"、"欧"）
        }

        # 中文韵母（39个主要韵母）
        self.chinese_finals = {
            # 单韵母
            'a', 'o', 'e', 'i', 'u', 'v',  # v代表ü
            # 复韵母
            'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie', 've', 'er',
            # 前鼻韵母
            'an', 'en', 'in', 'un', 'vn',
            # 后鼻韵母
            'ang', 'eng', 'ing', 'ong',
            # 复杂韵母
            'ian', 'uan', 'van', 'iang', 'uang', 'iong', 'uai'
        }

        # 英文CMU音素（ARPAbet）
        self.cmu_vowels = {
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
            'EH', 'ER', 'EY', 'IH', 'IY', 'OW',
            'OY', 'UH', 'UW'
        }

        self.cmu_consonants = {
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH',
            'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S',
            'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
        }

        # 英文音素简化映射（用于没有eng_to_ipa的情况）
        self.simple_eng_phoneme_map = {
            'th': ['TH', 'DH'],
            'ng': ['NG'],
            'sh': ['SH'],
            'ch': ['CH'],
            'zh': ['ZH'],
            'oy': ['OY'],
            'ow': ['AW', 'OW'],
            'er': ['ER'],
            'oo': ['UH', 'UW'],
            'ee': ['IY'],
            'ay': ['AY', 'EY']
        }

    def extract_initial_final(self, pinyin_str: str) -> Tuple[str, str]:
        """
        从拼音字符串提取声母和韵母

        Args:
            pinyin_str: 拼音字符串，如 "zhang", "qi", "a"

        Returns:
            (声母, 韵母) 如 ("zh", "ang"), ("q", "i"), ("", "a")
        """
        # 去除声调数字
        pinyin_str = re.sub(r'[0-9]', '', pinyin_str).lower()

        # 翘舌音（双字母声母）
        if pinyin_str.startswith(('zh', 'ch', 'sh')):
            return pinyin_str[:2], pinyin_str[2:] or 'i'  # "zhi" -> ("zh", "i")

        # 平舌音和其他辅音
        for initial in ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                       'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's']:
            if pinyin_str.startswith(initial):
                final = pinyin_str[len(initial):] or 'i'  # "ci" -> ("c", "i")
                # 特殊处理：ü在某些拼音中写作u
                if initial in ['j', 'q', 'x'] and 'u' in final:
                    final = final.replace('u', 'v')
                return initial, final

        # 零声母
        return '', pinyin_str

    def analyze_chinese_text(self, text: str) -> Dict:
        """分析中文文本的音素覆盖"""
        if not PYPINYIN_AVAILABLE:
            return {'error': '需要安装pypinyin: pip install pypinyin'}

        # 提取所有汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)

        if not chinese_chars:
            return {
                'found_initials': set(),
                'found_finals': set(),
                'char_count': 0
            }

        # 转换为拼音
        pinyins = pinyin(''.join(chinese_chars), style=Style.NORMAL, heteronym=False)

        found_initials = set()
        found_finals = set()
        pinyin_details = []

        for i, py_list in enumerate(pinyins):
            py = py_list[0]
            initial, final = self.extract_initial_final(py)

            found_initials.add(initial)
            found_finals.add(final)

            pinyin_details.append({
                'char': chinese_chars[i],
                'pinyin': py,
                'initial': initial or '(零声母)',
                'final': final
            })

        return {
            'found_initials': found_initials,
            'found_finals': found_finals,
            'char_count': len(chinese_chars),
            'pinyin_details': pinyin_details,
            'initial_coverage': len(found_initials & self.chinese_initials) / len(self.chinese_initials),
            'final_coverage': len(found_finals & self.chinese_finals) / len(self.chinese_finals)
        }

    def analyze_english_text(self, text: str) -> Dict:
        """分析英文文本的音素覆盖"""
        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())

        if not english_words:
            return {
                'found_phonemes': set(),
                'word_count': 0
            }

        found_phonemes = set()

        if ENG_IPA_AVAILABLE:
            # 使用精确的IPA转换
            try:
                import eng_to_ipa as ipa
                for word in english_words:
                    try:
                        ipa_str = ipa.convert(word)
                        # 这里需要从IPA转换到ARPAbet，简化处理
                        # 实际应该使用专门的库
                        found_phonemes.update(self._ipa_to_arpabet(ipa_str))
                    except:
                        pass
            except:
                pass

        # 简化方法：基于字母组合推断
        text_lower = ' '.join(english_words)
        for pattern, phonemes in self.simple_eng_phoneme_map.items():
            if pattern in text_lower:
                found_phonemes.update(phonemes)

        # 基本辅音检测
        for char in 'bcdfghjklmnpqrstvwxyz':
            if char in text_lower:
                found_phonemes.add(char.upper())

        all_cmu = self.cmu_vowels | self.cmu_consonants

        return {
            'found_phonemes': found_phonemes,
            'word_count': len(english_words),
            'vowel_coverage': len(found_phonemes & self.cmu_vowels) / len(self.cmu_vowels),
            'consonant_coverage': len(found_phonemes & self.cmu_consonants) / len(self.cmu_consonants)
        }

    def _ipa_to_arpabet(self, ipa_str: str) -> Set[str]:
        """IPA到ARPAbet的简化映射"""
        # 这是一个简化版本，实际应该更复杂
        arpabet = set()
        # 这里可以扩展更详细的映射
        return arpabet

    def analyze_coverage(self, script_json_path: str, verbose: bool = False):
        """分析训练脚本的音素覆盖度"""

        # 加载文案
        with open(script_json_path, 'r', encoding='utf-8') as f:
            script_data = json.load(f)

        # 收集所有文本
        all_chinese_text = []
        all_english_text = []

        for section in script_data:
            for sentence in section['sentences']:
                # 判断是否包含英文
                if re.search(r'[a-zA-Z]', sentence):
                    all_english_text.append(sentence)
                # 判断是否包含中文
                if re.search(r'[\u4e00-\u9fff]', sentence):
                    all_chinese_text.append(sentence)

        # 分析中文
        chinese_combined = ''.join(all_chinese_text)
        chinese_result = self.analyze_chinese_text(chinese_combined)

        # 分析英文
        english_combined = ' '.join(all_english_text)
        english_result = self.analyze_english_text(english_combined)

        # 计算缺失
        chinese_result['missing_initials'] = sorted(
            self.chinese_initials - chinese_result['found_initials']
        )
        chinese_result['missing_finals'] = sorted(
            self.chinese_finals - chinese_result['found_finals']
        )

        all_cmu = self.cmu_vowels | self.cmu_consonants
        english_result['missing_phonemes'] = sorted(
            all_cmu - english_result['found_phonemes']
        )

        # 识别关键缺失
        critical_phonemes = ['TH', 'DH', 'NG', 'ZH', 'OY', 'AW']
        english_result['critical_missing'] = [
            p for p in critical_phonemes
            if p in english_result['missing_phonemes']
        ]

        return {
            'chinese': chinese_result,
            'english': english_result,
            'summary': self._generate_summary(chinese_result, english_result)
        }

    def _generate_summary(self, chinese_result, english_result) -> Dict:
        """生成总结报告"""
        return {
            'chinese_initial_rate': f"{chinese_result['initial_coverage']*100:.1f}%",
            'chinese_final_rate': f"{chinese_result['final_coverage']*100:.1f}%",
            'english_vowel_rate': f"{english_result['vowel_coverage']*100:.1f}%",
            'english_consonant_rate': f"{english_result['consonant_coverage']*100:.1f}%",
            'overall_rating': self._calculate_rating(chinese_result, english_result)
        }

    def _calculate_rating(self, chinese_result, english_result) -> str:
        """计算总体评级"""
        avg_chinese = (chinese_result['initial_coverage'] + chinese_result['final_coverage']) / 2
        avg_english = (english_result['vowel_coverage'] + english_result['consonant_coverage']) / 2
        overall = (avg_chinese + avg_english) / 2

        if overall >= 0.9:
            return "优秀 (Excellent)"
        elif overall >= 0.75:
            return "良好 (Good)"
        elif overall >= 0.6:
            return "及格 (Pass)"
        else:
            return "不足 (Insufficient)"

    def print_report(self, analysis_result: Dict, show_details: bool = False):
        """打印分析报告"""
        print("\n" + "="*70)
        print("音素覆盖度分析报告（基于拼音自动分析）")
        print("="*70)

        # 中文部分
        chinese = analysis_result['chinese']
        print("\n【中文音素覆盖】")
        print(f"分析字数: {chinese['char_count']} 个汉字")
        print(f"\n声母覆盖: {len(chinese['found_initials'])}/{len(self.chinese_initials)} "
              f"({chinese['initial_coverage']*100:.1f}%)")
        print(f"  ✅ 已覆盖: {', '.join(sorted(i or '零声母' for i in chinese['found_initials']))}")

        if chinese['missing_initials']:
            missing = [i or '零声母' for i in chinese['missing_initials']]
            print(f"  ❌ 缺失: {', '.join(missing)}")

        print(f"\n韵母覆盖: {len(chinese['found_finals'])}/{len(self.chinese_finals)} "
              f"({chinese['final_coverage']*100:.1f}%)")
        print(f"  ✅ 已覆盖: {', '.join(sorted(chinese['found_finals']))}")

        if chinese['missing_finals']:
            print(f"  ❌ 缺失: {', '.join(chinese['missing_finals'])}")

        # 详细拼音信息（可选）
        if show_details and 'pinyin_details' in chinese:
            print("\n【详细拼音分析】（前20个字示例）")
            for detail in chinese['pinyin_details'][:20]:
                print(f"  {detail['char']} → {detail['pinyin']} "
                      f"(声母: {detail['initial']}, 韵母: {detail['final']})")

        # 英文部分
        english = analysis_result['english']
        print("\n【英文CMU音素覆盖】")
        print(f"分析单词: {english['word_count']} 个")
        print(f"\n元音覆盖: {len(english['found_phonemes'] & self.cmu_vowels)}/{len(self.cmu_vowels)} "
              f"({english['vowel_coverage']*100:.1f}%)")
        print(f"辅音覆盖: {len(english['found_phonemes'] & self.cmu_consonants)}/{len(self.cmu_consonants)} "
              f"({english['consonant_coverage']*100:.1f}%)")

        if english['critical_missing']:
            print(f"\n  ⚠️  关键缺失音素: {', '.join(english['critical_missing'])}")
            print(f"  这些音素对英文发音至关重要！")

        # 总结
        summary = analysis_result['summary']
        print("\n" + "="*70)
        print(f"【总体评级】: {summary['overall_rating']}")
        print("="*70)

        # 建议
        print("\n【改进建议】")
        recommendations = []
        if chinese['missing_initials']:
            recommendations.append(f"1. 补充中文声母: {', '.join([i or '零声母' for i in chinese['missing_initials'][:5]])}")
        if chinese['missing_finals']:
            recommendations.append(f"2. 补充中文韵母: {', '.join(chinese['missing_finals'][:5])}")
        if english['critical_missing']:
            recommendations.append(f"3. 必须补充英文关键音素: {', '.join(english['critical_missing'])}")

        if recommendations:
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("  ✅ 音素覆盖度良好，无需额外补充")

    def export_missing_examples(self, analysis_result: Dict, output_file: str = "missing_phonemes.txt"):
        """导出缺失音素的示例句子建议"""
        with open(output_file, 'w', encoding='utf-8') as f:
            chinese = analysis_result['chinese']
            english = analysis_result['english']

            f.write("# 缺失音素补充建议\n\n")

            if chinese['missing_initials']:
                f.write("## 中文缺失声母\n\n")
                for initial in chinese['missing_initials']:
                    display = initial or '零声母'
                    f.write(f"### {display}\n")
                    f.write(f"建议添加包含 {display} 的句子\n\n")

            if chinese['missing_finals']:
                f.write("## 中文缺失韵母\n\n")
                for final in chinese['missing_finals']:
                    f.write(f"### {final}\n")
                    f.write(f"建议添加包含韵母 {final} 的句子\n\n")

            if english['critical_missing']:
                f.write("## 英文关键缺失音素\n\n")
                for phoneme in english['critical_missing']:
                    f.write(f"### {phoneme}\n")
                    if phoneme == 'TH':
                        f.write("示例: think, three, through\n\n")
                    elif phoneme == 'DH':
                        f.write("示例: this, that, father\n\n")
                    elif phoneme == 'NG':
                        f.write("示例: sing, long, king\n\n")

        print(f"\n✅ 缺失音素报告已保存到: {output_file}")


# 使用示例
if __name__ == "__main__":
    if not PYPINYIN_AVAILABLE:
        print("\n❌ 错误: 必须安装pypinyin才能使用此工具")
        print("安装命令: pip install pypinyin")
        exit(1)

    analyzer = PhonemeAnalyzer()

    # 分析现有文案
    print("正在分析 training_extend_data.json...")
    result = analyzer.analyze_coverage('training_extend_data.json', verbose=True)

    # 打印报告
    analyzer.print_report(result, show_details=True)

    # 导出缺失音素建议
    analyzer.export_missing_examples(result)
