if __name__ == '__main__':
    import json

    # 合并两个文案
    with open('training_data.json', 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open('training_extend_data.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    complete = data1 + data2

    with open('training_complete.json', 'w', encoding='utf-8') as f:
        json.dump(complete, f, ensure_ascii=False, indent=2)

    print(f"✅ 合并完成！共 {len(complete)} 个section")
