## 验证test_data.json文件
import json

try:
    with open('d:/github/attack/data/test_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'文件格式正确，包含{len(data)}条数据记录')
except json.JSONDecodeError as e:
    print(f'JSON格式错误: {e}')
except Exception as e:
    print(f'发生错误: {e}')