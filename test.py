import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设置模型路径 (会自动从 HuggingFace 下载)
# 如果显存不够，改用 "Qwen/Qwen2.5-0.5B"
model_name = "Qwen/Qwen2.5-1.5B-Instruct" 

# 2. 自动判断设备 (支持 NVIDIA显卡, Mac M芯片, 或 CPU)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"正在使用设备: {device}")

# 3. 加载模型和分词器
print("正在下载/加载模型... (第一次运行会比较慢)")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto" if device == "cuda" else None, # 自动分配显存
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    trust_remote_code=True
).to(device)

# 4. 测试对话
prompt = "你好，请用一句话介绍什么是风控算法。"
messages = [
    {"role": "system", "content": "你是一个专业的算法工程师。"},
    {"role": "user", "content": prompt}
]

# 准备输入
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# 生成回答
print("\n=== Qwen 的回答 ===")
with torch.no_grad():
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 清理掉prompt部分，只显示回答
    print(response.split("assistant\n")[-1])