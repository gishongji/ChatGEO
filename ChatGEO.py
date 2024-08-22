# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

# 创建一个标题和一个副标题
st.title("🌍 地形地貌智能问答助手")

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
    
    # 设定或添加填充token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 选择将`<eod>`作为填充token
        # 或者添加一个新的token作为填充token
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print("Creating model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()

    print("Done.")
    return tokenizer, model

# 加载model和tokenizer
tokenizer, model = get_model()

# 初次运行时，session_state中没有"messages"，需要创建一个空列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 添加地形地貌领域的上下文
    domain_context = "您是一位专业的地理学家，专门研究地形地貌。"

    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 拼接对话历史
    prompt = domain_context + "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    # 获取`input_ids`和`attention_mask`
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()

    outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=1024) # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '')

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)