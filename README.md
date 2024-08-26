# ChatGEO
基于大模型的地理知识问答助手

---
[toc]

---

## 前言

此次开发尝试是为了完成datawhale夏令营中的“大模型应用开发”实践，代码参考了官方的demo，只是自己稍作修改，不具备真正意义上的“地理知识库”，仅仅通过预先调整提示词对大模型的输出进行一定控制，本质还是Yuan 2-2B Mars-HF对话大模型。

希望以后可以逐步加上RAG/LoRA等方法，继续完善应用，并搜集地形地貌知识重新训练模型，完成一个真正的地理知识问答大语言模型。

感谢datawhale与魔搭社区的各种支持！

---

## 项目背景

随着地理科学的进步和技术的发展，人们对地形地貌的研究越来越深入。然而，传统的研究方法往往依赖于大量的实地考察和文献查阅，这不仅耗时耗力，而且效率较低。为此，我们开发了一款基于大模型的地形地貌智能问答助手，旨在利用人工智能技术提升地形地貌研究的效率和质量。

---

## 产品功能

本产品能够实现以下主要功能：

1. **智能问答**：用户可以通过自然语言向助手提问有关地形地貌的问题，获得快速且准确的回答。
2. **领域知识集成**：内置专业领域知识库，能够针对地形地貌的相关问题给出专业解答。
3. **交互式对话**：支持多轮对话，能够理解上下文并做出相应的回答。
4. **实时反馈**：用户提出问题后，系统能够迅速响应并给出答案。

---

## 应用价值

- **教育科研**：为地理学教育和科研工作提供辅助工具，帮助学生和研究人员更好地理解和学习地形地貌知识。
- **决策支持**：为政府和企事业单位提供地形地貌方面的决策依据，辅助规划和管理。
- **公众普及**：提高公众对地理环境的认识水平，促进环境保护意识的形成。

---

## 技术方案

为了实现上述功能，我们采用了以下技术方案：

- **模型选择**：使用IEIT Yuan 2-2B Mars-HF模型，该模型基于Transformer架构，具有强大的自然语言处理能力。
- **框架支持**：采用Streamlit框架构建前端界面，实现用户友好的交互体验。
- **编程语言**：主要使用Python语言进行开发。
- **部署环境**：可在本地服务器或云平台上部署。

---

## 方案架构

1. **前端界面**：使用Streamlit搭建的用户界面，包括输入框、消息历史展示区等。
2. **后端逻辑**：包括模型加载、输入处理、生成回答等功能。
3. **模型层**：预训练的大规模语言模型，用于生成高质量的回答。

---

## 核心代码
代码已经加上详细注释：
```python
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
```

---

## 代码结构

1. **导入必需模块**：从`transformers`库导入`AutoTokenizer`和`AutoModelForCausalLM`类，以及从`torch`库导入`torch`，还有`streamlit`库用于创建用户界面。
2. **前端界面搭建**：使用`streamlit`创建基本的UI元素，例如标题和输入框。
3. **模型与Tokenizer的加载**：定义一个函数`get_model`来加载模型和tokenizer，并缓存这个过程以避免重复加载。
4. **模型推理**：定义逻辑来处理用户输入，生成模型的输出，并显示结果。

### 功能与实现逻辑

#### 1. 导入必需模块

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
```

- **功能**：导入所需的库和模块。
- **实现逻辑**：这些库提供了模型加载、模型推理以及用户界面创建所需的基本功能。

#### 2. 前端界面搭建

```python
st.title("🌍 地形地貌智能问答助手")
```

- **功能**：设置应用的标题。
- **实现逻辑**：使用`streamlit`的`title`函数来定义应用的主要标题。

#### 3. 模型与Tokenizer的加载

```python
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
path = model_dir

@st.cache_resource
def get_model():
    # ... (省略部分代码)
    return tokenizer, model
```

- **功能**：下载模型文件，加载模型和tokenizer。

- **实现逻辑**：
  - 使用`modelscope`的`snapshot_download`函数下载模型文件。
  - 使用`@st.cache_resource`装饰器确保模型只被加载一次。
  - `AutoTokenizer`用于加载tokenizer，`AutoModelForCausalLM`用于加载模型。
  - 扩展tokenizer以支持额外的特殊token。

#### 4. 模型推理

```python
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ... (省略部分代码)

if prompt := st.chat_input():
    # ... (省略部分代码)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # ... (省略部分代码)
    
    # 拼接对话历史
    prompt = domain_context + "\n".join(msg["content"] for msg in st.session_state.messages) + "<sep>"
    # ... (省略部分代码)
    
    # 使用模型生成回答
    outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=1024)
    output = tokenizer.decode(outputs[0])
    # ... (省略部分代码)
```

- **功能**：处理用户输入，使用模型生成回答，并显示结果。

- **实现逻辑**：
  - 使用`st.session_state`来存储和维护对话历史。
  - 用户通过`st.chat_input()`输入问题。
  - 输入的问题被添加到会话状态(`st.session_state`)中。
  - 输入问题前加入领域上下文。
  - 使用tokenizer对输入进行编码，并将编码后的数据移至GPU。
  - 调用模型的`generate`方法生成回答。
  - 对模型输出进行解码，并提取有效回答。
  - 将模型生成的回答显示在界面上。
#### 5. 加载模型和tokenizer

```python
def get_model():
    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    # 扩展tokenizer，增加特殊token
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)
    
    # 设定或添加填充token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 选择将`<eod>`作为填充token
    
    print("Creating model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()
    
    print("Done.")
    return tokenizer, model
```

- **功能**：加载模型和tokenizer。

- **实现逻辑**：
- 使用`AutoTokenizer.from_pretrained`加载tokenizer。
  - 扩展tokenizer以支持额外的特殊token。
- 使用`AutoModelForCausalLM.from_pretrained`加载模型，并将其移动到GPU上。

#### 6. 处理用户输入和生成回答

```python
if prompt := st.chat_input():
    # ... (省略部分代码)
    
    # 拼接对话历史
    prompt = domain_context + "\n".join(msg["content"] for msg in st.session_state.messages) + "<sep>"
    
    # 对输入进行编码
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    
    # 将输入数据移到GPU上
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    
    # 使用模型生成回答
    outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=1024)
    output = tokenizer.decode(outputs[0])
    
    # 解析模型的输出
    response = output.split("<sep>")[-1].replace("<eod>", '')
    
    # ... (省略部分代码)
```

- **功能**：处理用户输入，使用模型生成回答。
- **实现逻辑**：
  - 拼接用户输入和对话历史。
  - 使用tokenizer对输入进行编码。
  - 将输入数据移动到GPU上。
  - 使用模型的`generate`方法生成回答。
  - 解码模型输出，并提取有效回答。

---

## 运行效果与问题

参考“效果图”文件夹，问答助手可以有效的回答地理知识，并认为自己是回答地理知识的专家。对于“亚马逊，喜马拉雅“这样有多重含义的定义，会优先解释其地理学相关知识，且具备大模型的一般对话功能。

![效果图-问答效果1](https://github.com/gishongji/ChatGEO/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE/%E6%95%88%E6%9E%9C%E5%9B%BE-%E9%97%AE%E7%AD%94%E6%95%88%E6%9E%9C1.png)

但在”待优化情况“中，可以看到，模型存在”忘记“自己”身份“的情况。在这种情况下提问时，”亚马逊“就会以公司的情况被用作答案，而不是地理学概念。这种情况是有待于通过RAG LoRA等方法优化模型来实现的，也是本项目今后要考虑的内容。

---

## 远期计划

- **功能扩展**：增加更多地理相关的功能，如配图、提供参考链接等。
- **多语言支持**：未来版本将支持多种语言，满足不同地区用户的需求。
- **持续优化**：根据用户反馈持续改进模型性能和用户体验。

---

## 市场思考

- **目标用户**：主要面向地理学研究者、教师、学生以及对地形地貌感兴趣的公众。
- **市场需求**：目前市场上缺乏专业的地形地貌问答助手，存在较大的市场空间。
- **竞争优势**：结合最新的AI技术和专业的领域知识，提供高质量的服务。

---

## 推广策略

- **学术合作**：与高校和研究机构合作，共同举办研讨会和培训活动。
- **社交媒体宣传**：通过微博、知乎等平台分享使用案例和心得，吸引更多潜在用户。
- **免费试用**：提供一定期限的免费试用，鼓励用户尝试并留下反馈。
- **合作伙伴**：与地理信息相关的企业建立合作关系，共同推广产品。

---

未完待续······