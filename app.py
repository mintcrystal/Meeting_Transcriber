import streamlit as st
import assemblyai as aai
import google.generativeai as genai
import os

# ==========================================
# 1. 🔑 从 Streamlit Secrets 读取 API Keys
# ==========================================
# 当应用部署在 Streamlit Cloud 时，它会自动从后台 Secrets 读取
# 当你在本地运行测试时，它会读取本地的 .streamlit/secrets.toml 文件
ASSEMBLYAI_KEY = st.secrets["ASSEMBLYAI_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]

aai.settings.api_key = ASSEMBLYAI_KEY
genai.configure(api_key=GEMINI_KEY)

def format_time(ms):
    """将毫秒转换成 HH:MM:SS 或 MM:SS 的格式"""
    seconds = ms // 1000
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

@st.cache_data(ttl=3600)
def fetch_gemini_models(api_key):
    """动态获取 Gemini 支持 generateContent 的可用模型列表"""
    try:
        genai.configure(api_key=api_key)
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                available_models.append(name)
        return available_models
    except Exception:
        return []

# 页面基本设置（宽屏模式）
st.set_page_config(page_title="会议转录与分析助手", layout="wide")

st.title("我的会议录音转录与分析助手 🎙️🌐")
st.write("上传音频文件，系统将使用 AssemblyAI 进行转录。您可以选择调用 Gemini API 进行双语对照、会议总结与待办提取。")

# ==========================================
# 2. 🎛️ 侧边栏高级设置区
# ==========================================
st.sidebar.header("⚙️ 参数配置")

# 是否开启翻译的主开关
enable_translation = st.sidebar.checkbox("开启 AI 翻译与分析", value=True, help="如果只需要原始文字，请取消勾选。")

if enable_translation:
    st.sidebar.markdown("---")
    
    fetched_models = fetch_gemini_models(GEMINI_KEY)
    default_candidates = ["gemini-flash-latest", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    all_model_options = fetched_models if fetched_models else default_candidates

    default_index = 0
    if "gemini-flash-latest" in all_model_options:
        default_index = all_model_options.index("gemini-flash-latest")
    elif "gemini-1.5-flash" in all_model_options:
        default_index = all_model_options.index("gemini-1.5-flash")

    selected_model = st.sidebar.selectbox(
        "选择 Gemini 模型：",
        options=all_model_options,
        index=default_index
    )

    target_language = st.sidebar.selectbox(
        "翻译目标语言：",
        options=["英文", "中文", "日文", "德文", "法文", "韩文"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("附加生成项")
    
    # 拆分为独立的总结和待办按钮
    enable_summary = st.sidebar.checkbox("生成会议总结 (Executive Summary)", value=True)
    enable_action_items = st.sidebar.checkbox("生成待办事项 (Action Items)", value=True)

    extra_context = st.sidebar.text_area(
        "补充背景信息 / 术语表 (可选)：",
        placeholder="例如：\n参会人员：Alice, Bob\n专业术语：ENSO (厄尔尼诺-南方涛动), A320",
        height=120
    )

# ==========================================
# 3. 📂 主界面文件上传与处理
# ==========================================
uploaded_file = st.file_uploader("请上传音频文件 (最高 200MB，支持 mp3, m4a)", type=['mp3', 'm4a'])

if uploaded_file is not None:
    if st.button("🚀 开始处理"):
        with st.spinner("正在呼叫 AssemblyAI 进行语音转录，请耐心等待..."):
            temp_file_name = "temp_audio" + os.path.splitext(uploaded_file.name)[1]
            with open(temp_file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # --- 第一阶段：语音转文字 ---
                transcriber = aai.Transcriber()
                config = aai.TranscriptionConfig(speaker_labels=True, language_code="zh")
                transcript = transcriber.transcribe(temp_file_name, config=config)
                
                if transcript.error:
                    st.error(f"转录失败: {transcript.error}")
                else:
                    final_text = ""
                    for utterance in transcript.utterances:
                        time_str = format_time(utterance.start)
                        final_text += f"[{time_str}] 发言人 {utterance.speaker}: {utterance.text}\n\n"
                    
                    st.success("🎉 转录成功！")
                    
                    if not enable_translation:
                        st.markdown("### 📝 原始转录 (带时间戳)")
                        st.text_area("可全选复制 (Ctrl+A)：", value=final_text, height=600, key="raw_only")
                    
                    else:
                        with st.spinner(f"Gemini ({selected_model}) 正在生成分析与对照翻译..."):
                            model = genai.GenerativeModel(selected_model)
                            
                            # 构建主要翻译 Prompt
                            prompt_translate = []
                            if extra_context.strip():
                                prompt_translate.append(f"【背景信息与术语表】\n{extra_context.strip()}\n")
                            
                            prompt_translate.append(f"""你是一个专业的会议分析助手。请按照以下要求处理下方带时间戳的会议记录：
1. 翻译要求：
- 请将会议记录翻译成【{target_language}】。
- 请严格采用【一句原文，紧接着一句译文】的逐句对照格式输出。
- 格式示例：
  [00:15] 发言人 A: 我们的计划是这样的。
  [00:15] 发言人 A (译文): Our plan is like this.
- 如果上面提供了背景信息或术语表，请在翻译中准确应用其中的专业词汇和人名。

【会议记录原文】
{final_text}""")
                            full_prompt_trans = "\n\n".join(prompt_translate)
                            response_trans = model.generate_content(full_prompt_trans)
                            translated_text = response_trans.text

                            # --- 第 3.5 阶段：分别生成总结和待办 ---
                            summary_text = ""
                            if enable_summary:
                                prompt_summary = f"基于以下会议转录，请用【{target_language}】写一段简明扼要的“会议核心总结 (Executive Summary)”。字数控制在 200 字以内。\n\n【会议记录原文】\n{final_text}"
                                response_summary = model.generate_content(prompt_summary)
                                summary_text = response_summary.text
                            
                            action_items_text = ""
                            if enable_action_items:
                                prompt_actions = f"基于以下会议转录，请用【{target_language}】列出明确的“待办事项与负责责任人 (Action Items)”。如果没有明确提及，请回答“暂无明确待办”。\n\n【会议记录原文】\n{final_text}"
                                response_actions = model.generate_content(prompt_actions)
                                action_items_text = response_actions.text

                            # --- 第四阶段：UI 展示 ---
                            
                            # 顶部：如果有总结或待办，就分别显示在折叠面板（expander）中
                            if enable_summary or enable_action_items:
                                st.markdown("### 📊 会议概览")
                                if enable_summary:
                                    with st.expander("📌 会议核心总结 (Executive Summary)", expanded=True):
                                        st.write(summary_text)
                                if enable_action_items:
                                    with st.expander("✅ 待办事项 (Action Items)", expanded=True):
                                        st.write(action_items_text)
                                st.markdown("---")
                            
                            # 底部：双栏展示转录和翻译
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### 📝 原始转录")
                                st.text_area("可全选复制 (Ctrl+A)：", value=final_text, height=600, key="raw_col")
                                
                            with col2:
                                st.markdown(f"### 🌐 双语对照翻译 ({target_language})")
                                st.text_area("可全选复制 (Ctrl+A)：", value=translated_text, height=600, key="trans_col")
            
            except Exception as e:
                st.error(f"程序运行出了点小错: {e}")
                
            finally:
                if os.path.exists(temp_file_name):
                    os.remove(temp_file_name)