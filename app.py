import streamlit as st
import assemblyai as aai
import google.generativeai as genai
import os
import time

# ==========================================
# 1. 🔑 从 Streamlit Secrets 读取 API Keys
# ==========================================
ASSEMBLYAI_KEY = st.secrets.get("ASSEMBLYAI_KEY", "你的_ASSEMBLYAI_KEY_本地测试用")
GEMINI_KEY = st.secrets.get("GEMINI_KEY", "你的_GEMINI_KEY_本地测试用")

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

st.set_page_config(page_title="会议转录与分析助手", layout="wide")
st.title("我的会议录音转录与分析助手 🎙️🌐")
st.write("已启用长音频切片引擎与自动语种检测，完美支持多语言长会议的无损分析。")

# ==========================================
# 2. 🎛️ 侧边栏高级设置区
# ==========================================
st.sidebar.header("⚙️ 参数配置")

# --- 新增：AssemblyAI 转录设置 ---
st.sidebar.subheader("🎙️ 转录设置 (AssemblyAI)")

# 映射字典：将用户的自然语言选择映射为代码参数
aai_model_map = {
    "Best (最高准确率)": aai.SpeechModel.best,
    "Nano (极速转录)": aai.SpeechModel.nano
}
aai_lang_map = {
    "自动检测 (推荐)": "auto",
    "中文": "zh",
    "英文 (全球)": "en",
    "英文 (美国)": "en_us",
    "日语": "ja",
    "法语": "fr",
    "德语": "de"
}

selected_aai_model_ui = st.sidebar.selectbox("选择转录模型：", options=list(aai_model_map.keys()), index=0)
selected_aai_lang_ui = st.sidebar.selectbox("录音原始语言：", options=list(aai_lang_map.keys()), index=0)

st.sidebar.markdown("---")

# --- 原有：Gemini 翻译设置 ---
st.sidebar.subheader("🌐 翻译与分析设置 (Gemini)")
enable_translation = st.sidebar.checkbox("开启 AI 翻译与分析", value=True)

if enable_translation:
    fetched_models = fetch_gemini_models(GEMINI_KEY)
    default_candidates = ["gemini-flash-latest", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    all_model_options = fetched_models if fetched_models else default_candidates

    default_index = 0
    if "gemini-flash-latest" in all_model_options:
        default_index = all_model_options.index("gemini-flash-latest")
    elif "gemini-1.5-flash" in all_model_options:
        default_index = all_model_options.index("gemini-1.5-flash")

    selected_model = st.sidebar.selectbox("选择 Gemini 模型：", options=all_model_options, index=default_index)
    target_language = st.sidebar.selectbox("翻译目标语言：", options=["英文", "中文", "日文", "德文", "法文", "韩文"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("附加生成项")
    enable_summary = st.sidebar.checkbox("生成会议总结", value=True)
    enable_action_items = st.sidebar.checkbox("生成待办事项", value=True)

    extra_context = st.sidebar.text_area(
        "补充背景信息 / 术语表 (可选)：",
        placeholder="例如：\n参会人员：Alice, Bob\n专业术语：ENSO, A320",
        height=120
    )

# ==========================================
# 3. 📂 主界面文件上传与处理
# ==========================================
uploaded_file = st.file_uploader("请上传音频文件 (最高 200MB，推荐先将长录音压缩为 64kbps MP3)", type=['mp3', 'm4a'])

if uploaded_file is not None:
    if st.button("🚀 开始处理"):
        with st.spinner("正在呼叫 AssemblyAI 进行语音转录，长音频可能需要 3-5 分钟，请勿刷新页面..."):
            temp_file_name = "temp_audio" + os.path.splitext(uploaded_file.name)[1]
            with open(temp_file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # --- 第一阶段：语音转文字配置 ---
                transcriber = aai.Transcriber()
                
                # 动态生成 AssemblyAI 配置
                aai_config_params = {
                    "speaker_labels": True,
                    "speech_model": aai_model_map[selected_aai_model_ui]
                }
                
                # 判断是否开启自动语种检测
                if aai_lang_map[selected_aai_lang_ui] == "auto":
                    aai_config_params["language_detection"] = True
                else:
                    aai_config_params["language_code"] = aai_lang_map[selected_aai_lang_ui]
                
                config = aai.TranscriptionConfig(**aai_config_params)
                
                # 发送转录请求
                transcript = transcriber.transcribe(temp_file_name, config=config)
                
                if transcript.error:
                    st.error(f"转录失败: {transcript.error}")
                else:
                    st.success("🎉 转录成功！正在进行文本梳理...")
                    
                    # 10 分钟切片逻辑
                    CHUNK_DURATION_MS = 10 * 60 * 1000 
                    chunks = []
                    current_chunk_text = ""
                    current_bin = -1
                    final_full_text = ""

                    for utterance in transcript.utterances:
                        bin_idx = utterance.start // CHUNK_DURATION_MS
                        if current_bin == -1:
                            current_bin = bin_idx
                        
                        if bin_idx > current_bin:
                            chunks.append(current_chunk_text)
                            current_chunk_text = ""
                            current_bin = bin_idx
                        
                        time_str = format_time(utterance.start)
                        line = f"[{time_str}] 发言人 {utterance.speaker}: {utterance.text}\n\n"
                        current_chunk_text += line
                        final_full_text += line
                    
                    if current_chunk_text:
                        chunks.append(current_chunk_text)

                    if not enable_translation:
                        st.markdown("### 📝 原始转录 (带时间戳)")
                        st.text_area("可全选复制 (Ctrl+A)：", value=final_full_text, height=600, key="raw_only")
                    
                    else:
                        model = genai.GenerativeModel(selected_model)
                        
                        # --- 第二阶段：全局总结与待办 ---
                        summary_text = ""
                        action_items_text = ""
                        if enable_summary or enable_action_items:
                            with st.spinner("正在生成全局会议概览..."):
                                if enable_summary:
                                    prompt_summary = f"基于以下会议转录，请用【{target_language}】写一段简明扼要的“会议核心总结 (Executive Summary)”。字数控制在 200 字以内。\n\n【会议记录原文】\n{final_full_text}"
                                    summary_text = model.generate_content(prompt_summary).text
                                
                                if enable_action_items:
                                    prompt_actions = f"基于以下会议转录，请用【{target_language}】列出明确的“待办事项与负责责任人 (Action Items)”。如果没有明确提及，请回答“暂无明确待办”。\n\n【会议记录原文】\n{final_full_text}"
                                    action_items_text = model.generate_content(prompt_actions).text

                        # --- 第三阶段：循环切片翻译 ---
                        st.write("### 🌐 开始分块逐句对照翻译")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        final_translated_text = ""
                        total_chunks = len(chunks)

                        for i, chunk_text in enumerate(chunks):
                            status_text.text(f"Gemini 正在处理第 {i+1}/{total_chunks} 块录音 (每块约 10 分钟)......")
                            
                            prompt_translate = []
                            if extra_context.strip():
                                prompt_translate.append(f"【背景信息与术语表】\n{extra_context.strip()}\n")
                            
                            prompt_translate.append(f"""你是一个专业的会议分析助手。请将下方这一小块会议记录翻译成【{target_language}】。
- 请严格采用【一句原文，紧接着一句译文】的逐句对照格式输出。
- 格式示例：
  [00:15] 发言人 A: 我们的计划是这样的。
  [00:15] 发言人 A (译文): Our plan is like this.

【会议记录片段原文】
{chunk_text}""")
                            
                            try:
                                response_trans = model.generate_content("\n\n".join(prompt_translate))
                                final_translated_text += response_trans.text + "\n\n"
                            except Exception as e:
                                final_translated_text += f"\n\n[⚠️ 第 {i+1} 块翻译由于网络或频率限制出现中断: {e}]\n\n"
                            
                            progress_bar.progress((i + 1) / total_chunks)
                            time.sleep(2)
                            
                        status_text.text("✅ 所有区块翻译完毕！")

                        # --- 第四阶段：UI 展示 ---
                        if enable_summary or enable_action_items:
                            st.markdown("### 📊 会议概览")
                            if enable_summary:
                                with st.expander("📌 会议核心总结 (Executive Summary)", expanded=True):
                                    st.write(summary_text)
                            if enable_action_items:
                                with st.expander("✅ 待办事项 (Action Items)", expanded=True):
                                    st.write(action_items_text)
                            st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 📝 完整原始转录")
                            st.text_area("可全选复制 (Ctrl+A)：", value=final_full_text, height=600, key="raw_col")
                            
                        with col2:
                            st.markdown(f"### 🌐 完整双语对照翻译 ({target_language})")
                            st.text_area("可全选复制 (Ctrl+A)：", value=final_translated_text, height=600, key="trans_col")
            
            except Exception as e:
                st.error(f"程序运行出了点小错: {e}")
                
            finally:
                if os.path.exists(temp_file_name):
                    os.remove(temp_file_name)
