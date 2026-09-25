import streamlit as st
import io
import time
import json
import os
import urllib.request
import yt_dlp
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

# 1. 網頁基本設定
st.set_page_config(page_title="IG 影片靈感庫", layout="wide")
st.title("📱 我的 IG 靈感與分析庫")

# 2. 本地資料儲存與讀取
DATA_FILE = "history.json"
saved_api_key = st.secrets.get("GEMINI_API_KEY", "")

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(record):
    history = load_history()
    history.insert(0, record)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def delete_record(index):
    history = load_history()
    if 0 <= index < len(history):
        history.pop(index)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

# 3. 定義 AI 分析格式
class VideoAnalysisResult(BaseModel):
    category: Literal["攝影技巧", "美食製作", "跳舞或搞笑cover", "其他"] = Field(
        description="影片分類"
    )
    difficulty_rating: int = Field(
        description="難易度評分（1到5星）", ge=1, le=5
    )
    dish_name: Optional[str] = Field(
        default=None, 
        description="若分類為美食製作，提供成品名稱；其餘填 null"
    )
    analysis_reason: str = Field(
        description="給出分類與評分的具體分析原因"
    )

# 4. 記憶體串流分析核心
def process_and_analyze(ig_url: str, api_key: str) -> dict:
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(ig_url, download=False)
        video_direct_url = info.get('url')
        video_title = info.get('title', 'IG 影片')
        if not video_direct_url:
            raise ValueError("無法解析出影片串流直鏈，請確認該影片是否為公開貼文。")

    req = urllib.request.Request(
        video_direct_url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    video_buffer = io.BytesIO()
    with urllib.request.urlopen(req) as resp:
        video_buffer.write(resp.read())
    video_buffer.seek(0)

    client = genai.Client(api_key=api_key)
    video_file = client.files.upload(
        file=video_buffer,
        config={'mime_type': 'video/mp4'}
    )

    while video_file.state.name == "PROCESSING":
        time.sleep(1.5)
        video_file = client.files.get(name=video_file.name)

    prompt = """
    分析這段影片的內容，嚴格依據規則進行分類與難易度評定：
    1. 分類選項僅限：
       - "攝影技巧"（包含運鏡、構圖、打光、剪輯後製手法等）
       - "美食製作"（料理烹飪、烘焙、調飲等）
       - "跳舞或搞笑cover"（舞蹈跟跳、搞笑迷因模仿等）
       - 不符上述三者則歸為 "其他"
    2. 評定難易度（1 到 5 顆星，1 為一般新手可輕易完成，5 為需要專業設備或多年訓練）。
    3. 若為「美食製作」，必須給出具體的「成品名稱」；其他分類此欄位設為 null。
    """

    response = None
    last_err = None

    # 針對 gemini-3.8-flash 模型，若遇 503 伺服器忙線自動等待重試最多 3 次
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VideoAnalysisResult,
                    temperature=0.2,
                ),
            )
            if response:
                break
        except Exception as e:
            last_err = e
            time.sleep(3)

    client.files.delete(name=video_file.name)

    if not response:
        raise last_err

    result_dict = json.loads(response.text)
    result_dict["url"] = ig_url
    result_dict["title"] = video_title[:40] if video_title else "未命名影片"
    result_dict["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return result_dict

# 5. 前端頁面標籤架構
tab_analyze, tab_library = st.tabs(["🔍 分析新影片", "📚 我的影片靈感庫"])

with tab_analyze:
    ig_url = st.text_input("貼上 Instagram Reels / 影片連結", placeholder="https://www.instagram.com/reel/...")
    if st.button("開始分析並儲存", type="primary"):
        if not saved_api_key:
            st.error("系統尚未設定 GEMINI_API_KEY！")
        elif not ig_url:
            st.warning("請先輸入 IG 影片網址！")
        else:
            with st.spinner("AI 正在串流分析影片中（若遇忙線將自動重試）..."):
                try:
                    result = process_and_analyze(ig_url, saved_api_key)
                    save_to_history(result)
                    st.success("🎉 分析完成並已自動儲存至靈感庫！")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("影片分類", result["category"])
                    with c2:
                        stars = "★" * result["difficulty_rating"] + "☆" * (5 - result["difficulty_rating"])
                        st.metric("難易度", f"{stars} ({result['difficulty_rating']}/5)")
                    
                    if result["category"] == "美食製作" and result.get("dish_name"):
                        st.info(f"🍳 **料理成品名稱**：{result['dish_name']}")

                    with st.expander("🔍 分析依據與說明", expanded=True):
                        st.write(result["analysis_reason"])
                except Exception as e:
                    st.error(f"分析失敗：{e}")

with tab_library:
    records = load_history()
    if not records:
        st.info("靈感庫目前空空的，快去貼網址分析一部吧！")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            category_filter = st.selectbox(
                "📂 分類篩選", 
                ["全部", "美食製作", "跳舞或搞笑cover", "攝影技巧", "其他"]
            )
        with col_f2:
            sort_order = st.selectbox(
                "⭐ 難易度排序", 
                ["最新加入優先", "從最簡單開始 (★ ➔ ★★★★★)", "從高難度挑戰 (★★★★★ ➔ ★)"]
            )

        filtered = records if category_filter == "全部" else [r for r in records if r.get("category") == category_filter]

        if sort_order == "從最簡單開始 (★ ➔ ★★★★★)":
            filtered.sort(key=lambda x: x.get("difficulty_rating", 1))
        elif sort_order == "從高難度挑戰 (★★★★★ ➔ ★)":
            filtered.sort(key=lambda x: x.get("difficulty_rating", 1), reverse=True)

        st.caption(f"共找到 {len(filtered)} 筆紀錄")
        st.divider()

        for idx, item in enumerate(filtered):
            stars = "★" * item["difficulty_rating"] + "☆" * (5 - item["difficulty_rating"])
            with st.container():
                c1, c2, c3 = st.columns([4, 2, 1])
                with c1:
                    title_text = f"🍳 {item['dish_name']}" if item["category"] == "美食製作" and item.get("dish_name") else f"🎬 {item.get('category')}"
                    st.subheader(title_text)
                    st.write(f"**分析細節**：{item.get('analysis_reason')}")
                    st.caption(f"新增時間：{item.get('created_at', '未知')} | [🔗 開啟 IG 原影片]({item.get('url')})")
                with c2:
                    st.markdown(f"**類別**：`{item.get('category')}`")
                    st.markdown(f"**難度**：`{stars}` ({item['difficulty_rating']}/5)")
                with c3:
                    if st.button("🗑️ 刪除", key=f"del_{idx}"):
                        delete_record(idx)
                        st.rerun()
                st.divider()
