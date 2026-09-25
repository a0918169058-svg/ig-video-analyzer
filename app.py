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
from typing import Optional, Literal, List
from datetime import datetime

# 1. 網頁基本設定
st.set_page_config(page_title="IG 靈感行動庫", layout="wide")
st.title("📱 我的 IG 靈感行動庫")

# 2. 本地資料儲存與讀取
DATA_FILE = "history.json"
saved_api_key = st.secrets.get("GEMINI_API_KEY", "")

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 向下相容，確保每筆資料都有 id、is_done 與 user_note
                for i, item in enumerate(data):
                    if "id" not in item:
                        item["id"] = f"item_{i}_{int(time.time())}"
                    if "is_done" not in item:
                        item["is_done"] = False
                    if "user_note" not in item:
                        item["user_note"] = ""
                return data
        except Exception:
            return []
    return []

def save_all_history(records):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def save_to_history(record):
    history = load_history()
    record["id"] = f"item_{int(time.time() * 1000)}"
    record["is_done"] = False
    record["user_note"] = ""
    history.insert(0, record)
    save_all_history(history)

def delete_record_by_id(record_id):
    history = load_history()
    new_history = [r for r in history if r.get("id") != record_id]
    save_all_history(new_history)

def update_record_by_id(record_id, is_done=None, user_note=None):
    history = load_history()
    for r in history:
        if r.get("id") == record_id:
            if is_done is not None:
                r["is_done"] = is_done
            if user_note is not None:
                r["user_note"] = user_note
            break
    save_all_history(history)

# 3. 定義升級版 AI 結構化資料格式
class VideoAnalysisResult(BaseModel):
    category: Literal["攝影技巧", "美食製作", "跳舞或搞笑cover", "其他"] = Field(
        description="影片分類"
    )
    difficulty_rating: int = Field(
        description="難易度評分（1到5星）", ge=1, le=5
    )
    estimated_time: Literal["15分鐘以內 (快手)", "15~30分鐘 (日常)", "30~60分鐘 (精緻)", "1小時以上 (挑戰)"] = Field(
        description="預估完成此內容所需的時間"
    )
    dish_name: Optional[str] = Field(
        default=None, 
        description="若為美食製作，提供成品名稱；其餘填 null"
    )
    ingredients_or_props: List[str] = Field(
        default_factory=list,
        description="若為美食，列出主要食材配料；若為攝影/跳舞，列出所需道具、鏡頭或服裝特點"
    )
    key_steps_or_tips: List[str] = Field(
        default_factory=list,
        description="執行重點條列（美食：火候或關鍵步驟；跳舞：節奏卡點或動作要領；攝影：運鏡口訣或相機設置）"
    )
    analysis_reason: str = Field(
        description="總體分析與星級評定依據"
    )

# 4. 核心分析函式
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
    分析這段影片的內容，嚴格依據規則進行結構化拆解：
    1. 分類選項：
       - "攝影技巧"（運鏡、打光、構圖、特效）
       - "美食製作"（家常菜、甜品、烘焙、調飲等）
       - "跳舞或搞笑cover"（舞蹈跟跳、迷因模仿等）
       - 其他
    2. 評定難易度（1 到 5 星，1 為新手能直接複製，5 為需專業功底）。
    3. 評定預估耗時（從選項中挑選最符合的一項）。
    4. 拆解可執行的清單：
       - 若為「美食製作」：必須填寫成品名稱，並在 ingredients_or_props 列出影片中出現的食材備料，在 key_steps_or_tips 列出 2~4 點關鍵操作技巧。
       - 若為「跳舞或搞笑cover」：在 key_steps_or_tips 列出節奏卡點要領、動作記憶點或表演亮點。
       - 若為「攝影技巧」：在 key_steps_or_tips 提煉運鏡口訣或相機設置建議。
    """

    response = None
    last_err = None

    # 重試機制抗 503 尖峰
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

# === Tab 1: 分析新影片 ===
with tab_analyze:
    ig_url = st.text_input("貼上 Instagram Reels / 影片連結", placeholder="https://www.instagram.com/reel/...")
    if st.button("開始分析並儲存", type="primary"):
        if not saved_api_key:
            st.error("系統尚未設定 GEMINI_API_KEY！")
        elif not ig_url:
            st.warning("請先輸入 IG 影片網址！")
        else:
            with st.spinner("AI 正在深度解析動作/食材與步驟重點..."):
                try:
                    result = process_and_analyze(ig_url, saved_api_key)
                    save_to_history(result)
                    st.success("🎉 分析完成！已整理執行重點並存入靈感庫！")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("分類", result["category"])
                    with c2:
                        stars = "★" * result["difficulty_rating"] + "☆" * (5 - result["difficulty_rating"])
                        st.metric("難度", f"{stars} ({result['difficulty_rating']}/5)")
                    with c3:
                        st.metric("預估耗時", result.get("estimated_time", "未知"))
                    
                    if result["category"] == "美食製作" and result.get("dish_name"):
                        st.subheader(f"🍳 成品：{result['dish_name']}")

                    if result.get("ingredients_or_props"):
                        header = "🛒 食材備料清單" if result["category"] == "美食製作" else "🎒 必備道具/鏡頭"
                        st.markdown(f"**{header}**：")
                        st.write("、 ".join(result["ingredients_or_props"]))

                    if result.get("key_steps_or_tips"):
                        st.markdown("**💡 關鍵執行要點與技巧：**")
                        for step in result["key_steps_or_tips"]:
                            st.markdown(f"- {step}")

                    with st.expander("🔍 綜合分析依據", expanded=False):
                        st.write(result["analysis_reason"])
                except Exception as e:
                    st.error(f"分析失敗：{e}")

# === Tab 2: 我的靈感庫（升級版檢索與狀態追蹤）===
with tab_library:
    records = load_history()
    if not records:
        st.info("靈感庫目前空空的，快去貼網址分析一部吧！")
    else:
        # 搜尋框與篩選列
        search_query = st.text_input("🔍 搜尋靈感庫（輸入食材、菜名、動作關鍵字...）", placeholder="例如：義大利麵、雞胸、運鏡...")

        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 2])
        with col_f1:
            category_filter = st.selectbox(
                "📂 分類", 
                ["全部", "美食製作", "跳舞或搞笑cover", "攝影技巧", "其他"]
            )
        with col_f2:
            time_filter = st.selectbox(
                "⏱️ 耗時篩選",
                ["全部時間", "15分鐘以內 (快手)", "15~30分鐘 (日常)", "30~60分鐘 (精緻)", "1小時以上 (挑戰)"]
            )
        with col_f3:
            status_filter = st.selectbox(
                "📌 執行狀態",
                ["全部", "⏳ 待嘗試", "✅ 已完成"]
            )
        with col_f4:
            sort_order = st.selectbox(
                "⭐ 排序方式", 
                ["最新加入優先", "從最簡單開始 (★ ➔ ★★★★★)", "從高難度挑戰 (★★★★★ ➔ ★)"]
            )

        # 執行多條件過濾
        filtered = records

        # 1. 關鍵字搜尋過濾
        if search_query.strip():
            sq = search_query.strip().lower()
            filtered = [
                r for r in filtered
                if sq in str(r.get("dish_name", "")).lower()
                or sq in str(r.get("category", "")).lower()
                or any(sq in item.lower() for item in r.get("ingredients_or_props", []))
                or any(sq in step.lower() for step in r.get("key_steps_or_tips", []))
                or sq in str(r.get("analysis_reason", "")).lower()
                or sq in str(r.get("user_note", "")).lower()
            ]

        # 2. 分類過濾
        if category_filter != "全部":
            filtered = [r for r in filtered if r.get("category") == category_filter]

        # 3. 耗時過濾
        if time_filter != "全部時間":
            filtered = [r for r in filtered if r.get("estimated_time") == time_filter]

        # 4. 狀態過濾
        if status_filter == "⏳ 待嘗試":
            filtered = [r for r in filtered if not r.get("is_done", False)]
        elif status_filter == "✅ 已完成":
            filtered = [r for r in filtered if r.get("is_done", False)]

        # 5. 排序
        if sort_order == "從最簡單開始 (★ ➔ ★★★★★)":
            filtered.sort(key=lambda x: x.get("difficulty_rating", 1))
        elif sort_order == "從高難度挑戰 (★★★★★ ➔ ★)":
            filtered.sort(key=lambda x: x.get("difficulty_rating", 1), reverse=True)

        st.caption(f"共找到 {len(filtered)} 筆靈感")
        st.divider()

        # 展示卡片列表
        for item in filtered:
            item_id = item.get("id")
            stars = "★" * item.get("difficulty_rating", 1) + "☆" * (5 - item.get("difficulty_rating", 1))
            time_tag = item.get("estimated_time", "時間未標")
            is_done = item.get("is_done", False)

            with st.container():
                c1, c2, c3 = st.columns([5, 2, 1])
                with c1:
                    status_badge = "✅ [已完成]" if is_done else "⏳ [待嘗試]"
                    title_text = f"🍳 {item['dish_name']}" if item["category"] == "美食製作" and item.get("dish_name") else f"🎬 {item.get('category')}"
                    st.subheader(f"{status_badge} {title_text}")
                    
                    if item.get("ingredients_or_props"):
                        header = "🛒 食材備料" if item["category"] == "美食製作" else "🎒 道具/特點"
                        st.caption(f"**{header}**：{'、 '.join(item['ingredients_or_props'])}")

                    if item.get("key_steps_or_tips"):
                        st.markdown("**重點步驟 / 技巧口訣**：")
                        for step in item["key_steps_or_tips"]:
                            st.markdown(f"- {step}")

                    # 心得筆記區塊
                    current_note = item.get("user_note", "")
                    with st.expander("📝 我的實作筆記 / 心得", expanded=bool(current_note)):
                        note_input = st.text_area("個人筆記備忘", value=current_note, key=f"note_input_{item_id}", placeholder="例如：下次鹽放少一點、這首舞蹈第二個八拍手部要注意卡點...")
                        if st.button("儲存筆記", key=f"btn_note_{item_id}"):
                            update_record_by_id(item_id, user_note=note_input)
                            st.success("筆記已更新！")
                            st.rerun()

                    st.caption(f"新增時間：{item.get('created_at', '未知')} | [🔗 開啟 IG 原影片]({item.get('url')})")

                with c2:
                    st.markdown(f"**分類**：`{item.get('category')}`")
                    st.markdown(f"**難度**：`{stars}` ({item.get('difficulty_rating', 1)}/5)")
                    st.markdown(f"**耗時**：`⏱️ {time_tag}`")
                    
                    # 狀態打勾切換按鈕
                    if is_done:
                        if st.button("標記為 ⏳ 待嘗試", key=f"undo_{item_id}"):
                            update_record_by_id(item_id, is_done=False)
                            st.rerun()
                    else:
                        if st.button("標記為 ✅ 已完成", key=f"done_{item_id}"):
                            update_record_by_id(item_id, is_done=True)
                            st.rerun()

                with c3:
                    if st.button("🗑️ 刪除", key=f"del_{item_id}"):
                        delete_record_by_id(item_id)
                        st.rerun()
                st.divider()
