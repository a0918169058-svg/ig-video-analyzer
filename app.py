import streamlit as st
import io
import time
import json
import urllib.request
import yt_dlp
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime

# 1. 網頁基本設定
st.set_page_config(page_title="IG 靈感行動庫 (雲端同步版)", layout="wide")
st.title("📱 我的 IG 靈感行動庫")

# 2. 連接 Google Sheets 資料庫
conn = st.connection("gsheets", type=GSheetsConnection)

def load_history():
    try:
        df = conn.read(ttl="0s")
        if df is None or df.empty:
            return []
        
        # 轉換成乾淨的字典列表
        records = []
        for _, row in df.iterrows():
            if pd.isna(row.get("id")):
                continue
            
            # 解析 list 欄位
            props_raw = str(row.get("ingredients_or_props", ""))
            tips_raw = str(row.get("key_steps_or_tips", ""))
            
            try:
                props = json.loads(props_raw) if props_raw.startswith("[") else ([p.strip() for p in props_raw.split("、")] if props_raw else [])
            except Exception:
                props = [props_raw] if props_raw else []

            try:
                tips = json.loads(tips_raw) if tips_raw.startswith("[") else ([t.strip() for t in tips_raw.split("\n")] if tips_raw else [])
            except Exception:
                tips = [tips_raw] if tips_raw else []

            records.append({
                "id": str(row.get("id", "")),
                "category": str(row.get("category", "其他")),
                "difficulty_rating": int(row.get("difficulty_rating", 1)) if pd.notna(row.get("difficulty_rating")) else 1,
                "estimated_time": str(row.get("estimated_time", "未知")),
                "dish_name": str(row.get("dish_name", "")) if pd.notna(row.get("dish_name")) else "",
                "ingredients_or_props": props,
                "key_steps_or_tips": tips,
                "analysis_reason": str(row.get("analysis_reason", "")),
                "url": str(row.get("url", "")),
                "created_at": str(row.get("created_at", "")),
                "is_done": bool(row.get("is_done", False)) if pd.notna(row.get("is_done")) else False,
                "user_note": str(row.get("user_note", "")) if pd.notna(row.get("user_note")) else "",
            })
        return records
    except Exception as e:
        st.error(f"讀取試算表失敗，請確認共用權限與 Secrets 設定：{e}")
        return []

def save_all_records_to_sheets(records):
    rows = []
    for r in records:
        rows.append({
            "id": r.get("id"),
            "category": r.get("category"),
            "difficulty_rating": r.get("difficulty_rating"),
            "estimated_time": r.get("estimated_time"),
            "dish_name": r.get("dish_name") or "",
            "ingredients_or_props": json.dumps(r.get("ingredients_or_props", []), ensure_ascii=False),
            "key_steps_or_tips": json.dumps(r.get("key_steps_or_tips", []), ensure_ascii=False),
            "analysis_reason": r.get("analysis_reason") or "",
            "url": r.get("url") or "",
            "created_at": r.get("created_at") or "",
            "is_done": r.get("is_done", False),
            "user_note": r.get("user_note") or "",
        })
    df_new = pd.DataFrame(rows)
    conn.update(data=df_new)

def save_to_history(record):
    records = load_history()
    record["id"] = f"item_{int(time.time() * 1000)}"
    record["is_done"] = False
    record["user_note"] = ""
    records.insert(0, record)
    save_all_records_to_sheets(records)

def delete_record_by_id(record_id):
    records = load_history()
    records = [r for r in records if r.get("id") != record_id]
    save_all_records_to_sheets(records)

def update_record_by_id(record_id, is_done=None, user_note=None):
    records = load_history()
    for r in records:
        if r.get("id") == record_id:
            if is_done is not None:
                r["is_done"] = is_done
            if user_note is not None:
                r["user_note"] = user_note
            break
    save_all_records_to_sheets(records)

# 3. 定義 AI 格式
saved_api_key = st.secrets.get("GEMINI_API_KEY", "")

class VideoAnalysisResult(BaseModel):
    category: Literal["攝影技巧", "美食製作", "跳舞或搞笑cover", "其他"] = Field(description="影片分類")
    difficulty_rating: int = Field(description="難易度評分（1到5星）", ge=1, le=5)
    estimated_time: Literal["15分鐘以內 (快手)", "15~30分鐘 (日常)", "30~60分鐘 (精緻)", "1小時以上 (挑戰)"] = Field(description="預估完成時間")
    dish_name: Optional[str] = Field(default=None, description="美食成品名稱")
    ingredients_or_props: List[str] = Field(default_factory=list, description="食材清單或必備道具")
    key_steps_or_tips: List[str] = Field(default_factory=list, description="關鍵步驟或口訣")
    analysis_reason: str = Field(description="分析說明")

# 4. 串流分析核心
def process_and_analyze(ig_url: str, api_key: str) -> dict:
    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(ig_url, download=False)
        video_direct_url = info.get('url')
        video_title = info.get('title', 'IG 影片')
        if not video_direct_url:
            raise ValueError("無法解析出影片直鏈，請確認為公開貼文。")

    req = urllib.request.Request(video_direct_url, headers={'User-Agent': 'Mozilla/5.0'})
    video_buffer = io.BytesIO()
    with urllib.request.urlopen(req) as resp:
        video_buffer.write(resp.read())
    video_buffer.seek(0)

    client = genai.Client(api_key=api_key)
    video_file = client.files.upload(file=video_buffer, config={'mime_type': 'video/mp4'})

    while video_file.state.name == "PROCESSING":
        time.sleep(1.5)
        video_file = client.files.get(name=video_file.name)

    prompt = """
    分析這段影片內容，嚴格依據規則進行結構化拆解：
    1. 分類選項：攝影技巧、美食製作、跳舞或搞笑cover、其他。
    2. 評定難易度（1 到 5 星）。
    3. 評定預估耗時。
    4. 拆解可執行的清單：
       - 美食製作：給出成品名稱、食材備料清單(ingredients_or_props)、關鍵操作技巧(key_steps_or_tips)。
       - 跳舞或搞笑cover：列出節奏卡點要領或動作記憶點。
       - 攝影技巧：提煉運鏡口訣或相機設置建議。
    """

    response = None
    last_err = None
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

# 5. 前端頁面
tab_analyze, tab_library = st.tabs(["🔍 分析新影片", "📚 我的影片靈感庫"])

with tab_analyze:
    ig_url = st.text_input("貼上 Instagram Reels / 影片連結", placeholder="https://www.instagram.com/reel/...")
    if st.button("開始分析並儲存至 Google 試算表", type="primary"):
        if not saved_api_key:
            st.error("尚未設定 GEMINI_API_KEY！")
        elif not ig_url:
            st.warning("請輸入 IG 影片網址！")
        else:
            with st.spinner("AI 深度解析中並同步至雲端試算表..."):
                try:
                    result = process_and_analyze(ig_url, saved_api_key)
                    save_to_history(result)
                    st.success("🎉 分析完成！已永久同步至您的 Google 試算表！")
                    
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
                except Exception as e:
                    st.error(f"分析失敗：{e}")

with tab_library:
    records = load_history()
    if not records:
        st.info("靈感庫目前空空的，快去貼網址分析一部吧！")
    else:
        search_query = st.text_input("🔍 搜尋靈感庫（輸入食材、菜名、動作關鍵字...）", placeholder="例如：義大利麵、雞胸、運鏡...")

        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 2])
        with col_f1:
            category_filter = st.selectbox("📂 分類", ["全部", "美食製作", "跳舞或搞笑cover", "攝影技巧", "其他"])
        with col_f2:
            time_filter = st.selectbox("⏱️ 耗時篩選", ["全部時間", "15分鐘以內 (快手)", "15~30分鐘 (日常)", "30~60分鐘 (精緻)", "1小時以上 (挑戰)"])
        with col_f3:
            status_filter = st.selectbox("📌 執行狀態", ["全部", "⏳ 待嘗試", "✅ 已完成"])
        with col_f4:
            sort_order = st.selectbox("⭐ 排序方式", ["最新加入優先", "從最簡單開始 (★ ➔ ★★★★★)", "從高難度挑戰 (★★★★★ ➔ ★)"])

        filtered = records
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

        if category_filter != "全部":
            filtered = [r for r in filtered if r.get("category") == category_filter]

        if time_filter != "全部時間":
            filtered = [r for r in filtered if r.get("estimated_time") == time_filter]

        if status_filter == "⏳ 待嘗試":
            filtered = [r for r in filtered if not r.get("is_done", False)]
        elif status_filter == "✅ 已完成":
            filtered = [r for r in filtered if r.get("is_done", False)]

        if sort_order == "從最簡單開始 (★ ➔ ★★★★★)":
            filtered.sort(key=lambda x: x.get("difficulty_rating", 1))
        elif sort_order == "從高難度挑戰 (★★★★★ ➔ ★)":
            filtered.sort(key=lambda x: x.get("difficulty_rating", 1), reverse=True)

        st.caption(f"共找到 {len(filtered)} 筆靈感（已與 Google 試算表即時連線）")
        st.divider()

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

                    current_note = item.get("user_note", "")
                    with st.expander("📝 我的實作筆記 / 心得", expanded=bool(current_note)):
                        note_input = st.text_area("個人筆記備忘", value=current_note, key=f"note_input_{item_id}")
                        if st.button("儲存筆記至試算表", key=f"btn_note_{item_id}"):
                            update_record_by_id(item_id, user_note=note_input)
                            st.success("筆記已同步更新至 Google 試算表！")
                            st.rerun()

                    st.caption(f"新增時間：{item.get('created_at', '未知')} | [🔗 開啟 IG 原影片]({item.get('url')})")

                with c2:
                    st.markdown(f"**分類**：`{item.get('category')}`")
                    st.markdown(f"**難度**：`{stars}` ({item.get('difficulty_rating', 1)}/5)")
                    st.markdown(f"**耗時**：`⏱️ {time_tag}`")
                    
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
