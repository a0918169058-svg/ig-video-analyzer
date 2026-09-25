import streamlit as st
import time
import json
import os
import tempfile
import yt_dlp
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime

# 1. 網頁基本設定
st.set_page_config(page_title="靈感行動庫 (IG / FB / 短影音)", layout="wide")
st.title("📱 靈感行動庫 (IG / FB / 短影音)")

# 2. 本地資料儲存與讀取核心
DATA_FILE = "history.json"
saved_api_key = st.secrets.get("GEMINI_API_KEY", "")

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if "id" not in item:
                        item["id"] = f"item_{i}_{int(time.time())}"
                    if "is_done" not in item:
                        item["is_done"] = False
                    if "user_note" not in item:
                        item["user_note"] = ""
                    if "thumbnail" not in item:
                        item["thumbnail"] = ""
                    if "people_count" not in item:
                        item["people_count"] = "單人即可"
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

# 3. 定義結構化 AI 輸出格式
class VideoAnalysisResult(BaseModel):
    category: Literal["攝影技巧", "美食製作", "跳舞或搞笑cover", "其他"] = Field(
        description="影片分類"
    )
    people_count: Literal["單人即可", "雙人搭檔", "3~4人 (小團體)", "5人以上 (大陣仗)"] = Field(
        description="執行此內容或拍攝所需的人數（跳舞人數、攝影模特/掌鏡人數、或料理適合分食人數）"
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
        description="執行重點條列（美食：火候或關鍵步驟；跳舞：節奏卡點或動作要領；攝影：運鏡口訣或相機設置建議）"
    )
    analysis_reason: str = Field(
        description="總體分析與星級評定依據"
    )

# 4. 串流分析核心
def process_and_analyze(video_url: str, api_key: str) -> dict:
    clean_url = video_url.split("?si=")[0].split("&")[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_video_template = os.path.join(tmp_dir, "video.%(ext)s")
        
        # 允許 yt-dlp 自動選擇最合適格式，並自動處理音訊與影像
        ydl_opts = {
            'outtmpl': temp_video_template,
            'format': 'b/bestvideo+bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=True)
            video_title = info.get('title', '短影音')
            video_thumbnail = info.get('thumbnail', '')

        downloaded_files = os.listdir(tmp_dir)
        if not downloaded_files:
            raise ValueError("影片下載失敗，請確認該影片是否為公開貼文。")
        
        actual_path = os.path.join(tmp_dir, downloaded_files[0])

        client = genai.Client(api_key=api_key)
        video_file = client.files.upload(
            file=actual_path
        )

        while video_file.state.name != "ACTIVE":
            if video_file.state.name == "FAILED":
                raise ValueError("Google 伺服器處理該影片轉檔失敗，可能該影片受版權保護。")
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        prompt = """
        分析這段影片的內容，嚴格依據規則進行結構化拆解：
        1. 分類選項：攝影技巧、美食製作、跳舞或搞笑cover、其他。
        2. 評定所需人數：單人即可、雙人搭檔、3~4人 (小團體)、5人以上 (大陣仗)。請仔細看畫面中跳舞的人數、拍照姿勢需要幾人出鏡或掌鏡。
        3. 評定難易度（1 到 5 星，1 為新手能直接複製，5 為需專業功底）。
        4. 評定預估耗時。
        5. 拆解可執行的清單：
           - 若為「美食製作」：必須填寫成品名稱，並在 ingredients_or_props 列出影片中出現的食材備料，在 key_steps_or_tips 列出關鍵操作技巧。
           - 若為「跳舞或搞笑cover」：在 key_steps_or_tips 列出節奏卡點要領或動作記憶點。
           - 若為「攝影技巧」：在 key_steps_or_tips 提煉運鏡口訣或相機設置建議。
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
                time.sleep(15)

        client.files.delete(name=video_file.name)

        if not response:
            raise last_err

        result_dict = json.loads(response.text)
        result_dict["url"] = video_url
        result_dict["title"] = video_title[:40] if video_title else "未命名影片"
        result_dict["thumbnail"] = video_thumbnail
        result_dict["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return result_dict

# 5. 前端介面
tab_analyze, tab_library = st.tabs(["🔍 分析新影片", "📚 我的影片靈感庫"])

with tab_analyze:
    default_url = st.query_params.get("url", "")
    video_input = st.text_input("貼上 IG Reels / FB 影片 / Shorts 連結", value=default_url, placeholder="支援 IG、Facebook、Shorts 公開影片連結...")
    
    if st.button("開始分析並儲存", type="primary"):
        if not saved_api_key:
            st.error("系統尚未設定 GEMINI_API_KEY！")
        elif not video_input:
            st.warning("請先輸入影片網址！")
        else:
            with st.spinner("AI 正在深度解析人數/動作/食材與步驟重點..."):
                try:
                    result = process_and_analyze(video_input, saved_api_key)
                    save_to_history(result)
                    st.success("🎉 分析完成！已整理執行重點並存入靈感庫！")
                    
                    c_thumb, c_info = st.columns([1, 3])
                    with c_thumb:
                        if result.get("thumbnail"):
                            st.image(result["thumbnail"], use_container_width=True)
                    with c_info:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("分類", result["category"])
                        with c2:
                            st.metric("建議人數", result.get("people_count", "單人即可"))
                        with c3:
                            stars = "★" * result["difficulty_rating"] + "☆" * (5 - result["difficulty_rating"])
                            st.metric("難度", f"{stars} ({result['difficulty_rating']}/5)")
                        with c4:
                            st.metric("預估耗時", result.get("estimated_time", "未知"))
                        
                        if result["category"] == "美食製作" and result.get("dish_name"):
                            st.subheader(f"🍳 成品：{result['dish_name']}")

                    if result.get("ingredients_or_props"):
                        header = "🛒 食材備料清單" if result["category"] == "美食製作" else "🎒 必備道具/鏡頭"
                        st.markdown(f"**{header}**：")
                        st.write("、 ".join(result["ingredients_or_props"]))
                        if result["category"] == "美食製作":
                            st.caption("📋 採買備忘（右上角可一鍵複製）：")
                            st.code("\n".join(result["ingredients_or_props"]), language="text")

                    if result.get("key_steps_or_tips"):
                        st.markdown("**💡 關鍵執行要點與技巧：**")
                        for step in result["key_steps_or_tips"]:
                            st.markdown(f"- {step}")

                    with st.expander("🔍 綜合分析依據", expanded=False):
                        st.write(result["analysis_reason"])
                except Exception as e:
                    st.error(f"分析失敗：{e}")

with tab_library:
    records = load_history()
    if not records:
        st.info("靈感庫目前空空的，快去貼網址分析一部吧！")
    else:
        search_query = st.text_input("🔍 搜尋靈感庫（輸入食材、菜名、動作關鍵字...）", placeholder="例如：義大利麵、雞胸、運鏡...")

        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([2, 2, 2, 2, 2])
        with col_f1:
            category_filter = st.selectbox("📂 分類", ["全部", "美食製作", "跳舞或搞笑cover", "攝影技巧", "其他"])
        with col_f2:
            people_filter = st.selectbox("👥 人數需求", ["全部人數", "單人即可", "雙人搭檔", "3~4人 (小團體)", "5人以上 (大陣仗)"])
        with col_f3:
            time_filter = st.selectbox("⏱️ 耗時篩選", ["全部時間", "15分鐘以內 (快手)", "15~30分鐘 (日常)", "30~60分鐘 (精緻)", "1小時以上 (挑戰)"])
        with col_f4:
            status_filter = st.selectbox("📌 執行狀態", ["全部", "⏳ 待嘗試", "✅ 已完成"])
        with col_f5:
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

        if people_filter != "全部人數":
            filtered = [r for r in filtered if r.get("people_count", "單人即可") == people_filter]

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

        st.caption(f"共找到 {len(filtered)} 筆靈感")
        st.divider()

        for item in filtered:
            item_id = item.get("id")
            stars = "★" * item.get("difficulty_rating", 1) + "☆" * (5 - item.get("difficulty_rating", 1))
            time_tag = item.get("estimated_time", "時間未標")
            people_tag = item.get("people_count", "單人即可")
            is_done = item.get("is_done", False)
            thumb_url = item.get("thumbnail", "")

            with st.container():
                col_pic, col_main, col_side, col_del = st.columns([2, 5, 2, 1])

                with col_pic:
                    if thumb_url:
                        st.image(thumb_url, use_container_width=True)
                    else:
                        st.caption("（無封面預覽）")

                with col_main:
                    status_badge = "✅ [已完成]" if is_done else "⏳ [待嘗試]"
                    title_text = f"🍳 {item['dish_name']}" if item["category"] == "美食製作" and item.get("dish_name") else f"🎬 {item.get('category')}"
                    st.subheader(f"{status_badge} {title_text}")
                    
                    if item.get("ingredients_or_props"):
                        header = "🛒 食材備料" if item["category"] == "美食製作" else "🎒 道具/特點"
                        st.markdown(f"**{header}**：{'、 '.join(item['ingredients_or_props'])}")
                        
                        if item["category"] == "美食製作":
                            with st.expander("📋 一鍵複製採買清單", expanded=False):
                                st.caption("點擊右上方圖示即可複製：")
                                st.code("\n".join(item["ingredients_or_props"]), language="text")

                    if item.get("key_steps_or_tips"):
                        st.markdown("**重點步驟 / 技巧口訣**：")
                        for step in item["key_steps_or_tips"]:
                            st.markdown(f"- {step}")

                    current_note = item.get("user_note", "")
                    with st.expander("📝 我的實作筆記 / 心得", expanded=bool(current_note)):
                        note_input = st.text_area("個人筆記備忘", value=current_note, key=f"note_input_{item_id}")
                        if st.button("儲存筆記", key=f"btn_note_{item_id}"):
                            update_record_by_id(item_id, user_note=note_input)
                            st.success("筆記已更新！")
                            st.rerun()

                    st.caption(f"新增時間：{item.get('created_at', '未知')} | [🔗 開啟原影片]({item.get('url')})")

                with col_side:
                    st.markdown(f"**分類**：`{item.get('category')}`")
                    st.markdown(f"**人數**：`👥 {people_tag}`")
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

                with col_del:
                    if st.button("🗑️ 刪除", key=f"del_{item_id}"):
                        delete_record_by_id(item_id)
                        st.rerun()
                st.divider()
