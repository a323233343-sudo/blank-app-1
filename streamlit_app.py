import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="TSP 旅遊路線規劃", layout="wide")
st.title("🗺️ 智慧旅遊路線系統（RouteXL + Google Map 整合版）")

# -----------------------------
# CSV 上傳
# -----------------------------
uploaded_file = st.file_uploader("請上傳 .csv 座標檔", type=["csv"])

# -----------------------------
# 初始化 df 與 sidebar 選項
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({"name":[], "lat":[], "lon":[]})

if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✅ 上傳成功：{uploaded_file.name}")
        st.subheader("📄 CSV top 5 list：")
        st.dataframe(st.session_state.df.head())
    except pd.errors.EmptyDataError:
        st.error("❌ CSV 檔案為空或格式錯誤")
    except Exception as e:
        st.error(f"❌ 讀取 CSV 發生錯誤：{e}")

df = st.session_state.df

# -----------------------------
# Sidebar: 起點 / 終點 / 中途景點
# -----------------------------
st.sidebar.header("🧭 起點與終點設定")

start_point = st.sidebar.selectbox(
    "選擇起點",
    options=df["name"] if not df.empty else ["請先上傳 CSV"]
)
end_point = st.sidebar.selectbox(
    "選擇終點",
    options=df["name"] if not df.empty else ["請先上傳 CSV"]
)

st.sidebar.header("🏞️ 中途景點")
middle_points = st.sidebar.multiselect(
    "選擇想去的景點（可多選）",
    options=df["name"] if not df.empty else [],
    default=[x for x in df["name"] if x not in [start_point, end_point]] if not df.empty else []
)

# -----------------------------
# 顯示地圖與路線
# -----------------------------
if not df.empty and {"name","lat","lon"}.issubset(df.columns):
    selected_points = [start_point] + middle_points + [end_point]
    route_df = df[df["name"].isin(selected_points)]

    st.subheader("🌏 路線地圖")
    m = folium.Map(location=[route_df["lat"].mean(), route_df["lon"].mean()], zoom_start=13)
    coords = list(zip(route_df["lat"], route_df["lon"]))
    folium.PolyLine(coords, color="blue", weight=4, opacity=0.7).add_to(m)

    for i, row in enumerate(route_df.itertuples()):
        label = f"🏁 起點" if row.name == start_point else f"🎯 終點" if row.name == end_point else f"{i}. {row.name}"
        folium.Marker([row.lat, row.lon], popup=label, tooltip=row.name).add_to(m)

    st_folium(m, width=900, height=600)

    st.subheader("📋 路線順序")
    st.write(" → ".join(selected_points))

    st.download_button(
        label="💾 匯出 RouteXL 匯入格式 (CSV)",
        data=route_df.to_csv(index=False).encode("utf-8"),
        file_name="RouteXL_input.csv",
        mime="text/csv",
    )
else:
    st.info("⬆️ 請上傳 CSV 並確認包含欄位：`name`, `lat`, `lon`")
