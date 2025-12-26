import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import time
from NSGA import NSGAII_tsp

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="TSP 旅遊路線規劃", layout="wide")
st.title("🗺️ 智慧旅遊路線系統")
main_tab1, main_tab2 = st.tabs(["📍 路線規劃", " 🏞️ 景點資訊"])
with main_tab1:
    st.markdown("""
    本系統使用 **NSGA-II 多目標最佳化演算法**，協助使用者規劃最佳旅遊路線。
    使用者可上傳景點座標 CSV 檔案，選擇起點、終點及中途景點，並設定 NSGA-II 參數。
    系統將根據距離與時間兩個目標，找出多條最佳路線供使用者參考。
    """)
    # -----------------------------
    # CSV 上傳
    # -----------------------------
    uploaded_file = st.file_uploader("請上傳 .csv 座標檔（需含 name, lat, lon）", type=["csv"])
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
    # Sidebar: 起點 / 終點 / 中途景點 & NSGA settings
    # -----------------------------
    st.sidebar.header("🧭 起點與終點設定")
    if df.empty:
        st.sidebar.info("請先上傳 CSV")
    start_point = st.sidebar.selectbox("選擇起點", options=df["name"] if not df.empty else ["請先上傳 CSV"])
    end_point = st.sidebar.selectbox("選擇終點", options=df["name"] if not df.empty else ["請先上傳 CSV"])

    st.sidebar.header("🏞️ 中途景點")
    middle_points = st.sidebar.multiselect(
        "選擇想去的景點（可多選）",
        options=df["name"] if not df.empty else [],
        default=[x for x in df["name"] if x not in [start_point, end_point]] if not df.empty else []
    )

    st.sidebar.header("⚙️ NSGA-II 參數")
    st.sidebar.markdown(
        """
        <label title="每一代中的個體數量。越大代表探索空間越廣，但運算時間也越長。">
            🧬 族群大小 (pop_size)
        </label>
        """,
        unsafe_allow_html=True
    )

    pop_size = st.sidebar.number_input("", min_value=10, max_value=500, value=80, step=10)
    st.sidebar.markdown(
        """
        <label title="演算法進化的迭代次數。越多的迭代可能找到更好的解，但會增加計算時間。">
            🔁 迭代次數 (iter)
        </label>
        """,
        unsafe_allow_html=True
    )
    gens = st.sidebar.number_input("", min_value=10, max_value=2000, value=200, step=10)
    cx_prob = st.sidebar.slider("交配機率 (cx_prob)", 0.0, 1.0, 0.9)
    mut_prob = st.sidebar.slider("突變機率 (mut_prob)", 0.0, 1.0, 0.2)
    close_loop = st.sidebar.checkbox("封閉回到起點 (close loop)", value=False)
    seed_val = st.sidebar.number_input("隨機種子 (0 表示不固定)", value=0, step=1)

    # -----------------------------
    # 顯示地圖與按鈕執行 NSGA-II
    # -----------------------------
    if not df.empty and {"name","lat","lon"}.issubset(df.columns):
        selected_points = [start_point] + middle_points + [end_point]
        route_df = df[df["name"].isin(selected_points)].reset_index(drop=True)

        st.subheader("🌏 現選路線地點（按順序顯示）")
        m = folium.Map(location=[route_df["lat"].mean(), route_df["lon"].mean()], zoom_start=13)
        coords = list(zip(route_df["lat"], route_df["lon"]))
        folium.PolyLine(coords, color="blue", weight=4, opacity=0.7).add_to(m)
        for i, row in enumerate(route_df.itertuples()):
            label = f"🏁 起點" if row.name == start_point else f"🎯 終點" if row.name == end_point else f"{i}. {row.name}"
            folium.Marker([row.lat, row.lon], popup=label, tooltip=row.name).add_to(m)
        st_folium(m, width=900, height=700)

        st.subheader("📋 現選路線順序（使用者選擇順序）")
        st.write(" → ".join(selected_points))

        st.markdown("---")
        st.subheader("🚀 使用 NSGA-II 進行路線最佳化（多目標：距離 + 時間）")

        uploaded_Dist_file = st.file_uploader("請上傳 .csv 距離矩陣檔", type=["csv"])
        if "dist_df" not in st.session_state:
            st.session_state.dist_df = pd.DataFrame()

        if uploaded_Dist_file is not None:
            try:
                st.session_state.dist_df = pd.read_csv(uploaded_Dist_file)
                # 只保留在 middle_points 中的點
                selected_points = [start_point] + middle_points + [end_point]
                not_included = set(st.session_state.dist_df['name']) - set(selected_points)
                if not_included:
                    st.warning(f"⚠️ 距離矩陣中包含未選擇的點，將自動移除：{', '.join(not_included)}")
                    # 移除不需要的行和列
                    st.session_state.dist_df = st.session_state.dist_df[
                        st.session_state.dist_df['name'].isin(selected_points)
                    ]
                    # 只保留選擇點的列
                    columns_to_keep = ['name'] + selected_points
                    st.session_state.dist_df = st.session_state.dist_df[columns_to_keep]
                st.success(f"✅ 上傳成功：{uploaded_Dist_file.name}")
            except pd.errors.EmptyDataError:
                st.error("❌ CSV 檔案為空或格式錯誤")
            except Exception as e:
                st.error(f"❌ 讀取 CSV 發生錯誤：{e}")

        uploaded_Time_file = st.file_uploader("請上傳 .csv 時間矩陣檔", type=["csv"])
        if "time_df" not in st.session_state:
            st.session_state.time_df = pd.DataFrame()

        if uploaded_Time_file is not None:
            try:
                st.session_state.time_df = pd.read_csv(uploaded_Time_file)
                # 只保留在 middle_points 中的點
                selected_points = [start_point] + middle_points + [end_point]
                not_included = set(st.session_state.time_df['name']) - set(selected_points)
                if not_included:
                    st.warning(f"⚠️ 時間矩陣中包含未選擇的點，將自動移除：{', '.join(not_included)}")
                    # 移除不需要的行和列
                    st.session_state.time_df = st.session_state.time_df[
                        st.session_state.time_df['name'].isin(selected_points)
                    ]
                    # 只保留選擇點的列
                    columns_to_keep = ['name'] + selected_points
                    st.session_state.time_df = st.session_state.time_df[columns_to_keep]
                st.success(f"✅ 上傳成功：{uploaded_Time_file.name}")
            except pd.errors.EmptyDataError:
                st.error("❌ CSV 檔案為空或格式錯誤")
            except Exception as e:
                st.error(f"❌ 讀取 CSV 發生錯誤：{e}")

        D = st.session_state.dist_df
        st.subheader("📄 距離矩陣預覽：")
        st.dataframe(D)

        T = st.session_state.time_df
        st.subheader("📄 時間矩陣預覽：")
        st.dataframe(T)
        st.markdown("---")

        run_btn = st.button("執行 NSGA-II 最佳化")
        if D.empty or T.empty:
            st.warning("請先上傳距離矩陣和時間矩陣的 CSV 檔案。")
        else:
            start_idx = D.index[D['name'] == start_point][0]  # 取第一個符合的索引
            end_idx = D.index[D['name'] == end_point][0]  # 取第一個符合的索引
        if run_btn:
            # 初始化 session state 變數
            if 'optimization_results' not in st.session_state:
                st.session_state.optimization_results = None
                
            # 執行最佳化並儲存結果
            with st.spinner("演算法運行中..."):
                # 準備距離矩陣與時間矩陣（使用歐氏距離）
                coords = list(zip(route_df["lat"].astype(float), route_df["lon"].astype(float)))
                n = len(coords)
                # 轉為 numpy array（忽略第一欄 name）
                if D.empty or 'name' not in D.columns:
                    st.warning("⚠️ 未上傳距離矩陣，將自動以歐氏距離計算")
                    D_mat = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            D_mat[i, j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                    T_mat = D_mat.copy()
                else:
                    D_mat = D.drop(columns=['name'], errors='ignore').to_numpy(dtype=float)
                    T_mat = T.drop(columns=['name'], errors='ignore').to_numpy(dtype=float)

                # 對應：route_df 的第 k 個點 對應 nsga2 使用的索引 k (0..n-1)
                # 執行 NSGA-II
                
                nsga = NSGAII_tsp(start_idx=start_idx, end_idx=end_idx)
                st.info("開始執行 NSGA-II，請稍候... 可能需要一些時間（依 gens 與 pop_size 而定）")
                start_time = time.time()
                pareto = nsga.nsga2_tsp(
                    D_mat, T_mat, coords=coords,
                    pop_size=int(pop_size), gens=int(gens),
                    cx_prob=float(cx_prob), mut_prob=float(mut_prob),
                    close_loop=bool(close_loop),
                    seed=(None if seed_val == 0 else int(seed_val))
                )
                elapsed = time.time() - start_time
                # 把 pareto routes 由索引轉回名稱/座標
                for idx, p in enumerate(pareto):
                    p['route_names'] = [route_df.iloc[i]['name'] for i in p['route']]
                    p['route_coords'] = [coords[i] for i in p['route']]

                # 儲存結果到 session state
                st.session_state.optimization_results = {
                    'pareto': pareto,
                    'best': min(pareto, key=lambda x: x['objs'][0]),
                    'elapsed': elapsed,
                    'coords': coords
                }
        
        # 如果有最佳化結果，顯示它們
        if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
            results = st.session_state.optimization_results
            
            st.success(f"完成！共找到 {len(results['pareto'])} 個 Pareto 解，耗時 {results['elapsed']:.1f} 秒")
            
            # 顯示最佳路線
            best = results['best']
            st.subheader("🏆 示範最佳路線（以總距離最短為準）")
            st.write("總距離：", best['objs'][0], " 總時間：", best['objs'][1])
            st.write("路線：", " → ".join(best['route_names']))
            
            # 顯示地圖
            m2 = folium.Map(location=[np.mean([c[0] for c in best['route_coords']]),
                                    np.mean([c[1] for c in best['route_coords']])], zoom_start=13)
            folium.PolyLine(best['route_coords'] + ([best['route_coords'][0]] if close_loop else []),
                            color="red", weight=4, opacity=0.8).add_to(m2)
            for i, (name, (lat, lon)) in enumerate(zip(best['route_names'], best['route_coords'])):
                label = "🏁 起點" if i == 0 else ("🎯 終點" if i == len(best['route_names'])-1 else f"{i}. {name}")
                folium.Marker([lat, lon], popup=label, tooltip=name).add_to(m2)
            st_folium(m2, width=900, height=700)

            # 下載按鈕
            best_df = pd.DataFrame({
                "order": list(range(1, len(best['route_names']) + 1)),
                "name": best['route_names'],
                "lat": [c[0] for c in best['route_coords']],
                "lon": [c[1] for c in best['route_coords']]
            })
            st.download_button(
                "💾 下載最佳路線 (CSV)",
                best_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="best_route.csv",
                mime="text/csv"
            )
            
            # Pareto 解顯示
            st.subheader("🧭 所有 Pareto 最佳解")
            
            # 顯示 Pareto map
            m3 = folium.Map(location=[np.mean([c[0] for c in results['coords']]),
                                        np.mean([c[1] for c in results['coords']])], zoom_start=13)
            
            colors = [
                "#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#00FFFF",
                "#0000FF", "#8B00FF", "#FF1493", "#20B2AA", "#808000"
            ]
            for idx, p in enumerate(results['pareto']):
                route_coords = p['route_coords']
                color = colors[idx % len(colors)]
                folium.PolyLine(
                    route_coords + ([route_coords[0]] if close_loop else []),
                    color=color, weight=3, opacity=0.6,
                    tooltip=f"Route {idx+1} | dist={p['objs'][0]:.2f}, time={p['objs'][1]:.2f}"
                ).add_to(m3)
            
            # 起點終點 marker
            folium.Marker(
                best['route_coords'][0],
                icon=folium.Icon(color="green", icon="play"),
                popup="起點"
            ).add_to(m3)
            folium.Marker(
                best['route_coords'][-1],
                icon=folium.Icon(color="red", icon="stop"),
                popup="終點"
            ).add_to(m3)
            
            st_folium(m3, width=900, height=700)
            
            # 顯示 Pareto front table
            st.write("📊 Pareto 路線摘要（依距離排序）")
            pareto_summary = pd.DataFrame([
                {
                    "Route_ID": idx + 1,
                    "Distance": p['objs'][0],
                    "Time": p['objs'][1],
                    "Route": " → ".join(p['route_names']),
                }
                for idx, p in enumerate(sorted(results['pareto'], key=lambda x: x['objs'][0]))
            ]).drop_duplicates(subset=['Route'], keep='first')  # 只保留不重複的路線

            # 重新設定 Route_ID
            pareto_summary['Route_ID'] = range(1, len(pareto_summary) + 1)
            st.dataframe(pareto_summary)

            # 提供下載
            csv_bytes = pareto_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "💾 下載所有 Pareto 路線 (CSV)",
                csv_bytes,
                file_name="pareto_routes.csv",
                mime="text/csv"
            )
    else:
        st.info("⬆️ 請上傳 CSV 並確認包含欄位：`name`, `lat`, `lon`")

with main_tab2:
    st.markdown("""
    ## 🏞️ 景點資訊

    在此頁面，您可以查看各個景點的詳細資訊，包括名稱、位置和其他相關描述。
    
    請確保在上傳的 CSV 檔案中包含必要的欄位，以便系統能夠正確顯示景點資訊。
    """)
    if st.session_state.df.empty:
        st.subheader("📍 景點列表")
    else:
        st.subheader("📍 景點列表")
        st.dataframe(st.session_state.df)