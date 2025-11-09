import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import random
import time

# ----------------------------
# NSGA-II class (your version, integrated)
# ----------------------------
class NSGAII_tsp:
    def ordered_crossover_fixed(self, p1, p2):
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b + 1] = p1[a:b + 1]
        pos = 0
        for i in range(n):
            idx = (b + 1 + i) % n
            if child[idx] != -1:
                continue
            while p2[pos] in child:
                pos += 1
            child[idx] = p2[pos]
        return child

    def swap_mutation_fixed(self, route, prob=0.2):
        r = route[:]
        if random.random() < prob:
            i, j = random.sample(range(len(r)), 2)
            r[i], r[j] = r[j], r[i]
        return r

    def dominates(self, a, b):
        le = all(x <= y for x, y in zip(a, b))
        lt = any(x < y for x, y in zip(a, b))
        return le and lt

    def fast_non_dominated_sort(self, pop_objs):
        S = [set() for _ in pop_objs]
        n_dom = [0] * len(pop_objs)
        fronts = [[]]
        for p in range(len(pop_objs)):
            for q in range(len(pop_objs)):
                if p == q:
                    continue
                if self.dominates(pop_objs[p], pop_objs[q]):
                    S[p].add(q)
                elif self.dominates(pop_objs[q], pop_objs[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        nxt.append(q)
            i += 1
            fronts.append(nxt)
        if not fronts[-1]:
            fronts.pop()
        return fronts

    def crowding_distance(self, front_objs):
        l = len(front_objs)
        if l == 0:
            return {}
        nobj = len(front_objs[0][1])
        dist = {idx: 0 for idx, _ in front_objs}
        for m in range(nobj):
            sorted_front = sorted(front_objs, key=lambda x: x[1][m])
            minv, maxv = sorted_front[0][1][m], sorted_front[-1][1][m]
            dist[sorted_front[0][0]] = dist[sorted_front[-1][0]] = float('inf')
            if maxv == minv:
                continue
            for i in range(1, l - 1):
                prevv, nextv = sorted_front[i - 1][1][m], sorted_front[i + 1][1][m]
                dist[sorted_front[i][0]] += (nextv - prevv) / (maxv - minv)
        return dist

    def tournament_selection(self, pop):
        a, b = random.sample(pop, 2)
        if a['rank'] < b['rank']:
            return a
        if a['rank'] > b['rank']:
            return b
        return a if a['cd'] > b['cd'] else b

    def enforce_order(self, route):
        # 順序約束：第13點(12) 要在第14點(13)之前
        precedence_rules = [(12, 13)]
        for a, b in precedence_rules:
            if a >= len(route) or b >= len(route):
                continue
            # only if both in route
            if a in route and b in route:
                ia, ib = route.index(a), route.index(b)
                if ia > ib:
                    route[ia], route[ib] = route[ib], route[ia]

                # 新增第13必須在倒數第二或倒數第三位
                ia = route.index(a)
                n = len(route)
                if ia < n - 3:
                    elem = route.pop(ia)
                    route.insert(n - 3, elem)
        return route

    def nsga2_tsp(self, D, T, coords=None, pop_size=80, gens=200, cx_prob=0.9, mut_prob=0.2, close_loop=False, start_idx=0, end_idx=None, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n = D.shape[0]

        def obj_distance(r):
            total = sum(D[r[i], r[i + 1]] for i in range(len(r) - 1))
            if close_loop:
                total += D[r[-1], r[0]]
            return total

        def obj_time(r):
            total = sum(T[r[i], r[i + 1]] for i in range(len(r) - 1))
            if close_loop:
                total += T[r[-1], r[0]]
            return total

        if end_idx is None:
            end_idx = n - 1

        # === 初始化族群 ===
        all_nodes = [i for i in range(n) if i not in [start_idx, end_idx]]
        population = []
        for _ in range(pop_size):
            middle = random.sample(all_nodes, len(all_nodes))
            route = [start_idx] + middle + [end_idx]
            route = self.enforce_order(route)
            population.append({'route': route, 'objs': None})

        def evaluate(pop):
            for ind in pop:
                ind['objs'] = (obj_distance(ind['route']), obj_time(ind['route']))

        evaluate(population)

        for gen in range(gens):
            pop_objs = [ind['objs'] for ind in population]
            fronts = self.fast_non_dominated_sort(pop_objs)
            for i, f in enumerate(fronts):
                for idx in f:
                    population[idx]['rank'] = i
            for f in fronts:
                f_objs = [(idx, population[idx]['objs']) for idx in f]
                cd = self.crowding_distance(f_objs)
                for idx in f:
                    population[idx]['cd'] = cd.get(idx, 0)

            offspring = []
            while len(offspring) < pop_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                child = self.ordered_crossover_fixed(p1['route'][1:-1], p2['route'][1:-1]) if random.random() < cx_prob else p1['route'][1:-1][:]
                child = self.swap_mutation_fixed(child, mut_prob)
                child = self.enforce_order(child)
                # ✅ 保留起終點
                child = [start_idx] + child + [end_idx]
                offspring.append({'route': child, 'objs': None})

            evaluate(offspring)

            combined = population + offspring
            comb_objs = [ind['objs'] for ind in combined]
            fronts = self.fast_non_dominated_sort(comb_objs)
            new_pop = []
            for f in fronts:
                if len(new_pop) + len(f) <= pop_size:
                    for idx in f:
                        new_pop.append(combined[idx])
                else:
                    f_objs = [(idx, combined[idx]['objs']) for idx in f]
                    cd = self.crowding_distance(f_objs)
                    f_sorted = sorted(f, key=lambda i: cd.get(i, 0), reverse=True)
                    remain = pop_size - len(new_pop)
                    for idx in f_sorted[:remain]:
                        new_pop.append(combined[idx])
                    break
            population = new_pop

            # progress print for debugging - in streamlit we'll show spinner instead
            if (gen + 1) % 50 == 0:
                best_dist = min(ind['objs'][0] for ind in population)
                print(f"Gen {gen + 1}/{gens}: best_distance={best_dist:.2f}")

        # final pareto front (first front)
        fronts = self.fast_non_dominated_sort([ind['objs'] for ind in population])
        pareto = [population[i] for i in fronts[0]]
        return pareto

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="TSP 旅遊路線規劃", layout="wide")
st.title("🗺️ 智慧旅遊路線系統")

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
    folium.PolyLine(coords, color="blue", weight=4, opacity=1).add_to(m)
    for i, row in enumerate(route_df.itertuples()):
        label = f"🏁 起點" if row.name == start_point else f"🎯 終點" if row.name == end_point else f"{i}. {row.name}"
        folium.Marker([row.lat, row.lon], popup=label, tooltip=row.name).add_to(m)
    st_folium(m, width=900, height=700)

    st.subheader("📋 現選路線順序（使用者選擇順序）")
    st.write(" → ".join(selected_points))

    st.markdown("---")
    st.subheader("🚀 使用 NSGA-II 進行路線最佳化（多目標：距離 + 時間）")
    
    st.subheader("距離矩陣與時間矩陣範例格式說明")
    data = {
    "name": ["高鐵", "左營孔子廟", "大港橋", "公園二路(集合)", "晚餐(鹽埕區夜市)", "鹽埕區鹽埕國民小學(集合)"],
    "高鐵": [0, 3.5, 9.8, 9.2, 9.7, 9.8],
    "左營孔子廟": [3.5, 0, 9.8, 9.4, 8.9, 8.9],
    "大港橋": [9.8, 9.8, 0, 0.45, 1, 0.85],
    "公園二路(集合)": [9.2, 9.4, 0.45, 0, 0.55, 0.4],
    "晚餐(鹽埕區夜市)": [9.7, 8.9, 1, 0.55, 0, 0.6],
    "鹽埕區鹽埕國民小學(集合)": [9.8, 8.9, 0.85, 0.4, 0.6, 0]
    }
    example_df = pd.DataFrame(data)
    

    with st.expander("📄 距離/時間矩陣格式範例（點擊展開/收合）"):
        st.markdown("距離矩陣與時間矩陣的 CSV 檔案應包含以下格式：")
        st.dataframe(example_df)
        st.markdown(f"""
        其中第一欄為地點名稱，後續欄位為各地點之間的距離或時間（單位可自行定義，如公里或分鐘）。
        注意：距離矩陣與時間矩陣中的地點名稱必須與您在左側選擇的起點、終點及中途景點一致。
        任何不在您選擇清單中的地點將自動從矩陣中移除。
        例如，若您選擇的地點為：{', '.join(example_df['name'].values[random.sample(range(len(example_df)), 4)])}，則矩陣中應僅包含這些地點的資料。
        """)

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

    # -----------------------------
    # 🔄 根據當前 middle_points 自動同步 D, T
    # -----------------------------
    if not st.session_state.dist_df.empty or not st.session_state.time_df.empty:
        selected_points = [start_point] + middle_points + [end_point]

        def filter_matrix(df, label):
            if df.empty:
                return df
            # 移除未選擇的點
            not_included = set(df['name']) - set(selected_points)
            if not_included:
                st.warning(f"⚠️ {label} 中包含未選擇的點，將自動移除：{', '.join(not_included)}")
            df = df[df['name'].isin(selected_points)]
            columns_to_keep = ['name'] + [p for p in selected_points if p in df.columns]
            df = df[columns_to_keep]
            return df

        D = filter_matrix(st.session_state.dist_df.copy(), "距離矩陣")
        T = filter_matrix(st.session_state.time_df.copy(), "時間矩陣")

        st.session_state.dist_df = D
        st.session_state.time_df = T

        st.subheader("📄 距離矩陣預覽：")
        st.dataframe(D)

        st.subheader("📄 時間矩陣預覽：")
        st.dataframe(T)

    run_btn = st.button("執行 NSGA-II 最佳化")

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
            nsga = NSGAII_tsp()
            st.info("開始執行 NSGA-II，請稍候... 可能需要一些時間（依 iter 與 pop_size 而定）")
            
            start_time = time.time()
            
            idx_map = {name: i for i, name in enumerate(selected_points)}
            start_idx = idx_map[start_point]
            end_idx = idx_map[end_point]

            nsga = NSGAII_tsp()
            pareto = nsga.nsga2_tsp(
                D, T, coords=coords, pop_size=pop_size, gens=gens,
                cx_prob=cx_prob, mut_prob=mut_prob,
                close_loop=close_loop,
                start_idx=start_idx, end_idx=end_idx,
                seed=seed_val if seed_val != 0 else None
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
        st.download_button("💾 下載最佳路線 (CSV)", 
                         best_df.to_csv(index=False).encode("utf-8"),
                         file_name="best_route.csv", 
                         mime="text/csv")
        
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
            }
            for idx, p in enumerate(sorted(results['pareto'], key=lambda x: x['objs'][0]))
        ]).drop_duplicates(subset=['Route'], keep='first')  # 只保留不重複的路線

        # 重新設定 Route_ID
        pareto_summary['Route_ID'] = range(1, len(pareto_summary) + 1)
        st.dataframe(pareto_summary)

        # 提供下載
        csv_bytes = pareto_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 下載所有 Pareto 路線 (CSV)",
            csv_bytes,
            file_name="pareto_routes.csv",
            mime="text/csv"
        )
else:
    st.info("⬆️ 請上傳 CSV 並確認包含欄位：`name`, `lat`, `lon`")
