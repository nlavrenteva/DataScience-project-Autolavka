import argparse
import math
import csv
import sys
import time
from pathlib import Path
 

def load_csv(path: str) -> list[dict]:

    points = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["@lat"])
                lon = float(row["@lon"])
                name = row.get("name", "").strip()
                if not name:
                    continue
                points.append({
                    "name":     name,
                    "lat":      lat,
                    "lon":      lon,
                    "type":     row.get("place", ""),
                    "region":   row.get("addr:region", ""),
                    "district": row.get("addr:district", ""),
                })
            except (ValueError, KeyError):
                continue
    return points
 
 
def filter_points(
    points: list[dict],
    region:   str | None = None,
    district: str | None = None,
    place_type: str | None = None,
) -> list[dict]:
    result = points
    if region:
        result = [p for p in result if p["region"] == region]
    if district:
        result = [p for p in result if p["district"] == district]
    if place_type:
        result = [p for p in result if p["type"] == place_type]
    # убираем точки без координат 
    return result
 

 
def haversine(a: dict, b: dict) -> float:
    R = 6371.0
    dlat = math.radians(b["lat"] - a["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a["lat"]))
         * math.cos(math.radians(b["lat"]))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
 
 
def total_distance(route: list[dict]) -> float:
    return sum(haversine(route[i], route[i + 1]) for i in range(len(route) - 1))
 
 

 
def nearest_neighbor(points: list[dict], start_idx: int = 0) -> list[dict]:

    n = len(points)
    visited = [False] * n
    route = [points[start_idx]]
    visited[start_idx] = True
 
    for _ in range(n - 1):
        cur = route[-1]
        best_d = math.inf
        best_j = -1
        for j in range(n):
            if not visited[j]:
                d = haversine(cur, points[j])
                if d < best_d:
                    best_d = d
                    best_j = j
        visited[best_j] = True
        route.append(points[best_j])
 
    return route
 
 
def two_opt(route: list[dict], max_iter: int = 5000) -> list[dict]:

    best = route[:]
    improved = True
    iterations = 0
 
    while improved and iterations < max_iter:
        improved = False
        n = len(best)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d_before = haversine(best[i - 1], best[i]) + haversine(best[j], best[(j + 1) % n])
                d_after  = haversine(best[i - 1], best[j]) + haversine(best[i], best[(j + 1) % n])
                if d_after < d_before - 1e-6:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
            iterations += 1
 
    return best
 
 
def find_best_start(points: list[dict], candidates: int = 5) -> list[dict]:

    n = len(points)
    step = max(1, n // candidates)
    starts = range(0, n, step)
 
    best_route = None
    best_dist  = math.inf
 
    for s in starts:
        r = nearest_neighbor(points, start_idx=s)
        d = total_distance(r)
        if d < best_dist:
            best_dist  = d
            best_route = r
 
    return best_route
 
 
def optimize(points: list[dict], use_two_opt: bool = True) -> list[dict]:
    if not points:
        return []
 
    print(f"  Nearest Neighbor ({len(points)} точек)...", end=" ", flush=True)
    t0 = time.time()
    route = find_best_start(points, candidates=min(8, len(points)))
    print(f"{time.time() - t0:.1f}с  →  {total_distance(route):.1f} км")
 
    if use_two_opt and len(points) <= 300:
        print(f"  2-opt...", end=" ", flush=True)
        t0 = time.time()
        route = two_opt(route)
        print(f"{time.time() - t0:.1f}с  →  {total_distance(route):.1f} км")
    elif len(points) > 300:
        print("  2-opt пропущен (>300 точек, используйте --district для фильтрации)")
 
    return route
 
 

 
def export_csv(route: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["№", "Название", "Тип", "Регион", "Район", "Широта", "Долгота"])
        for i, p in enumerate(route, 1):
            writer.writerow([i, p["name"], p["type"], p["region"], p["district"], p["lat"], p["lon"]])
    print(f"  Сохранено: {path}")
 
 
def export_html_map(route: list[dict], path: str) -> None:
    try:
        import folium
    except ImportError:
        print("folium не установлен. Запустите: pip install folium")
        return
 
    if not route:
        return
 
    center_lat = sum(p["lat"] for p in route) / len(route)
    center_lon = sum(p["lon"] for p in route) / len(route)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")
 
    # Линия маршрута
    coords = [(p["lat"], p["lon"]) for p in route]
    folium.PolyLine(coords, color="#E24B4A", weight=2, opacity=0.8, dash_array="8 4").add_to(m)
 

    color_map = {"village": "blue", "hamlet": "green", "isolated_dwelling": "orange"}
    for i, p in enumerate(route, 1):
        color = color_map.get(p["type"], "gray")
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=7,
            color="#ffffff",
            weight=1.5,
            fill=True,
            fill_color="#E24B4A",
            fill_opacity=0.9,
            tooltip=f"{i}. {p['name']}",
            popup=folium.Popup(
                f"<b>{i}. {p['name']}</b><br>{p['type']}<br>{p['district']}<br>"
                f"<small>{p['lat']:.5f}, {p['lon']:.5f}</small>",
                max_width=200
            ),
        ).add_to(m)

  
    for label, idx in [("Старт", 0), ("Финиш", -1)]:
        p = route[idx]
        folium.Marker(
            location=[p["lat"], p["lon"]],
            tooltip=f"{label}: {p['name']}",
            icon=folium.Icon(color="red" if idx == 0 else "darkblue", icon="flag"),
        ).add_to(m)
 
    m.save(path)
    print(f"  Сохранено: {path}")
 
 
def print_summary(route: list[dict]) -> None:
    dist_km = total_distance(route)
    regions  = sorted({p["region"] for p in route if p["region"]})
    types    = {}
    for p in route:
        types[p["type"]] = types.get(p["type"], 0) + 1
 
    print("\n" + "═" * 50)
    print(f"  МАРШРУТ: {len(route)} населённых пунктов")
    print(f"  Расстояние (по прямой): {dist_km:.1f} км")
    print(f"  Регионы: {', '.join(regions) if regions else '—'}")
    print(f"  Типы: " + ", ".join(f"{v} {k}" for k, v in sorted(types.items())))
    print("═" * 50)
    print(f"\n  {'№':>3}  {'Название':<28}  {'Тип':<18}  {'Район'}")
    print("  " + "─" * 80)
    for i, p in enumerate(route, 1):
        print(f"  {i:>3}  {p['name']:<28}  {p['type']:<18}  {p['district']}")
 
 

 
def run_all_regions(points: list[dict], out_dir: Path, place_type: str | None) -> None:
    regions = sorted({p["region"] for p in points if p["region"]})
    print(f"\nНайдено регионов: {len(regions)}")
 
    for region in regions:
        pts = filter_points(points, region=region, place_type=place_type)
        if len(pts) < 2:
            print(f"\n[{region}] слишком мало точек ({len(pts)}), пропуск")
            continue
 
        print(f"\n{'─' * 50}")
        print(f"  {region}  ({len(pts)} точек)")
        route = optimize(pts)
 
        slug = region.replace(" ", "_").replace("(", "").replace(")", "")
        export_csv(route,     str(out_dir / f"route_{slug}.csv"))
        export_html_map(route, str(out_dir / f"route_{slug}.html"))
 
    print(f"\n✓ Готово. Файлы сохранены в: {out_dir}/")
 
 

 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оптимизатор маршрута объезда деревень")
    p.add_argument("--input",    default="osm_villages_raw.csv",
                   help="Путь к CSV-файлу (default: osm_villages_raw.csv)")
    p.add_argument("--region",   default=None,
                   help="Фильтр по региону, напр. 'Курская область'")
    p.add_argument("--district", default=None,
                   help="Фильтр по району, напр. 'Льговский район'")
    p.add_argument("--type",     default=None, dest="place_type",
                   choices=["village", "hamlet", "isolated_dwelling"],
                   help="Фильтр по типу населённого пункта")
    p.add_argument("--start",    default=None, nargs=2, type=float,
                   metavar=("LAT", "LON"),
                   help="Начальная точка маршрута (широта долгота)")
    p.add_argument("--no-2opt",  action="store_true",
                   help="Пропустить 2-opt (быстрее, хуже качество)")
    p.add_argument("--out",      default=".",
                   help="Папка для выходных файлов (default: текущая)")
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 

    csv_path = args.input

 
    print(f"Загрузка {csv_path}...")
    all_points = load_csv(csv_path)
    print(f"Загружено: {len(all_points)} точек")
 

    if args.region is None and args.district is None:
        run_all_regions(all_points, out_dir, args.place_type)
        return
 

    pts = filter_points(all_points, args.region, args.district, args.place_type)
    if len(pts) < 2:
        print(f"Слишком мало точек после фильтрации: {len(pts)}")
        sys.exit(1)
 
    print(f"Точек после фильтрации: {len(pts)}")
 

    if args.start:
        start_pt = {"name": "Старт", "lat": args.start[0], "lon": args.start[1],
                    "type": "", "region": "", "district": ""}

        nearest_idx = min(range(len(pts)), key=lambda i: haversine(start_pt, pts[i]))
        pts.insert(0, pts.pop(nearest_idx))
 
    print("\nОптимизация маршрута...")
    route = optimize(pts, use_two_opt=not args.no_2opt)
 
    print_summary(route)
 

    parts = []
    if args.region:   parts.append(args.region.replace(" ", "_"))
    if args.district: parts.append(args.district.replace(" ", "_"))
    slug = "_".join(parts) if parts else "route"
 
    export_csv(route,      str(out_dir / f"{slug}.csv"))
    export_html_map(route,  str(out_dir / f"{slug}.html"))
    print(f"\n✓ Готово.")
 
 
import sys
sys.argv = ['village_routes.py']
main()
