import argparse
import math
import csv
import sys
import time
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Point:
    name: str
    lat: float
    lon: float
    place_type: str
    region: str
    district: str

def load_csv(path: str) -> list[Point]:
    points: list[Point] = []

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                lat = float(row["@lat"])
                lon = float(row["@lon"])
                name = row.get("name", "").strip()

                if not name:
                    continue

                point = Point(
                    name=name,
                    lat=lat,
                    lon=lon,
                    place_type=row.get("place", "").strip(),
                    region=row.get("addr:region", "").strip(),
                    district=row.get("addr:district", "").strip(),
                )

                points.append(point)

            except (ValueError, KeyError):
                continue

    return points
 
 
def filter_points(
    points: list[Point],
    region: str | None = None,
    district: str | None = None,
    place_type: str | None = None,
) -> list[Point]:
    result = points

    if region:
        result = [p for p in result if p.region == region]

    if district:
        result = [p for p in result if p.district == district]

    if place_type:
        result = [p for p in result if p.place_type == place_type]

    return result
 

 
def haversine(a: Point, b: Point) -> float:
    R = 6371.0

    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)

    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(a.lat))
        * math.cos(math.radians(b.lat))
        * math.sin(dlon / 2) ** 2
    )

    return R * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
 
 
def total_distance(route: list[Point]) -> float:
    if len(route) < 2:
        return 0.0

    distance = sum(
        haversine(route[i], route[i + 1])
        for i in range(len(route) - 1)
    )

    distance += haversine(route[-1], route[0])

    return distance
 
 

 
def nearest_neighbor(points: list[Point], start_idx: int = 0) -> list[Point]:
    n = len(points)

    if n == 0:
        return []

    visited = [False] * n
    route = [points[start_idx]]
    visited[start_idx] = True

    for _ in range(n - 1):
        current = route[-1]
        best_distance = math.inf
        best_index = -1

        for j in range(n):
            if not visited[j]:
                distance = haversine(current, points[j])

                if distance < best_distance:
                    best_distance = distance
                    best_index = j

        if best_index == -1:
            break

        visited[best_index] = True
        route.append(points[best_index])

    return route
 
 
def two_opt(route: list[Point], max_iter: int = 5000) -> list[Point]:
    if len(route) < 4:
        return route[:]

    best = route[:]
    improved = True
    iterations = 0
    epsilon = 1e-6

    while improved and iterations < max_iter:
        improved = False
        n = len(best)

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = (
                    haversine(best[i - 1], best[i])
                    + haversine(best[j], best[(j + 1) % n])
                )

                after = (
                    haversine(best[i - 1], best[j])
                    + haversine(best[i], best[(j + 1) % n])
                )

                if after < before - epsilon:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True

            iterations += 1

    return best
 
 
def find_best_start(points: list[Point], candidates: int = 5) -> list[Point]:
    if not points:
        return []

    n = len(points)
    step = max(1, n // candidates)
    start_indices = range(0, n, step)

    best_route: list[Point] = []
    best_distance = math.inf

    for start_idx in start_indices:
        route = nearest_neighbor(points, start_idx=start_idx)
        distance = total_distance(route)

        if distance < best_distance:
            best_distance = distance
            best_route = route

    return best_route
 
 
def optimize(points: list[Point], use_two_opt: bool = True) -> list[Point]:
    if not points:
        return []

    route = find_best_start(points, candidates=min(8, len(points)))

    if use_two_opt and len(points) <= 300:
        route = two_opt(route)

    return route
 
 
def export_csv(route: list[Point], path: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "№",
            "Название",
            "Тип",
            "Регион",
            "Район",
            "Широта",
            "Долгота"
        ])

        for i, p in enumerate(route, 1):
            writer.writerow([
                i,
                p.name,
                p.place_type,
                p.region,
                p.district,
                p.lat,
                p.lon,
            ])

    print(f"Сохранено: {path}")
 
 
def export_html_map(route: list[Point], path: str) -> None:
    try:
        import folium
    except ImportError:
        print("folium не установлен. Запустите: pip install folium")
        return

    if not route:
        return

    center_lat = sum(p.lat for p in route) / len(route)
    center_lon = sum(p.lon for p in route) / len(route)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap"
    )

    coords = [(p.lat, p.lon) for p in route]
    coords.append((route[0].lat, route[0].lon))

    folium.PolyLine(
        coords,
        color="#E24B4A",
        weight=2,
        opacity=0.8,
        dash_array="8 4"
    ).add_to(m)

    color_map = {
        "village": "blue",
        "hamlet": "green",
        "isolated_dwelling": "orange"
    }

    for i, p in enumerate(route, 1):
        color = color_map.get(p.place_type, "gray")

        folium.CircleMarker(
            location=[p.lat, p.lon],
            radius=7,
            color="#ffffff",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"{i}. {p.name}",
            popup=folium.Popup(
                f"<b>{i}. {p.name}</b><br>"
                f"{p.place_type}<br>"
                f"{p.district}<br>"
                f"<small>{p.lat:.5f}, {p.lon:.5f}</small>",
                max_width=200
            ),
        ).add_to(m)

    for label, idx in [("Старт", 0), ("Финиш", -1)]:
        p = route[idx]

        folium.Marker(
            location=[p.lat, p.lon],
            tooltip=f"{label}: {p.name}",
            icon=folium.Icon(
                color="red" if idx == 0 else "darkblue",
                icon="flag",
            ),
        ).add_to(m)

    m.save(path)
    print(f"Сохранено: {path}")
 
 
def print_summary(route: list[Point]) -> None:
    if not route:
        print("Маршрут пуст.")
        return

    dist_km = total_distance(route)

    regions = sorted({p.region for p in route if p.region})

    types: dict[str, int] = {}
    for p in route:
        types[p.place_type] = types.get(p.place_type, 0) + 1

    print("\n" + "═" * 50)
    print(f"МАРШРУТ: {len(route)} населённых пунктов")
    print(f"Длина замкнутого маршрута: {dist_km:.1f} км")
    print(f"Регионы: {', '.join(regions) if regions else '—'}")
    print("Типы: " + ", ".join(f"{v} {k}" for k, v in sorted(types.items())))
    print("═" * 50)

    print(f"\n{'№':>3}  {'Название':<28}  {'Тип':<18}  {'Район'}")
    print(" " + "─" * 80)

    for i, p in enumerate(route, 1):
        print(f"{i:>3}  {p.name:<28}  {p.place_type:<18}  {p.district}")
 

 
def run_all_regions(
    points: list[Point],
    out_dir: Path,
    place_type: str | None
) -> None:
    regions = sorted({p.region for p in points if p.region})

    print(f"\nНайдено регионов: {len(regions)}")

    for region in regions:
        pts = filter_points(points, region=region, place_type=place_type)

        if len(pts) < 2:
            print(f"\n[{region}] слишком мало точек ({len(pts)}), пропуск")
            continue

        print(f"\n{'─' * 50}")
        print(f"{region} ({len(pts)} точек)")

        route = optimize(pts)

        slug = region.replace(" ", "_").replace("(", "").replace(")", "")

        export_csv(route, str(out_dir / f"route_{slug}.csv"))
        export_html_map(route, str(out_dir / f"route_{slug}.html"))

    print(f"\nГотово. Файлы сохранены в: {out_dir}/")
 
 

 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оптимизатор маршрута объезда деревень"
    )

    parser.add_argument(
        "--input",
        default="osm_villages_raw.csv",
        help="Путь к CSV-файлу (default: osm_villages_raw.csv)"
    )

    parser.add_argument(
        "--region",
        help="Фильтр по региону, напр. 'Курская область'"
    )

    parser.add_argument(
        "--district",
        help="Фильтр по району, напр. 'Льговский район'"
    )

    parser.add_argument(
        "--type",
        dest="place_type",
        choices=["village", "hamlet", "isolated_dwelling"],
        help="Фильтр по типу населённого пункта"
    )

    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        help="Начальная точка маршрута (широта, долгота)"
    )

    parser.add_argument(
        "--no-2opt",
        action="store_true",
        help="Пропустить 2-opt (быстрее, хуже качество)"
    )

    parser.add_argument(
        "--out",
        default=".",
        help="Папка для выходных файлов (default: текущая)"
    )

    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Загружать данные динамически через Playwright"
    )

    return parser.parse_args()
 
 
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

    points = filter_points(
        all_points,
        args.region,
        args.district,
        args.place_type,
    )

    if len(points) < 2:
        print(f"Слишком мало точек после фильтрации: {len(points)}")
        sys.exit(1)

    print(f"Точек после фильтрации: {len(points)}")

    if args.start:
        start_point = Point(
            name="Старт",
            lat=args.start[0],
            lon=args.start[1],
            place_type="start",
            region="",
            district="",
        )

        nearest_idx = min(
            range(len(points)),
            key=lambda i: haversine(start_point, points[i]),
        )

        points.insert(0, points.pop(nearest_idx))

    print("\nОптимизация маршрута...")
    route = optimize(points, use_two_opt=not args.no_2opt)

    print_summary(route)

    parts = []

    if args.region:
        parts.append(args.region.replace(" ", "_"))

    if args.district:
        parts.append(args.district.replace(" ", "_"))

    slug = "_".join(parts) if parts else "route"

    export_csv(route, str(out_dir / f"{slug}.csv"))
    export_html_map(route, str(out_dir / f"{slug}.html"))

    print("\nГотово.")


if __name__ == "__main__":
    main()
