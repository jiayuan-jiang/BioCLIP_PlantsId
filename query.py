import urllib.request
import urllib.parse
import json
import sys
import os

def get(url, headers={}):
    h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def get_ipni_id(taxon_name):
    query = 'SELECT ?ipni WHERE { ?item wdt:P225 "' + taxon_name + '" . ?item wdt:P5037 ?ipni . } LIMIT 1'
    url = "https://query.wikidata.org/sparql?query=" + urllib.parse.quote(query) + "&format=json"
    res = get(url, {"Accept": "application/sparql-results+json"})
    bindings = res["results"]["bindings"]
    return bindings[0]["ipni"]["value"] if bindings else None

def get_native_only(ipni_id):
    encoded = urllib.parse.quote(ipni_id, safe="")
    url = "https://powo.science.kew.org/api/2/taxon/" + encoded + "?fields=distribution"
    data = get(url, {"Accept": "application/json"})
    origin   = data.get("taxonRemarks", "")
    lifeform = data.get("lifeform", "")
    climate  = data.get("climate", "")
    dist     = data.get("distribution", {})
    native_codes = {item["tdwgCode"] for item in dist.get("natives", []) if item.get("tdwgCode")}
    return native_codes, origin, lifeform, climate

GEOJSON_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdwg_level3.geojson")

def get_tdwg_geojson():
    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    data = get("https://raw.githubusercontent.com/tdwg/wgsrpd/master/geojson/level3.geojson")
    with open(GEOJSON_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

def build_map(taxon_name):
    taxon_name = taxon_name.strip()
    if not taxon_name:
        return False

    print("\n" + "=" * 50)
    print("查询: " + taxon_name)
    print("=" * 50)

    print("[1/4] Wikidata -> IPNI ID ...")
    ipni_id = get_ipni_id(taxon_name)
    if not ipni_id:
        print("  ❌ 未找到，请检查拉丁名拼写")
        return False
    print("  ✅ " + ipni_id)

    print("[2/4] POWO 原生分布 ...")
    native_codes, origin, lifeform, climate = get_native_only(ipni_id)
    if not native_codes:
        print("  ❌ 无原生分布数据")
        return False
    print("  ✅ " + str(len(native_codes)) + " 个原生区域")
    print("     起源: " + (origin or "N/A"))

    print("[3/4] TDWG GeoJSON ...")
    geojson = get_tdwg_geojson()
    print("  ✅ " + str(len(geojson["features"])) + " 个区域")

    print("[4/4] 生成地图 ...")

    # 只保留 native 区域 + 灰色背景
    native_features   = []
    bg_features       = []
    matched           = 0

    for f in geojson["features"]:
        code = f["properties"].get("LEVEL3_COD", "")
        name = f["properties"].get("LEVEL3_NAM", code)
        if code in native_codes:
            matched += 1
            native_features.append({
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": {"code": code, "name": name}
            })
        else:
            bg_features.append({
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": {}
            })

    print("  匹配 native 区域: " + str(matched) + " / " + str(len(native_codes)))

    bg_geojson     = json.dumps({"type": "FeatureCollection", "features": bg_features})
    native_geojson = json.dumps({"type": "FeatureCollection", "features": native_features})

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "<meta charset='utf-8'/>\n"
        "<title>" + taxon_name + " — Native Origin</title>\n"
        "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>\n"
        "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>\n"
        "<style>\n"
        "* { box-sizing:border-box; margin:0; padding:0; }\n"
        "body { font-family:-apple-system,sans-serif; background:#a8c8e8; }\n"
        "#map { height:100vh; width:100%; }\n"
        "#panel {\n"
        "  position:absolute; top:12px; left:58px; z-index:1000;\n"
        "  background:white; padding:14px 18px; border-radius:10px;\n"
        "  box-shadow:0 2px 12px rgba(0,0,0,.2); max-width:380px;\n"
        "}\n"
        "#panel h3 { font-size:16px; font-style:italic; font-weight:600; margin-bottom:6px; }\n"
        "#panel p  { font-size:12px; color:#555; line-height:1.7; margin-top:4px; }\n"
        ".tag { display:inline-block; font-size:11px; padding:2px 8px; border-radius:99px; background:#f0f0f0; margin:3px 3px 0 0; }\n"
        "#legend {\n"
        "  position:absolute; bottom:28px; right:10px; z-index:1000;\n"
        "  background:white; padding:12px 16px; border-radius:10px;\n"
        "  box-shadow:0 2px 12px rgba(0,0,0,.2); font-size:13px;\n"
        "}\n"
        "#legend b { display:block; margin-bottom:8px; font-size:11px; color:#888; text-transform:uppercase; letter-spacing:.05em; }\n"
        ".leg-row { display:flex; align-items:center; gap:8px; margin:5px 0; }\n"
        ".leg-box { width:14px; height:14px; border-radius:3px; flex-shrink:0; }\n"
        "</style>\n</head>\n<body>\n"
        "<div id='map'></div>\n"
        "<div id='panel'>\n"
        "  <h3>" + taxon_name + "</h3>\n"
        "  <p><b>Native origin:</b> " + (origin or "N/A") + "</p>\n"
        "  <p>\n"
        "    <span class='tag'>🌿 " + (lifeform or "N/A") + "</span>\n"
        "    <span class='tag'>🌡 " + (climate or "N/A") + "</span>\n"
        "    <span class='tag'>📍 " + str(matched) + " native regions</span>\n"
        "  </p>\n"
        "</div>\n"
        "<div id='legend'>\n"
        "  <b>Native Origin</b>\n"
        "  <div class='leg-row'><div class='leg-box' style='background:#2ecc71'></div>Native range</div>\n"
        "  <div class='leg-row'><div class='leg-box' style='background:#dce8d4;border:1px solid #ccc'></div>Not native</div>\n"
        "</div>\n"
        "<script>\n"
        "var map = L.map('map', {zoomControl:true}).setView([20,0],2);\n"
        "L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png',{\n"
        "  attribution:'© OpenStreetMap © CARTO | Distribution © POWO / RBG Kew'\n"
        "}).addTo(map);\n"
        "\n"
        "// 背景灰色区域\n"
        "L.geoJSON(" + bg_geojson + ",{\n"
        "  style:function(){return {fillColor:'#dce8d4',fillOpacity:0.6,color:'#b0c4a8',weight:0.4};}\n"
        "}).addTo(map);\n"
        "\n"
        "// 原生区域高亮\n"
        "L.geoJSON(" + native_geojson + ",{\n"
        "  style:function(){return {fillColor:'#2ecc71',fillOpacity:0.85,color:'#27ae60',weight:1};},\n"
        "  onEachFeature:function(f,layer){\n"
        "    layer.bindPopup('<b>'+f.properties.name+'</b><br><small>'+f.properties.code+'</small><br><span style=\"color:#27ae60;font-weight:500\">Native</span>');\n"
        "    layer.on('mouseover',function(){layer.setStyle({fillColor:'#27ae60',fillOpacity:1,weight:2,color:'#1a8a47'});});\n"
        "    layer.on('mouseout', function(){layer.setStyle({fillColor:'#2ecc71',fillOpacity:0.85,weight:1,color:'#27ae60'});});\n"
        "  }\n"
        "}).addTo(map);\n"
        "</script>\n</body>\n</html>"
    )

    fname = taxon_name.replace(" ", "_") + "_native.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ 完成 -> " + fname)
    return True

def main():
    if len(sys.argv) > 1:
        build_map(" ".join(sys.argv[1:]))
        return

    print("植物原生分布地图生成器")
    print("输入拉丁名查询，直接回车退出\n")
    while True:
        try:
            name = input("植物拉丁名 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not name:
            print("退出")
            break
        build_map(name)

if __name__ == "__main__":
    main()