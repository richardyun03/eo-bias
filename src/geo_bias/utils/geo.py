"""Geometry helpers: CRS conversions for chip-center handling."""


def utm_epsg(lon: float, lat: float) -> str:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"
