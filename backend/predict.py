import numpy as np
import pandas as pd
from math import isclose
from typing import Optional, Dict, Tuple, Any, Literal
from datetime import datetime

from pydantic_models import UserInput, PredictionResponse, meta_dict, reverse_id_map

# === Load precomputed feature CSV and construct lookup ===
precomputed_df = pd.read_csv("backend/files/precomputed_features.csv")

# Add ISO timestamp column for consistency
precomputed_df["TimeStamp"] = pd.to_datetime(
    precomputed_df["Day"].astype(str) + " " + precomputed_df["Time"],
    format="%w %H:%M:%S"
)
precomputed_df["TimeStampStr"] = precomputed_df["TimeStamp"].apply(lambda x: x.isoformat())

# Create df_lookup from precomputed_df
df_lookup: Dict[Tuple[int, str], Dict[str, Any]] = {
    (int(row.SystemCodeNumber), row.TimeStampStr): row.to_dict()
    for _, row in precomputed_df.iterrows()
}

# === Pre-trained regression weights ===
weights: np.ndarray = np.array([
    10.00488263,  # Intercept
    3.5161334,    # Utilization
    2.39221267,   # QueuePressure
    0.94771733,   # QueueLengthNorm
    0.33298709,   # VehicleType / 3
    0.25926018,   # TrafficConditionNearby / 2
    0.491136,     # IsSpecialDay
    0.5006001     # TimeCategory / 2
])

# === Constants ===
BASE_PRICE: float = 10
PEAK_MULTIPLIER: float = 2.0
MIN_PRICE: float = 5
MAX_PRICE: float = 20

# === Normalize scalar using MinMax ===
def normalize_minmax_scalar(val: float, col_values: list) -> float:
    min_val = float(np.min(col_values))
    max_val = float(np.max(col_values))
    denom = max_val - min_val if not isclose(max_val, min_val) else 1
    return (val - min_val) / denom

# === Haversine distance in km ===
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))

# === Core Prediction Function ===
def predict_price(user: UserInput) -> PredictionResponse:
    key = (user.SystemCodeNumber, user.TimeStamp.isoformat())

    queue_vals = [row["AvgQueueLength"] for row in df_lookup.values()]
    queue_norm: float = normalize_minmax_scalar(user.QueueLength, queue_vals)

    # Step 1: Rule-based override
    if user.QueueLength >= 10 and user.QueuePressure >= 1.0:
        price = BASE_PRICE * PEAK_MULTIPLIER
    else:
        features = np.array([
            1,
            user.Utilization,
            user.QueuePressure,
            queue_norm,
            user.VehicleType / 3,
            user.TrafficConditionNearby / 2,
            user.IsSpecialDay,
            user.TimeCategory / 2,
        ])
        price = float(np.dot(weights, features))

    price = round(np.clip(price, MIN_PRICE, MAX_PRICE), 2)

    # Step 2: Rerouting suggestion
    reroute_int: Optional[int] = suggest_reroute(
        scn=user.SystemCodeNumber,
        ts=user.TimeStamp,
        qpress=user.QueuePressure,
        curr_price=price
    )
    reroute_str: Optional[str] = reverse_id_map.get(reroute_int) if reroute_int is not None else None

    # Step 3: Traffic interpretation
    # Define the type explicitly for validation
    TrafficLevel = Literal["Low", "Moderate", "High"]

    def get_traffic_level(code: int) -> TrafficLevel:
        if code == 0:
            return "Low"
        elif code == 1:
            return "Moderate"
        elif code == 2:
            return "High"
        else:
            return "Moderate"

    return PredictionResponse(
        price=price,
        reroute_to=reroute_str,
        ExpectedTrafficNearby=get_traffic_level(user.TrafficConditionNearby)
    )

# === Suggest reroute function ===
def suggest_reroute(
    scn: int,
    ts: datetime,
    qpress: float,
    curr_price: float,
    queue_pressure_thresh_high: float = 1.1,
    queue_pressure_thresh_moderate: float = 0.95,
    price_diff_thresh: float = 3.0,
    radius_km: float = 2.0
) -> Optional[int]:
    lat1 = meta_dict[scn]["Latitude"]
    lon1 = meta_dict[scn]["Longitude"]
    ts_str = ts.isoformat()

    candidates = []

    for other_scn, meta in meta_dict.items():
        if other_scn == scn:
            continue

        lat2, lon2 = meta["Latitude"], meta["Longitude"]
        dist = haversine(lat1, lon1, lat2, lon2)

        if dist > radius_km:
            continue

        other_row = df_lookup.get((other_scn, ts_str))
        if not other_row:
            continue

        other_price = other_row["price"]
        other_qpress = other_row["QueuePressure"]

        # Rule 1: High congestion & better qpress
        if qpress > queue_pressure_thresh_high and other_qpress < qpress:
            candidates.append((other_scn, dist, other_price, other_qpress))

        # Rule 2: Moderate congestion + price advantage
        elif (
            queue_pressure_thresh_moderate <= qpress <= queue_pressure_thresh_high and
            (curr_price - other_price) >= price_diff_thresh and
            other_qpress < qpress
        ):
            candidates.append((other_scn, dist, other_price, other_qpress))

    if candidates:
        best = sorted(candidates, key=lambda x: (x[3], x[2], x[1]))[0]
        return int(best[0])

    return None
