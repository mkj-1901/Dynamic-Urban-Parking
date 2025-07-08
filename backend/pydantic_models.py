from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Dict, Optional, Literal
from datetime import datetime, time
import pandas as pd

# === Load static data ===
df = pd.read_csv('backend/files/precomputed_features.csv')

meta_dict: Dict[int, Dict[str, float]] = {
    0: {'SystemCodeNumber': 'BHMBCCMKT01', 'Capacity': 577, 'Latitude': 26.14453614, 'Longitude': 91.73617216},
    1: {'SystemCodeNumber': 'BHMBCCTHL01', 'Capacity': 387, 'Latitude': 26.14449459, 'Longitude': 91.73620513},
    2: {'SystemCodeNumber': 'BHMEURBRD01', 'Capacity': 470, 'Latitude': 26.14901995, 'Longitude': 91.7395035},
    3: {'SystemCodeNumber': 'BHMMBMMBX01', 'Capacity': 687, 'Latitude': 20.0000347, 'Longitude': 78.00000286},
    4: {'SystemCodeNumber': 'BHMNCPHST01', 'Capacity': 1200, 'Latitude': 26.14001386, 'Longitude': 91.73099967},
    5: {'SystemCodeNumber': 'BHMNCPNST01', 'Capacity': 485, 'Latitude': 26.14004753, 'Longitude': 91.73097233},
    6: {'SystemCodeNumber': 'Broad Street', 'Capacity': 690, 'Latitude': 26.13795775, 'Longitude': 91.74099445},
    7: {'SystemCodeNumber': 'Others-CCCPS105a', 'Capacity': 2009, 'Latitude': 26.14747299, 'Longitude': 91.72804914},
    8: {'SystemCodeNumber': 'Others-CCCPS119a', 'Capacity': 2803, 'Latitude': 26.14754061, 'Longitude': 91.72797041},
    9: {'SystemCodeNumber': 'Others-CCCPS135a', 'Capacity': 3883, 'Latitude': 26.14749943, 'Longitude': 91.72800489},
    10: {'SystemCodeNumber': 'Others-CCCPS202', 'Capacity': 2937, 'Latitude': 26.14749053, 'Longitude': 91.72799688},
    11: {'SystemCodeNumber': 'Others-CCCPS8', 'Capacity': 1322, 'Latitude': 26.14754886, 'Longitude': 91.72799519},
    12: {'SystemCodeNumber': 'Others-CCCPS98', 'Capacity': 3103, 'Latitude': 26.14749998, 'Longitude': 91.72797778},
    13: {'SystemCodeNumber': 'Shopping', 'Capacity': 1920, 'Latitude': 26.15050395, 'Longitude': 91.73353109}
}

reverse_id_map: Dict[int, str] = {i: v['SystemCodeNumber'] for i, v in meta_dict.items()}

vehicle_map : Dict[int,str] = {0: 'cycle', 1: 'bike', 2: 'car', 3: 'truck'}

def round_time_to_30_min(dt: datetime) -> time:
    """Round to nearest 30-min slot between 08:00 and 16:30."""
    rounded_min = 30 if dt.minute >= 15 else 0
    t = time(dt.hour, rounded_min)
    return max(min(t, time(16, 30)), time(8, 0))


class UserInput(BaseModel):
    SystemCodeNumber: Annotated[int, Field(..., description='Code of parking spot', examples=[0, 1, 2])]
    TimeStamp: Annotated[datetime, Field(..., description='Datetime input from user', examples=[datetime(2020, 1, 1, 12, 0, 0)])]
    VehicleType: Annotated[int, Field(..., description='Type of vehicle', examples=[0, 1, 2])]

    @computed_field(return_type=int)
    @property
    def Day(self) -> int:
        return self.TimeStamp.weekday()

    @computed_field(return_type=str)
    @property
    def Time(self) -> str:
        return round_time_to_30_min(self.TimeStamp).strftime('%H:%M:%S')

    def _filter_df(self, col: str) -> pd.Series:
        """Try to match all fields. If not, ignore VehicleType."""
        full_match = (
            (df['SystemCodeNumber'] == self.SystemCodeNumber) &
            (df['Day'] == self.Day) &
            (df['Time'] == self.Time) &
            (df['VehicleType'] == self.VehicleType)
        )
        series = df.loc[full_match, col]
        if not series.empty:
            return series

        partial_match = (
            (df['SystemCodeNumber'] == self.SystemCodeNumber) &
            (df['Day'] == self.Day) &
            (df['Time'] == self.Time)
        )
        return df.loc[partial_match, col]

    def _fallback_mean(self, series: pd.Series, fallback_col: str, round_int: bool = True) -> float:
        if not series.empty:
            mean_val = series.mean()
        else:
            mean_val = df[fallback_col].mean()
        return round(mean_val) if round_int else float(mean_val)

    @computed_field(return_type=int)
    @property
    def TrafficConditionNearby(self) -> int:
        return int(self._fallback_mean(self._filter_df('AvgTrafficConditionNearby'), 'AvgTrafficConditionNearby'))

    @computed_field(return_type=int)
    @property
    def QueueLength(self) -> int:
        return int(self._fallback_mean(self._filter_df('AvgQueueLength'), 'AvgQueueLength'))

    @computed_field(return_type=int)
    @property
    def Occupancy(self) -> int:
        return int(self._fallback_mean(self._filter_df('AvgOccupancy'), 'AvgOccupancy'))

    @computed_field(return_type=float)
    @property
    def Utilization(self) -> float:
        cap = meta_dict[self.SystemCodeNumber]['Capacity']
        return round(self.Occupancy / cap, 4)

    @computed_field(return_type=float)
    @property
    def QueuePressure(self) -> float:
        return self._fallback_mean(self._filter_df('QueuePressure'), 'QueuePressure', round_int=False)

    @computed_field(return_type=int)
    @property
    def IsSpecialDay(self) -> int:
        return 1 if self.Day in [5, 6] else 0  # Saturday or Sunday

    @computed_field(return_type=int)
    @property
    def TimeCategory(self) -> int:
        return 0 if self.TimeStamp.hour < 12 else 1

# === Prediction Response Model ===
class PredictionResponse(BaseModel):
    price: Annotated[float, Field(..., description='Price predicted by model', examples=[10, 20, 30, 40, 50])]
    reroute_to: Annotated[Optional[str], Field(..., description='Route predicted by model', examples=['BHMBCCMKT01'])]
    ExpectedTrafficNearby: Annotated[Literal["Low", "Moderate", "High"], Field(..., description='Expected Traffic Condition Nearby')]
