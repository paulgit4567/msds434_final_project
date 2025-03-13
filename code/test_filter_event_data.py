import pandas as pd
from .filter_for_pga_sg_and_stats import filter_event_data

def test_filter_event_data():

    data = {
        "calendar_year": [2025, 2025],
        "date": ["2025-02-16", "2025-02-09"],
        "event_id": [7, 3],
        "event_name": ["The Genesis Invitational", "Pauls Made Up Tournament"],
        "sg_categories": ["yes", "no"],
        "tour": ["pga", "abc"],
        "traditional_stats": ["yes", "no"]
    }
    df = pd.DataFrame(data)

    filtered_df = filter_event_data(df)
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]["event_id"] == 7
    assert filtered_df.iloc[0]["event_name"] == "The Genesis Invitational"
