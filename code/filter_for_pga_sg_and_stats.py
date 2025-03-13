import pandas as pd

# test change 

def filter_event_data(event_df):

    event_df["tour"] = event_df["tour"].str.lower()
    event_df["sg_categories"] = event_df["sg_categories"].str.lower()
    event_df["traditional_stats"] = event_df["traditional_stats"].str.lower()

    event_df = event_df[
        (event_df["tour"] == "pga") &
        (event_df["sg_categories"] == "yes") &
        (event_df["traditional_stats"] == "yes")
    ]
    return event_df
