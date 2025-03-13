# pulls the eventlist data from the eventlist api.
# filters down to just pga tour with sg and traditional stats
# checks whether each round already exists in the eventlist table
# if not, adds the list then calls the player_round api to pull the data for that tournament
# adds the tournament to the player_round_stats table
# saves a csv with the new tournament dat to the s3 bucket


import os
import io
import requests
import pymysql
import pandas as pd
import boto3

s3_client = boto3.client('s3')
s3_bucket = os.environ.get('S3_BUCKET')


def lambda_handler(event, context):
    api_key = os.environ.get('API_KEY')
    db_host = os.environ.get('RDS_HOST')
    db_user = os.environ.get('RDS_USER')
    db_password = os.environ.get('RDS_PASSWORD')
    db_name = os.environ.get('RDS_DB_NAME')

    if not api_key:
        return {"statusCode": 500, "body": "api key error"}

    eventlist_api = f"https://feeds.datagolf.com/historical-raw-data/event-list?file_format=csv&key={api_key}"
    try:
        response = requests.get(eventlist_api)
        # https://www.geeksforgeeks.org/response-raise_for_status-python-requests/#
        response.raise_for_status()
        # https://andrewpwheeler.com/2022/11/02/using-io-objects-in-python-to-read-data/
        event_df = pd.read_csv(io.StringIO(response.text))

    except Exception as e:
        print("eventlist error:", e)
        return {"statusCode": 500, "body": f"error: {e}"}

    # only pga tournaments with sg and traditional stats
    event_df = event_df[
        (event_df["tour"].str.lower() == "pga") &
        (event_df["sg_categories"].str.lower() == "yes") &
        (event_df["traditional_stats"].str.lower() == "yes")
        ]

    ###########################################################################################
    ###########################################################################################
    # necking down to just pulling in tournaments after feb 24, 2025 for testing
    # loading all old data into the micro rds instance was taking too long
    ###########################################################################################
    ###########################################################################################

    event_df["date"] = pd.to_datetime(event_df["date"])
    event_df = event_df[event_df["date"] > pd.Timestamp("2025-02-24")]

    # pga tournaments with sg stats
    print(len(event_df))

    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            db=db_name,
            connect_timeout=5,
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        print("db connection error:", e)
        return {"statusCode": 500, "body": f"error: {e}"}

    num_new_events = 0
    num_new_rounds = 0

    try:
        with connection.cursor() as cursor:
            for _, event_info in event_df.iterrows():
                event_id = event_info['event_id']
                tour = event_info['tour']
                calendar_year = event_info['calendar_year']
                date = event_info['date']

                event_exists_query = """
                    SELECT COUNT(*) AS count
                    FROM pga_with_sg_cat_eventslist
                    WHERE event_id = %s AND tour = %s AND calendar_year = %s
                """
                cursor.execute(event_exists_query, (event_id, tour, calendar_year))
                result = cursor.fetchone()

                # https://dev.mysql.com/doc/connector-python/en/connector-python-example-cursor-transaction.html
                # https://stackoverflow.com/questions/20463333/mysqldb-python-insert-d-and-s
                # https://stackoverflow.com/questions/74064500/im-trying-to-insert-values-into-mysql-table-in-python-but-i-keep-getting-a-err

                if result['count'] == 0:
                    insert_event_query = """
                        INSERT INTO pga_with_sg_cat_eventslist (
                            calendar_year, date, event_id, event_name, 
                            sg_categories, tour, traditional_stats
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    # https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-execute.html
                    cursor.execute(insert_event_query, (
                        event_info['calendar_year'],
                        date,
                        event_id,
                        event_info['event_name'],
                        event_info['sg_categories'],
                        tour,
                        event_info['traditional_stats']
                    ))
                    num_new_events += 1

                    # player-round stats api call
                    ############################################################################################
                    ###########################################################################################

                    player_rounds_stats_api = (
                        f"https://feeds.datagolf.com/historical-raw-data/rounds"
                        f"?tour={tour}&event_id={event_id}&year={calendar_year}"
                        f"&file_format=csv&key={api_key}"
                    )
                    try:
                        rounds_response = requests.get(player_rounds_stats_api)
                        rounds_response.raise_for_status()
                        rounds_df = pd.read_csv(io.StringIO(rounds_response.text))
                    except Exception as e:
                        print(f"player_round stats error for event={event_id}:", e)
                        continue

                    columns_to_keep_for_model = ['sg_t2g', 'sg_total', 'driving_dist', 'driving_acc',
                                                 'gir', 'scrambling', 'prox_rgh', 'prox_fw', 'great_shots', 'poor_shots'
                                                 ]
                    cleaned_df_for_s3 = rounds_df[columns_to_keep_for_model].dropna()

                    # https://stackoverflow.com/questions/38154040/save-dataframe-to-csv-directly-to-s3-python

                    ############################################################################################
                    ###########################################################################################

                    csv_buffer = io.StringIO()
                    cleaned_df_for_s3.to_csv(csv_buffer, index=False)

                    s3_bucket_path_and_filename = f"data/{event_id}_{date}.csv"

                    try:
                        s3_client.put_object(Bucket=s3_bucket, Key=s3_bucket_path_and_filename,
                                             Body=csv_buffer.getvalue())

                    except Exception as e:
                        print("error with s3 csv upload:", e)

                # removing attempt at loading all the stats into the db. kept running into hiccups. switched to simply saving a csv to the s3 for each new event
                # insert_round_query = """
                #     INSERT INTO player_round_stats_pga_with_sg_data (
                #         tour,
                #         year,
                #         season,
                #         event_name,
                #         event_id,
                #         player_name,
                #         dg_id,
                #         fin_text,
                #         round_num,
                #         course_name,
                #         course_num,
                #         course_par,
                #         start_hole,
                #         teetime,
                #         round_score,
                #         sg_putt,
                #         sg_arg,
                #         sg_app,
                #         sg_ott,
                #         sg_t2g,
                #         sg_total,
                #         driving_dist,
                #         driving_acc,
                #         gir,
                #         scrambling,
                #         prox_rgh,
                #         prox_fw,
                #         great_shots,
                #         poor_shots
                #     )
                #     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                # """
                # for _, round_data in rounds_df.iterrows():
                #     cursor.execute(insert_round_query, tuple(round_data.fillna("").values))
                #     num_new_rounds += 1

                else:
                    pass

            connection.commit()
    except Exception as e:
        print("error updating db:", e)
        return {"statusCode": 500, "body": f"error updating db: {e}"}
    finally:
        connection.close()

    message = (
        f"num new events : {num_new_events}, "
        f"nume new rounds: {num_new_rounds}"
    )
    print(message)
    return {"statusCode": 200, "body": message}
