import os
import datetime
import pandas as pd

class General:
    @staticmethod
    def time_cell():
        now = datetime.datetime.now()
        return [now.strftime("%Y"), now.strftime("%b"), now.strftime("%d"), now.strftime("%H.%M")]

    @staticmethod
    def file_make(file_name, titles):
        if not os.path.exists(file_name):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            pd.DataFrame(columns=titles).to_csv(file_name, index=False)

    @staticmethod
    def log_info(file_name, data):
        df = pd.read_csv(file_name)
        df.loc[len(df)] = data
        df.to_csv(file_name, index=False)
