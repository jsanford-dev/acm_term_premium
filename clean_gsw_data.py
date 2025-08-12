import pandas as pd

def fetch_gsw_data(path):
    """Fetches and formats GSW data from data folder."""

    df = pd.read_csv(path, skiprows=9, header=0)
    nss_params = ["Date", "BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    df = df[nss_params]

    # convert types
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    
    return df


def filter_dates(df, date_begin, date_end):
    """Filters dates based on input and end of month values."""

    df = df.set_index("Date")
    df = df.resample("ME").last()
    start_eom = pd.to_datetime(date_begin) + pd.offsets.MonthEnd(0)
    end_eom = pd.to_datetime(date_end) + pd.offsets.MonthEnd(0)
    df = df.loc[start_eom:end_eom]
    df = df.ffill()

    return df


def clean_gsw_data():
    # User inputs
    data_path  = "data/raw/gsw_params.csv"
    output_path = 'data/clean/nss_params.csv'
    date_begin = '1987-01-31'
    date_end = '2011-12-31'

    df = fetch_gsw_data(data_path)
    df = filter_dates(df, date_begin, date_end)
    
    df.to_csv(output_path)

if __name__ == "__main__":
    clean_gsw_data()