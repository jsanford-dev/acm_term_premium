import pandas as pd
import yaml

def fetch_gsw_data(path):
    """Fetches and formats GSW data from data folder."""

    df = pd.read_csv(
        path,
        skiprows=9, 
        header=0,
        parse_dates=["Date"],
        dayfirst=True
    )
    nss_params = ["Date", "BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    df = df[nss_params]
    
    return df


def filter_dates(df, date_begin, date_end):
    """Filters dates based on input and end of month values."""

    df = df.set_index("Date")
    df = df.resample("ME").last()
    start_eom = pd.to_datetime(date_begin) + pd.offsets.MonthEnd(0)
    end_eom = pd.to_datetime(date_end) + pd.offsets.MonthEnd(0)
    end_eom_plus1 = end_eom + pd.offsets.MonthEnd(1)
    df = df.loc[start_eom:end_eom_plus1]
    df = df.ffill()

    return df


def generate_nss_params():
    
    # Load user inputs
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    data_path  = config["gsw_params_path"]
    output_path = config["nss_params_path"]
    date_begin = config["date_begin"]
    date_end = config["date_end"]

    df = fetch_gsw_data(data_path)
    df = filter_dates(df, date_begin, date_end)
    
    df.to_csv(output_path)

if __name__ == "__main__":
    generate_nss_params()