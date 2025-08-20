import numpy as np
import pandas as pd
import yaml

class ACM():
    def __init__(self):
        self.config = self.load_config()
        self.maturities = np.asarray(self.config["maturity_array"], dtype=int)

        self.K = self.config["K"]

    def load_config(self):
        """Loads user inputs from config.yaml."""
        
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        return config
    
    def nss(self, maturity, beta_0, beta_1, beta_2, beta_3, tau_1, tau_2):
        """Nelson-Siegel-Svensson Model."""

        # convert month maturities to years for NSS.
        t = np.asarray(maturity, dtype=float) / 12.0

        yld = (
            beta_0
            + beta_1 * ((1 - np.exp(-t / tau_1)) / (t/ tau_1))
            + beta_2 * ((1 - np.exp(-t / tau_1)) / (t / tau_1)
                            - np.exp(-t / tau_1))
            + beta_3 * ((1 - np.exp(-t / tau_2)) / (t / tau_2)
                            - np.exp(-t / tau_2))
        ) / 100

        return yld
    

    def generate_yields(self, nss_params, maturities):
        """Generates zero coupon bond yields based on NSS params."""

        yields = pd.DataFrame(index=nss_params.index, columns=maturities)

        for date, row in nss_params.iterrows():
           yields.loc[date, :] = self.nss(
               maturities,
               row["BETA0"],
               row["BETA1"],
               row["BETA2"],
               row["BETA3"],
               row["TAU1"],
               row["TAU2"]
           )

        return yields
    
    def generate_excess_returns(self, nss_params):
        """Generates risk free and excess returns per eq. 6."""

        # Derive last two components of eq. 6.
        log_prices = -self.yields * (self.yields.columns / 12)
        rf = -log_prices[log_prices.columns[0]]
        log_prices = log_prices.drop(columns=1)

        # Derive first component of eq. 6.
        self.maturities = self.maturities[1:] # remove 1 month rate from array.
        adj_maturities = [x - 1 for x in self.maturities]
        adj_yields = self.generate_yields(nss_params, adj_maturities)
        adj_yields = adj_yields.drop(adj_yields.index[0])
        log_prices_t1 = -adj_yields * (adj_yields.columns / 12)

        # convert to numpy arrays for easier calculations
        log_prices_t1 = log_prices_t1.to_numpy(dtype=float)
        log_prices = log_prices.to_numpy(dtype=float)
        rf = rf.to_numpy(dtype=float).reshape(-1, 1)

        # excess returns - per eq. 6.
        rx = log_prices_t1 - log_prices - rf
        print(self.K)

        return rx

    

    def acm_model(self):
        """ Runs ACM model. """

        # Preparing data
        nss_params = (
            pd.read_csv(
                self.config["nss_params_path"],
                parse_dates=["Date"],
                date_format="%Y-%m-%d"
            )
            .sort_values("Date")
            .set_index("Date", drop=True)
        )

        self.yields = self.generate_yields(nss_params, self.maturities)
        self.yields = self.yields.drop(self.yields.index[-1])

        self.generate_excess_returns(nss_params)

        # Step 1, 2, 3 go here later.


        return
