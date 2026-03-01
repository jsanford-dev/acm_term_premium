import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import yaml

class ACM():
    def __init__(self):
        # Set up.
        self.config = self.load_config()
        self.maturities = self.load_maturities()

        # Define dimensions and parameters.
        self.K = self.config["K"]
        self.N = len(self.config["maturity_array"])
        self.T = None
        self.max_maturity = max(self.maturities)

    def load_config(self):
        """Loads user inputs from config.yaml."""
        
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        return config
    
    def load_maturities(self):
        """ Loads maturities and ensures 1 month rate in array."""

        maturities = np.asarray(self.config["maturity_array"], dtype=int)
        maturities = np.sort(maturities)
        
        if maturities[0] == 1:
            pass
        elif maturities[0] != 1 and maturities[0] >= 2:
            maturities = np.insert(maturities, 0, 1)
        else:
            raise ValueError("Please ensure that the maturity array" \
            " starts with a value than 1.")

        return maturities
    
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

        return yields.astype(float)
    
    def generate_excess_returns(self, nss_params):
        """Generates risk free and excess returns per eq. 6."""

        # Derive last two components of eq. 6.
        log_prices = -self.yields * (self.yields.columns / 12)
        rf = -log_prices[log_prices.columns[0]]
        log_prices = log_prices.drop(columns=log_prices.columns[0])

        # Define dimensions of dimension T.
        self.T = len(log_prices.index)

        # Derive first component of eq. 6.
        maturities_excl_1m = self.maturities[1:] # remove 1 month rate from array.
        adj_maturities = [x - 1 for x in maturities_excl_1m]
        adj_yields = self.generate_yields(nss_params, adj_maturities)
        adj_yields = adj_yields.drop(adj_yields.index[0])
        log_prices_t1 = -adj_yields * (adj_yields.columns / 12)

        assert (log_prices.index == nss_params.index[:-1]).all(), \
            "log_prices index misaligned with nss_params[:-1]"

        assert (log_prices_t1.index == nss_params.index[1:]).all(), \
            "log_prices_t1 index misaligned with nss_params[1:]"

        # convert to numpy arrays for easier calculations
        log_prices_t1 = log_prices_t1.to_numpy(dtype=float)
        log_prices = log_prices.to_numpy(dtype=float)
        rf = rf.to_numpy(dtype=float).reshape(-1, 1)

        # excess returns - per eq. 6.
        rx = log_prices_t1 - log_prices - rf
        rx = rx.T  # Paper wants N x T per page 6.
        assert rx.shape == (self.N, self.T), f"rx has shape {rx.shape}, " +\
            "expected (N, T)."
        
        return rx

    
    def get_principal_components(self):
        """PCA on monthly yields."""
        # Per section 3.1, PCA based on n = 3,..., 120 months.
        pca_maturities = self.config["maturity_array"]
        pca_yields = self.yields[pca_maturities].astype(float)

        # De-mean yields for analysis
        mean_yields = pca_yields.mean()
        demean_yields = pca_yields - mean_yields

        # Run PCA
        pca = PCA(n_components=self.K)
        X_ = pca.fit_transform(demean_yields).T
        beta = pca.components_

        assert X_.shape == (self.K, self.T), \
            f"Expected X_ shape {(self.K, self.T)}, got {X_.shape}"
        
        assert beta.shape == (self.K, self.N), \
            f"Expected beta shape {(self.K, self.N)}, got {beta.shape}"

        return X_, beta
    
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

        rx = self.generate_excess_returns(nss_params)

        X_, beta = self.get_principal_components()

        # Step 1, 2, 3 go here later.

        return
