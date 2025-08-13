import numpy as np
import pandas as pd
import yaml

class ACM():
    def __init__(self):
        self.config = self.load_config()
        self.maturities = self.config["maturity_array"]
        self.yields = self.generate_yields()

    def load_config(self):
        """Loads user inputs from config.yaml."""
        
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        return config
    

    def generate_yields(self):
        """Generates zero coupon bond yields based on NSS params."""

        nss_params = (
            pd.read_csv(self.config["nss_params_path"])
            .set_index("Date", drop=True)
        )

        def apply_nss(maturity, beta_0, beta_1, beta_2, beta_3, tau_1, tau_2):
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

        yields = pd.DataFrame(index=nss_params.index, columns=self.maturities)

        for date, row in nss_params.iterrows():
           yields.loc[date, :] = apply_nss(
               self.maturities,
               row["BETA0"],
               row["BETA1"],
               row["BETA2"],
               row["BETA3"],
               row["TAU1"],
               row["TAU2"]
           )

        print(yields)

        return yields
    

    def generate_term_premium(self):
        return
