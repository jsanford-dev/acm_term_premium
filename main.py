from acm import ACM
from clean_gsw_data import generate_nss_params

def main():
    generate_nss_params()

    acm = ACM()
    acm.acm_model()

if __name__ == '__main__':
    main()