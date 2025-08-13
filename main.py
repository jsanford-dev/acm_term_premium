from acm import ACM
from clean_gsw_data import generate_nss_params

def main():
    generate_nss_params()

    acm = ACM()
    acm.generate_term_premium()

if __name__ == '__main__':
    main()