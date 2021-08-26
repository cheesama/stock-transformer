from datetime import datetime, date

import os, sys

def get_all_stock_data():
    if not os.path.isdir('marcap'):
        os.system('git clone "https://github.com/FinanceData/marcap.git" marcap')
    else:
        os.system('cd marcap && git pull')

    from marcap.marcap_utils import marcap_data

    df = marcap_data('2000-01-01', datetime.today().strftime('%Y-%m-%d'))

    return df