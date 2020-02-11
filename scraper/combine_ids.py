import pandas as pd
import os

def main():
    df_list = []
    csv_list = [os.path.join('ids', x) for x in os.listdir('ids') if '.csv' in x and x != 'ids.csv']
    df_list = [pd.read_csv(x) for x in csv_list]
    new_df = pd.concat(df_list).reset_index(drop=True)
    new_df.to_csv('ids/ids.csv', index=False)

if __name__ == '__main__':
    main()