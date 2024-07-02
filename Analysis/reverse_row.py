import json
import pandas as pd

def reverse_excel_rows(input_file, output_file):
    df = pd.read_excel(input_file)
    df_reversed = df.iloc[::-1]
    df_reversed.to_excel(output_file, index=False)


if __name__ == "__main__":
    reverse_excel_rows("New_Corn_PriceHistory.xlsx", "R_New_Corn_PriceHistory.xlsx")
    reverse_excel_rows("New_Wheat_PriceHistory.xlsx", "R_New_Wheat_PriceHistory.xlsx")
