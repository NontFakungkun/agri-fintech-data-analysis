import json
import pandas as pd

def convert_date(input_file, output_file):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Convert the date column to datetime format (assuming the column name is 'Date')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Write the reversed DataFrame back to the Excel file
    df.to_excel(output_file, index=False)


if __name__ == "__main__":
    convert_date("Weather.xlsx", "new_Weather.xlsx")
