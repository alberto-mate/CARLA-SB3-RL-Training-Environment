import pandas as pd
import matplotlib.pyplot as plt
import os

def create_histogram(df, output_path):
    df.set_index('episode', inplace=True)
    df = df[df['center_dev_mean'] < df['center_dev_mean'].quantile(0.8)][['center_dev_mean', 'center_dev_std']]
    center_dev_mean = df['center_dev_mean']
    center_dev_std = df['center_dev_std']

    fig, ax = plt.subplots()

    ax.bar(center_dev_mean.index, center_dev_mean.values, yerr=center_dev_std.values, capsize=3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Center Deviation Mean')
    ax.set_title('Histogram of Center Deviation Mean')

    plt.savefig(output_path)

log_dir = 'tensorboard'

# get all the folder of log_dir
log_dir_list = os.listdir(log_dir)
file_path = ""
df_total = pd.DataFrame()
for folder in log_dir_list:
    # get the path of each folder
    number_id = folder.split('_')[-1][2:]
    id = "id" + f"{number_id.zfill(2)}"
    folder_path = os.path.join(log_dir, folder, 'eval')
    if not os.path.exists(folder_path):
        continue
    # get the file that end with eval_summary.csv
    file_list = os.listdir(folder_path)
    for file in file_list:
        if file.endswith('eval_summary.csv'):
            file_path = os.path.join(folder_path, file)
            break

    if file_path == "":
        continue

    # read the file and load the data in pandas
    df = pd.read_csv(file_path)
    # get the last row of the data
    df = df.tail(1)
    # Convert the numeric value to max 2 decimal places
    df = df.round(4)

    # change the epsidoe column to id
    df['episode'] = id

    # add the data to the total dataframe
    df_total = df_total.append(df, ignore_index=True)
    file_path = ""

# Sort the dataframe by episode
df_total = df_total.sort_values(by=['episode'])

# save the total dataframe to csv
output_path = os.path.join(log_dir, 'eval_summary.csv')
print(f"\nSave summary in {output_path}")
df_total.to_csv(output_path, index=False)
create_histogram(df_total, os.path.join(log_dir, 'center_dev_histogram.png'))