


def print_table_model_selected(df_model_selected):
    print('start print table')
    df_model_selected.index = [
        'Zero',
        'One for all GCM-RCM pairs',
        'One for each GCM',
        'One for each RCM',
        'One for each GCM-RCM pair'
    ]
    total = 'Total'
    df_model_selected.loc[total, :] = df_model_selected.sum(axis=0)
    df_model_selected.loc[:, total] = df_model_selected.sum(axis=1)
    print(df_model_selected)
    # Transform into percentages
    total = df_model_selected.loc[total, total]
    df_model_selected *= 100 / total
    print(df_model_selected)
    # Plot line by line the latex
    print('\n\n')
    for i, row in df_model_selected.iterrows():
        print(row.name, end=" & ")
        print(" & ".join(["{}\%".format(int(round(v))) for v in row.values]), end=" ")
        if row.name == "One for each GCM-RCM pair":
            print('\\\\ \\hline \\hline')
        else:
            print('\\\\ \\hline')
    print('\n\n')



short_name_to_parametrization_number = {
    "no effect": 0,
    "is_ensemble_member": 5,
    "gcm": 1,
    "rcm": 2,
    "gcm_and_rcm": 4,
}
