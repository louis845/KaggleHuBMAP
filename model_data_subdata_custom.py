import model_data_manager

if __name__ == '__main__':
    # restrict to dataset 1
    info = model_data_manager.data_information.loc[model_data_manager.data_information["dataset"] == 1]

    wsi1 = info.loc[info["source_wsi"] == 1]
    wsi2 = info.loc[info["source_wsi"] == 2]
    dataset1 = info.loc[info["dataset"] == 1]

    condition1 = (wsi1["i"] + wsi1["j"]) < 49152
    condition2 = wsi2["j"] >= 22528

    split1 = list(wsi1.loc[condition1].index) + list(wsi2.loc[condition2].index)
    split2 = list(wsi1.loc[~condition1].index) + list(wsi2.loc[~condition2].index)

    split1 = model_data_manager.sort_index_list(split1)
    split2 = model_data_manager.sort_index_list(split2)

    model_data_manager.create_subdata("dataset1_regional_split1", split1, "Created by model_data_subdata_custom.py. Restrict to dataset 1.")
    model_data_manager.create_subdata("dataset1_regional_split2", split2, "Created by model_data_subdata_custom.py. Restrict to dataset 1.")
    model_data_manager.create_subdata("dataset1", list(dataset1.index), "Created by model_data_subdata_custom.py. Restrict to dataset 1.")