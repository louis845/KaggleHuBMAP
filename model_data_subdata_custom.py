import model_data_manager

if __name__ == '__main__':
    # restrict to dataset 1
    info = model_data_manager.data_information.loc[model_data_manager.data_information["dataset"] == 1]
    info_dset2 = model_data_manager.data_information.loc[model_data_manager.data_information["dataset"] == 2]

    wsi1 = info.loc[info["source_wsi"] == 1]
    wsi2 = info.loc[info["source_wsi"] == 2]
    dataset1 = info.loc[info["dataset"] == 1]

    condition1 = (wsi1["i"] + wsi1["j"]) < 49152
    condition2 = wsi2["j"] >= 22528
    condition_extra = model_data_manager.data_information.loc[model_data_manager.data_information["dataset"] == 2]
    condition_extra = list(condition_extra.loc[(condition_extra["source_wsi"] != 4) | (condition_extra["j"] > 16384)].index)

    split1 = list(wsi1.loc[condition1].index) + list(wsi2.loc[condition2].index)
    split2 = list(wsi1.loc[~condition1].index) + list(wsi2.loc[~condition2].index)
    split1_extra = split1 + condition_extra
    split2_extra = split2 + condition_extra
    dataset1_extra = list(model_data_manager.data_information.loc[((model_data_manager.data_information["dataset"] == 2)
                                                                | (model_data_manager.data_information["dataset"] == 1))].index)

    split1 = model_data_manager.sort_index_list(split1)
    split2 = model_data_manager.sort_index_list(split2)
    split1_extra = model_data_manager.sort_index_list(split1_extra)
    split2_extra = model_data_manager.sort_index_list(split2_extra)
    dataset1_extra = model_data_manager.sort_index_list(dataset1_extra)

    model_data_manager.create_subdata("dataset1_regional_split1", split1, "Created by model_data_subdata_custom.py. Restrict to dataset 1.")
    model_data_manager.create_subdata("dataset1_regional_split2", split2, "Created by model_data_subdata_custom.py. Restrict to dataset 1.")
    model_data_manager.create_subdata("dataset1_regional_split1_extra", split1_extra, "Created by model_data_subdata_custom.py. Restrict to dataset 1.")
    model_data_manager.create_subdata("dataset1_regional_split2_extra", split2_extra, "Created by model_data_subdata_custom.py. Restrict to dataset 1.")
    model_data_manager.create_subdata("dataset1", list(dataset1.index), "Created by model_data_subdata_custom.py. Restrict to dataset 1.")
    model_data_manager.create_subdata("dataset1_extra", list(dataset1_extra.index), "Created by model_data_subdata_custom.py. Dataset 1 combined with dataset 2.")