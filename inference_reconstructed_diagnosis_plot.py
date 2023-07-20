import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

import model_data_manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot of diagnosis of a multiclass reconstructed U-Net model")
    model_data_manager.transform_add_argparse_arguments(parser, requires_model=False)
    args = parser.parse_args()
    input_data_loader, output_data_writer, model_path, subdata_entries, train_subdata_entries, val_subdata_entries = model_data_manager.transform_get_argparse_arguments(args)

    diagnosis_group = input_data_loader.data_store["diagnosis"]  # h5py.Group
    # loop through the keys of the diagnosis group. they are float32 np arrays, with shape (N,), load them and plot histograms
    diagnosis_keys = list(diagnosis_group.keys())
    for key in diagnosis_keys:
        diagnosis = diagnosis_group[key][()]
        diagnosis = diagnosis.astype(np.float32)
        plt.figure(figsize=(19.2, 10.8))
        sns.histplot(diagnosis)
        plt.title(key)
        plt.xlabel("Diagnosis")
        plt.ylabel("Frequency")
        plt.show()

    input_data_loader.close()
    output_data_writer.close()
    print("Plotting Complete")
