import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

# FFHQ (7978, 3604) images: train, validation, test [3244, 360, 908]
# FFHQ (94660, 42767) images: train, validation, test [38490, 4277, 10641]
# FFHQ (21416, 9676) images: train, validation, test [8708, 968, 2404]
#
# celeba (4804, 4136) images: train, validation [3722, 414]
# celeba (39943, 39275) images: train, validation [35347, 3928]

internal_sources = [
    "Palette",
    "VQGAN",
    "StableDiffusion",
    "STYLEGAN",
    "StyleGAN2",
    "stylegan3",
    "DCFACe",
    "dfdc",
    "celebdf",
    "ff++",
    "dfd",
]


external_sources = ["metfaces", "celeba", "casia_filtered"]
val_split = 0.1  # 10% data saved for validation

external_sampler = {
    "celeba": {
        "Palette": [3722, 414],
        "VQGAN": [35347, 3928],
    },
    "metfaces": {"Palette": [601, 67], "VQGAN": [601, 67]},
    "casia": {"DCFACe": [382007, 42445]},
}

internal_sampler = {
    "FFHQ": {
        "STYLEGAN": [3244, 360, 908],
        "StyleGAN2": [38490, 4277, 10641],
        "stylegan3": [8708, 968, 2404],
    },
    "wiki": {"StableDiffusion": [23833, 2648, 6668]},
}
columns_metadata = [
    "source",
    "image_path",
    "original_path",
    "Predicted Gender",
    "Predicted Age",
    "Skin Tone",
    "Target",
    "Skin Tone Group",
    "Intersection",
]


def add_images(
    list_of_images,
    metadata_file,
    dataset_name,
    src_dataset_path,
    dest_dataset_path,
    resume=True,
    offset=1000,
):
    annotations_df = pd.DataFrame(columns=columns_metadata)
    save_annotation = False
    leading_zeros = len(str(len(list_of_images)))
    print(f"Leading zeros: {leading_zeros}")
    labels = [0] * (len(columns_metadata) - 3)
    idx_offset = 1000
    if resume and os.path.exists(metadata_file):
        existing_annotations = pd.read_csv(metadata_file)
        idx_offset = len(existing_annotations)
        del existing_annotations
    # Add images to the dataset
    for idx, image in enumerate(list_of_images):
        if idx % offset == 0 and idx >= 0 and idx >= offset:
            save_annotation = True
            # print(f"Processed {idx} images")
        elif idx == len(list_of_images) - 1:
            save_annotation = True

        src_path = os.path.join(src_dataset_path, image)
        new_file_name = f"{dataset_name}_{idx_offset + idx + 1:0{leading_zeros}d}{Path(image).suffix}"
        shutil.copy(src_path, dest_dataset_path)
        os.rename(
            os.path.join(dest_dataset_path, Path(image).name),
            os.path.join(dest_dataset_path, new_file_name),
        )
        annotations_df.loc[idx] = [
            dataset_name,
            os.path.join(dest_dataset_path, new_file_name),
            src_path,
        ] + labels
        if save_annotation:
            print(f"Processed {idx + 1}/{len(list_of_images)} images")
            if os.path.exists(metadata_file):
                existing_annotations = pd.read_csv(metadata_file)
                annotations_df = pd.concat([existing_annotations, annotations_df])
                del existing_annotations
            annotations_df.to_csv(metadata_file, index=False)
            annotations_df = pd.DataFrame(columns=columns_metadata)
            save_annotation = False


def process_real_data(dataset_name, sampler="external"):
    dataset_dir = os.environ.get("DATASET_PATH", "datasets/AI_Face_imagesV2")
    dest_dataset_path = os.environ.get("DATASET", "datasets/ai_face")
    src_images = None
    sampling_details = None
    test_split = False
    src_test_images = None
    if sampler == "external":
        src = os.path.join(sampler, dataset_name)
        if dataset_name == "casia":
            src = os.path.join(sampler, "casia_filtered", "images")
        sampling_details = external_sampler[dataset_name]
        src_images = os.listdir(src)
        np.random.shuffle(src_images)
        for model, sample_sizes in sampling_details.items():
            t_metadata_file = os.path.join("metadata", "train", f"{dataset_name}.csv")
            v_metadata_file = os.path.join("metadata", "val", f"{dataset_name}.csv")
            dest_train = os.path.join(dest_dataset_path, model, "train", "real")
            dest_val = os.path.join(dest_dataset_path, model, "val", "real")
            os.makedirs(dest_train, exist_ok=True)
            os.makedirs(dest_val, exist_ok=True)
            train_images = src_images[: sample_sizes[0]]
            src_images = src_images[sample_sizes[0] :]
            print(
                f"adding {len(train_images)} images to train, remaining {len(src_images)}"
            )
            add_images(train_images, t_metadata_file, dataset_name, src, dest_train)
            val_images = src_images[: sample_sizes[1]]
            src_images = src_images[sample_sizes[1] :]
            print(
                f"adding {len(val_images)} images to validation, remaining {len(src_images)}"
            )
            add_images(val_images, v_metadata_file, dataset_name, src, dest_val)

            if test_split and src_test_images:
                test_metadata_file = os.path.join(
                    "metadata", "test", f"{dataset_name}.csv"
                )
                test_images = src_test_images[: sample_sizes[2]]
                dest_test = os.path.join(dest_dataset_path, model, "test", "real")
                os.makedirs(dest_test, exist_ok=True)
                add_images(
                    test_images, test_metadata_file, dataset_name, src, dest_test
                )

    elif sampler == "internal":
        test_split = True
        src = dataset_dir
        sampling_details = internal_sampler[dataset_name]
        train_df = pd.read_csv(os.path.join("data", "train.csv"))
        filter_str = (
            f"imdb_wiki/{dataset_name}" if dataset_name == "wiki" else dataset_name
        )
        custom_train_df = train_df[train_df["Image Path"].str.contains(filter_str)]
        del train_df
        # shuffle
        custom_train_df = custom_train_df.sample(frac=1, random_state=42).reset_index(
            drop=True
        )
        test_df = pd.read_csv(os.path.join("data", "test.csv"))
        custom_test_df = test_df[test_df["Image Path"].str.contains(filter_str)]
        del test_df
        # shuffle
        custom_test_df = custom_test_df.sample(frac=1, random_state=42).reset_index(
            drop=True
        )
        for model, sample_sizes in sampling_details.items():
            train_images = custom_train_df.head(sample_sizes[0])
            process_data(
                train_images,
                dataset_name,
                "train",
                offset=1000,
                resume=False,
                label="real",
                destination=model,
            )
            custom_train_df = custom_train_df.iloc[sample_sizes[0] :]
            val_images = custom_train_df.head(sample_sizes[1])
            process_data(
                val_images,
                dataset_name,
                "val",
                offset=1000,
                resume=True,
                label="real",
                destination=model,
            )
            custom_train_df = custom_train_df.iloc[sample_sizes[1] :]
            test_images = custom_test_df.head(sample_sizes[2])
            process_data(
                test_images,
                dataset_name,
                "test",
                offset=1000,
                resume=False,
                label="real",
                destination=model,
            )
            custom_test_df = custom_test_df.iloc[sample_sizes[2] :]

    else:
        raise ValueError("Invalid sampler")
    if src_images is None or sampling_details is None:
        return


def process_data(
    df, dataset_name, mode, offset=1000, resume=False, label="fake", destination=None
):
    if destination is None:
        destination = dataset_name
    dataset_dir = os.environ.get("DATASET_PATH", "datasets/AI_Face_imagesV2")
    output_dir = os.environ.get("DATASET", "datasets/ai_face")
    dest_dir = os.path.join(output_dir, destination, mode, label)
    os.makedirs(dest_dir, exist_ok=True)
    relative_path = f"{destination}/{mode}/{label}"
    column_names = ["source", "image_path", "original_path"]
    column_names.extend(df.columns.tolist()[1:])
    annotation_df = pd.DataFrame(columns=column_names)
    leading_zeros = len(str(len(df)))
    idx_offset = 0
    existing_images = []
    metadata_file = f"metadata/{mode}/{dataset_name}.csv"
    save_annotation = False
    processed_images = 0
    df_unprocessed = deepcopy(df)
    if resume and os.path.exists(metadata_file):
        existing_annotations = pd.read_csv(metadata_file)
        idx_offset = len(existing_annotations)
        print(f"Resuming from index {idx_offset}")
        existing_images = existing_annotations["original_path"].tolist()
        df_unprocessed = df[~df["Image Path"].isin(existing_annotations["image_path"])]
        print(idx_offset, len(df_unprocessed))
        del existing_annotations

    # add training data
    for idx, row in df_unprocessed.iterrows():
        image_path = row["Image Path"]
        if image_path in existing_images:
            continue

        if processed_images == offset:
            save_annotation = True
        if processed_images % offset == 0 and processed_images > offset:
            save_annotation = True
        if processed_images == df.shape[0] - 1:
            save_annotation = True

        new_file_name = f"{dataset_name}_{mode}_{(idx_offset + 1):0{leading_zeros}d}{Path(image_path).suffix}"

        file_to_copy = os.path.join(dataset_dir, image_path)
        # source_dir = image_path.split("/")[1]
        try:
            shutil.copy(file_to_copy, dest_dir)
            os.rename(
                os.path.join(dest_dir, Path(file_to_copy).name),
                os.path.join(dest_dir, new_file_name),
            )
            file_path = f"{relative_path}/{new_file_name}"

            annotation_df.loc[processed_images] = [
                dataset_name,
                file_path,
                image_path,
            ] + row[1:].tolist()
            processed_images += 1
            idx_offset += 1
            if save_annotation:
                if os.path.exists(metadata_file):
                    existing_annotations = pd.read_csv(metadata_file)
                    annotation_df = pd.concat([existing_annotations, annotation_df])
                    del existing_annotations
                annotation_df.to_csv(metadata_file, index=False)
                annotation_df = pd.DataFrame(columns=column_names)
                save_annotation = False
                print(f"Processed {processed_images}/{df.shape[0]} images")
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    # train_df = pd.read_csv(os.path.join("data", "train.csv"))
    test_df = pd.read_csv(os.path.join("data", "test.csv"))
    # sprint("Processing training data...")
    # for src in internal_sources:
    #     filter_string = f"deepfakes/{src}" if src == "ff++" else src
    #     custom_train_df = train_df[train_df["Image Path"].str.contains(filter_string)]
    #     print(f"Processing {src} with {custom_train_df.shape[0]} training images")
    #     custom_test_df = test_df[test_df["Image Path"].str.contains(filter_string)]
    #     print(f"Processing {src} with {custom_test_df.shape[0]} test images")
    #     real_df = pd.DataFrame(custom_train_df[custom_train_df["Target"] == 0])
    #     if real_df.shape[0] == 0:
    #         print(f"No real images found in {src}")
    #     else:
    #         print(f"Processing {src} with {real_df.shape[0]} real images")
    #         r_val_size = round(real_df.shape[0] * val_split)
    #         r_train_size = real_df.shape[0] - r_val_size
    #         print(
    #             f"Splitting {src} real images into {r_train_size} training and {r_val_size} validation"
    #         )
    #         # add training data
    #         process_data(
    #             real_df.head(r_train_size), dataset_name=src, mode="train", label="real"
    #         )
    #         # add validation data
    #         process_data(
    #             real_df.tail(r_val_size), dataset_name=src, mode="val", label="real"
    #         )
    #     fake_df = pd.DataFrame(custom_train_df[custom_train_df["Target"] == 1])
    #     if fake_df.shape[0] == 0:
    #         print(f"No fake images found in {src}")
    #     else:
    #         print(f"Processing {src} with {fake_df.shape[0]} fake images")
    #         f_val_size = round(fake_df.shape[0] * val_split)
    #         f_train_size = fake_df.shape[0] - f_val_size
    #         print(
    #             f"Splitting {src} fake images into {f_train_size} training and {f_val_size} validation"
    #         )
    #         # add training data
    #         process_data(
    #             fake_df.head(f_train_size), dataset_name=src, mode="train", label="fake"
    #         )
    #         # add validation data
    #         process_data(
    #             fake_df.tail(f_val_size), dataset_name=src, mode="val", label="fake"
    #         )
    #     # add test data
    #     print(f"Processing {src} test data")
    #     t_real_df = pd.DataFrame(custom_test_df[custom_test_df["Target"] == 0])
    #     if t_real_df.shape[0] == 0:
    #         print(f"No real images found in {src}")
    #     else:
    #         print(f"Processing real images from {src} test data")
    #         process_data(t_real_df, dataset_name=src, mode="test", label="real")
    #     t_fake_df = pd.DataFrame(custom_test_df[custom_test_df["Target"] == 1])
    #     if t_fake_df.shape[0] == 0:
    #         print(f"No fake images found in {src}")
    #     else:
    #         print(f"Processing fake images from {src} test data")
    #         process_data(t_fake_df, dataset_name=src, mode="test", label="fake")

    # print("Processing external data")
    # for sampler in external_sampler:
    #     process_real_data(sampler, "external")

    print("Processing internal real data")
    for sampler in internal_sampler:
        process_real_data(sampler, "internal")
