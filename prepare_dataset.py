import os
import shutil
import random

def split_train_test(source_dir, train_dir, test_dir, test_ratio=0.2):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        cls_source = os.path.join(source_dir, cls)
        cls_train = os.path.join(train_dir, cls)
        cls_test = os.path.join(test_dir, cls)

        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_test, exist_ok=True)

        images = [f for f in os.listdir(cls_source) if os.path.isfile(os.path.join(cls_source, f))]
        random.shuffle(images)

        test_count = int(len(images) * test_ratio)
        test_images = images[:test_count]
        train_images = images[test_count:]

        for img in train_images:
            shutil.copy2(os.path.join(cls_source, img), os.path.join(cls_train, img))

        for img in test_images:
            shutil.copy2(os.path.join(cls_source, img), os.path.join(cls_test, img))

    print(f"Dataset split completed. Train samples and test samples are created in {train_dir} and {test_dir} respectively.")

if __name__ == "__main__":
    source_dir = "data/train/sharp/Sign-Language-Digits-Dataset-master/Dataset"
    train_dir = "data/train_split"
    test_dir = "data/test"
    split_train_test(source_dir, train_dir, test_dir)
