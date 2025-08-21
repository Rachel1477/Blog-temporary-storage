import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class KaggleHouseDataset(Dataset):
    def __init__(self, csv_file, features, labels=None):
        """
        :param csv_file: pandas DataFrame
        :param features: 特征 DataFrame
        :param labels: 标签 Series (可为 None，用于 test 数据)
        """
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.features[idx]
        return self.features[idx], self.labels[idx]


def load_kaggle_house(batch_size=64):
    # 读取数据
    train_df = pd.read_csv("kaggle_house_pred_train.csv")
    test_df = pd.read_csv("kaggle_house_pred_test.csv")

    label = "SalePrice"

    # 拼接特征（去掉 Id 和 标签）
    features = pd.concat([
        train_df.drop(columns=["Id", label]),
        test_df.drop(columns=["Id"])
    ])

    # 数值型特征标准化
    numeric_features = features.dtypes[features.dtypes != "object"].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    features[numeric_features] = features[numeric_features].fillna(0)

    # 类别型特征 one-hot
    features = pd.get_dummies(features, dummy_na=True)

    # 切分回 train/test
    n_train = train_df.shape[0]
    train_features = features[:n_train]
    test_features = features[n_train:]

    train_labels = train_df[label]

    # 写回处理后的数据
    train_out = train_features.copy()
    train_out[label] = train_labels
    test_out = test_features.copy()

    train_out.to_csv("train_preprocessed.csv", index=False)
    test_out.to_csv("test_preprocessed.csv", index=False)

    # 转 PyTorch Dataset + DataLoader
    train_dataset = KaggleHouseDataset(train_df, train_features, train_labels)
    test_dataset = KaggleHouseDataset(test_df, test_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_out, test_out


if __name__ == "__main__":
    train_loader, test_loader, train_out, test_out = load_kaggle_house(batch_size=64)
    print("预处理后训练集大小:", train_out.shape)
    print("预处理后测试集大小:", test_out.shape)

    # 取一个 batch 看看
    for X, y in train_loader:
        print("X batch:", X.shape)
        print("y batch:", y.shape)
        break
