#
# import h5py
# import pickle
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
#
# # 加载数据集
# with open(r'D:\Project\AMR-Benchmark-main\AMR-Benchmark-main\RML201610a\RML2016.10a_dict.pkl', 'rb') as f:
#     dataset = pickle.load(f, encoding='latin1')
#
# # 提取特征数据、标签和信噪比
# X = []  # 特征数据容器
# Y = []  # 标签容器
# Z = []  # SNR容器
#
# # 假设数据在同一字典中，并且我们有一个容器存储它
# for (modulation, snr), signals in dataset.items():
#     for signal in signals:
#         X.append(signal)  # 存储信号数据
#         Y.append(modulation)  # 存储调制类型
#         Z.append(snr)  # 存储信噪比
#
# # 转换为 NumPy 数组
# X = np.array(X)
# Y = np.array([list(dataset.keys()).index((mod, snr)) for mod, snr in zip(Y, Z)])  # 标签转为索引
# Z = np.array(Z)
#
# # 划分训练集和验证集
# X_train, X_val, Y_train, Y_val, Z_train, Z_val = train_test_split(X, Y, Z, test_size=0.2, random_state=42)
#
# # 创建 HDF5 文件并写入数据
# with h5py.File('train_new.h5', 'w') as train_file:
#     train_file.create_dataset('X', data=X_train)
#     train_file.create_dataset('Y', data=Y_train)
#     train_file.create_dataset('Z', data=Z_train)
#
# with h5py.File('val_new.h5', 'w') as val_file:
#     val_file.create_dataset('X', data=X_val)
#     val_file.create_dataset('Y', data=Y_val)
#     val_file.create_dataset('Z', data=Z_val)
#
# print("train_new.h5 和 val_new.h5 文件已生成。")

import pickle
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# 设定路径
pkl_path = r'D:\Project\RML\2018_8.dat'
train_h5_path = './train_new.h5'
val_h5_path = './val_new.h5'

# 加载pkl文件
with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 确认数据结构
print(f"Type of data: {type(data)}")

# 通常RML2016.10a.pkl是一个dict，键是 (modulation, snr)，值是对应的IQ样本
# 例如：('BPSK', -20): (1000, 2, 128)

# 整理数据
X = []  # IQ samples
Y = []  # modulation labels
Z = []  # SNR labels

modulations = sorted(list(set([key[0] for key in data.keys()])))
modulation2idx = {mod: idx for idx, mod in enumerate(modulations)}
print("Modulation types:", modulation2idx)

for (mod, snr) in data.keys():
    samples = data[(mod, snr)]  # shape: (N, 2, 128)
    X.append(samples)
    Y += [modulation2idx[mod]] * samples.shape[0]
    Z += [snr] * samples.shape[0]

X = np.vstack(X)  # (num_samples, 2, 128)
Y = np.array(Y)   # (num_samples,)
Z = np.array(Z)   # (num_samples,)

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
print("Shape of Z:", Z.shape)

# 划分训练/验证集 (8:2比例)
X_train, X_val, Y_train, Y_val, Z_train, Z_val = train_test_split(
    X, Y, Z, test_size=0.2, random_state=42, stratify=Y
)

print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

# 保存为train_new.h5
with h5py.File(train_h5_path, 'w') as f:
    f.create_dataset('X', data=X_train)
    f.create_dataset('Y', data=Y_train)
    f.create_dataset('Z', data=Z_train)

# 保存为val_new.h5
with h5py.File(val_h5_path, 'w') as f:
    f.create_dataset('X', data=X_val)
    f.create_dataset('Y', data=Y_val)
    f.create_dataset('Z', data=Z_val)

print("Finished writing train_new.h5 and val_new.h5 ✅")
