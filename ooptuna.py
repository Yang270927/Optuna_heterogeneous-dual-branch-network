import optuna
import pickle
import pywt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torchinfo import summary
from tqdm import tqdm

# === 加载数据 ===

# data_path = 'D:\\Project\\RML2016.10a_dict\\RML2016.10a_dict.pkl'#RML2016.10a
# data_path = 'D:\\Project\\RML.2016.10b\\RML2016.10b.dat\\RML2016.10b.dat'#RML2016.10b
data_path = 'D:\\Project\\RML\\2018_8.dat'
Xd = pickle.load(open(data_path, 'rb'), encoding='bytes')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for _ in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)
X_i = X[:, 0, :]
X_q = X[:, 1, :]
X_complex = X[:, 0, :] + 1j * X[:, 1, :]
SNR = np.array(lbl)[:, 1].astype(str).astype(int) #220000
labels =np.array(lbl)[:, 0].astype(str) #220000

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)
SNRS = torch.tensor(SNR, dtype=torch.float32)
# === 裕量计算 ===
def calculate_cl(signal):
    x_abs = np.abs(signal)
    numerator = np.max(x_abs)
    denominator = (np.mean(np.sqrt(x_abs))) ** 2
    return numerator / denominator

# === 1. 平均 CL 值按 SNR 分组 ===
unique_snrs = np.unique(SNR)
average_margins = []
for snr in unique_snrs:
    indices = np.where(SNR == snr)[0]
    cl_vals = [calculate_cl(X_complex[i]) for i in indices]
    average_margins.append(np.mean(cl_vals))

# === 2. KMeans 聚类 SNR 的平均 CL 值 ===
kmeans = KMeans(n_clusters=3,random_state=0)
cluster_labels = kmeans.fit_predict(np.array(average_margins).reshape(-1, 1))
cluster_centers = kmeans.cluster_centers_.flatten()

# import matplotlib.pyplot as plt
# colors = ['blue', 'orange', 'green']
# bar_colors = [colors[label] for label in cluster_labels]
# plt.figure(figsize=(12, 6))
# # 创建柱状图
# plt.bar(unique_snrs, average_margins, color=bar_colors)
#
# # 添加标题和标签
# plt.title('Clustering result of margin value as a function of SNR')
# plt.xlabel(' SNRs')
# plt.ylabel(' Margins')
# plt.xticks(unique_snrs)
#
# # 显示图形
# plt.show()


# === 3. 统一聚类标签顺序===
sorted_indices = np.argsort(cluster_centers)
level_mapping = {old_label: level for old_label, level in zip(sorted_indices, [2,2,1])}   #[2,2,1]a

# === 4. 为每个 SNR 分配分解层数 ===
snr_to_layer = {snr: level_mapping[cluster_labels[i]] for i, snr in enumerate(unique_snrs)}
snr_to_layer = {}




def wavelet_reconstruction(signal, level,wavelet_basis):

    coeffs = pywt.wavedec(signal, wavelet_basis, level=level)


    n = len(coeffs[-1])  # N 信号的长度 128
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(n)) * 0.5


    filtered_detail_coeffs = [pywt.threshold(coeff, threshold, mode='soft') for coeff in coeffs]  #软阈值方法进行去噪 得到处理后的系数

    low_frequency_reconstruction = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet_basis)
    high_frequency_reconstruction = pywt.waverec([np.zeros_like(coeffs[0])] + filtered_detail_coeffs[1:], wavelet_basis)

    return low_frequency_reconstruction, high_frequency_reconstruction

def create_dataloader_with_wavelet_transform(wavelet_basis, snr_to_layer,SNR, X_i, X_q, encoded_labels,batch_size=400):
    low_freqs_i = []
    high_freqs_i = []
    low_freqs_q = []
    high_freqs_q = []

    # 执行小波去噪和重建
    for i in range(len(X_i)):
        snr = SNR[i]
        level = snr_to_layer[snr]
        signal_i = X_i[i]
        signal_q = X_q[i]


        def zscore_iq(signal_i, signal_q):
            # 归一化实部
            mean_i = np.mean(signal_i)
            std_i = np.std(signal_i) + 1e-8
            signal_i_norm = (signal_i - mean_i) / std_i

            # 归一化虚部
            mean_q = np.mean(signal_q)
            std_q = np.std(signal_q) + 1e-8
            signal_q_norm = (signal_q - mean_q) / std_q

            return signal_i_norm, signal_q_norm
        signal_i,signal_q = zscore_iq(signal_i,signal_q) #归一化



        low_freq_i, high_freq_i = wavelet_reconstruction(signal_i, level,wavelet_basis)
        low_freq_q, high_freq_q = wavelet_reconstruction(signal_q, level,wavelet_basis)

        low_freqs_i.append(low_freq_i)
        high_freqs_i.append(high_freq_i)
        low_freqs_q.append(low_freq_q)
        high_freqs_q.append(high_freq_q)

    low_freqs_i = np.array(low_freqs_i)
    high_freqs_i = np.array(high_freqs_i)
    low_freqs_q = np.array(low_freqs_q)
    high_freqs_q = np.array(high_freqs_q)

    # 组合低频和高频信号
    low_freqs_reshaped = np.stack((low_freqs_i, low_freqs_q), axis=1)  # (220000,2,128)
    high_freqs_reshaped = np.stack((high_freqs_i, high_freqs_q), axis=1)  # (220000,2,128)



    data_low = torch.tensor(low_freqs_reshaped, dtype=torch.float32)  # 220000
    data_high = torch.tensor(high_freqs_reshaped, dtype=torch.float32)
    labels = encoded_labels
    snrS = torch.tensor(SNR, dtype=torch.float32)

    total_size = len(data_low)

    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size  # 确保总和正确

    # 随机打乱索引
    indices = torch.randperm(total_size)

    # 划分数据
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建 TensorDataset
    dataset = TensorDataset(data_low, data_high, labels, snrS)

    # 根据索引划分数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class LowFeatureExtractor(nn.Module):
    def __init__(self, in_channels=2):
        super(LowFeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)
        self.residual2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.MaxPool1d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64)
        )
        self.relu = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        self.residual3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        # 第二层
        residual = self.residual2(x)  # 使用 1x1 卷积 + pool 对齐尺寸
        x = self.conv2(x)
        x = self.pool2(x)
        x = x + residual
        x = self.relu(x)

        # 第三层
        residual = self.residual3(x)  # 只通道对齐，无 pool
        x = self.conv3(x)
        x = x + residual
        x = self.relu(x)
        x = self.pool3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x
class HighFeatureExtractor(nn.Module):
    def __init__(self, in_channels=2):
        super(HighFeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)
        self.residual2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.MaxPool1d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(2)
        self.residual3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.MaxPool1d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool1d(2)
        self.residual4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.MaxPool1d(2)
        )

        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.pool1(x)

        # 第二层残差连接
        residual = self.residual2(x)  # 残差分支用定义好的模块
        x = self.conv2(x)
        x = self.pool2(x)
        x = x + residual
        x = self.relu(x)

        # 第三层残差连接
        residual = self.residual3(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x + residual
        x = self.relu(x)

        # 第四层残差连接
        residual = self.residual4(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = x + residual
        x = self.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # print(f"Input shape to LSTM: {x.shape}")
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = attn_weights * lstm_out
        context = torch.sum(context, dim=1)
        out = self.classifier(context)
        return out

class MultiModalNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalNet, self).__init__()
        self.low = LowFeatureExtractor(in_channels=2)
        self.high = HighFeatureExtractor(in_channels=2)
        self.lstm_attention = LSTMAttention(192, 128, 1, num_classes)  #128


    def forward(self, x_low, x_high):
        x_low = self.low(x_low)
        x_high = self.high(x_high)
        x = torch.cat((x_low, x_high), dim=1)
        x = x.unsqueeze(1)  # Adding a sequence dimension
        x = self.lstm_attention(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalNet(len(label_encoder.classes_)).to(device)

try:
    summary(model, input_size=[(32,2, 1024), (32,2, 1024)])

except RuntimeError as e:
    print(f"Failed to run torchinfo summary: {e}")


def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, optimizer, criterion, scheduler, patience=5, trial=None):
    best_model_wts = None
    best_val_acc = 0.0
    epochs_without_improve = 0
    save_dir = "best_model_20180" #best_model_2016BN


    for epoch in range(num_epochs):
        # ======== Train =========
        model.train()
        running_loss = 0.0
        for inputs_LOW, inputs_HIGH, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", dynamic_ncols=True):
            inputs_LOW, inputs_HIGH, labels = inputs_LOW.to(device), inputs_HIGH.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs_LOW, inputs_HIGH)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Avg Train Loss: {avg_train_loss:.4f}")

        # ======== Validate =========
        val_acc = evaluate_model(model, val_loader,"Validation", plot=False,value =False )

        # Learning rate scheduler step
        if scheduler:
            scheduler.step(val_acc)

        print(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ======= Save best model =======
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            epochs_without_improve = 0
            print("✅ Best model updated.")

            if trial:
                # torch.save(best_model_wts, f"best_model_trial_{trial.number}.pth")
                torch.save(best_model_wts, os.path.join(save_dir, f"best_model_trial_{trial.number}.pth"))

                # torch.save(best_model_wts, "best_model.pth")

        else:
            epochs_without_improve += 1

        # Early stopping
        if epochs_without_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    print(f"🏁 Best Validation Accuracy: {best_val_acc:.4f}")

    # # ======= Final Test =========
    # print("\n🎯 Evaluating on test set...")
    # evaluate_model(model, test_loader, "Test", plot=True)

    return best_val_acc



def evaluate_model(model, dataloader, dataset_name, plot, value, save_dir='figure_20180'):
    model.eval()
    total = correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    snr_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0, 'y_true': [], 'y_pred': []})
    all_y_true, all_y_pred = [], []


    with torch.no_grad():
        for inputs_LOW, inputs_HIGH, labels, snrs in dataloader:
            inputs_LOW, inputs_HIGH, labels = inputs_LOW.to(device), inputs_HIGH.to(device), labels.to(device)
            outputs = model(inputs_LOW, inputs_HIGH)

            loss = criterion(outputs, labels)  # <-- 添加 loss 计算
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)

            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for idx, snr in enumerate(snrs):
                snr_value = snr.item()
                snr_accuracy[snr_value]['y_true'].append(labels[idx].item())
                snr_accuracy[snr_value]['y_pred'].append(predicted[idx].item())
                snr_accuracy[snr_value]['correct'] += (predicted[idx] == labels[idx]).item()
                snr_accuracy[snr_value]['total'] += 1

    # 计算总体准确率
    avg_loss = total_loss / total
    overall_acc = correct / total
    print(f"{dataset_name} Validation Loss: {avg_loss:.4f}")
    print(f"{dataset_name} Overall Accuracy: {overall_acc:.4f}")

    if value:
        ACC, SNR = [], []
        for snr_value in sorted(snr_accuracy.keys()):
            data = snr_accuracy[snr_value]
            acc = data['correct'] / data['total']
            precision = precision_score(data['y_true'], data['y_pred'], average='weighted')
            recall = recall_score(data['y_true'], data['y_pred'], average='weighted')
            f1 = f1_score(data['y_true'], data['y_pred'], average='weighted')

            print(f"SNR {snr_value}: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            ACC.append(acc)
            SNR.append(snr_value)

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(SNR, ACC, linewidth=1, color="red", marker="o")
            plt.xlabel("SNR(dB)")
            plt.ylabel("accuracy")
            plt.grid()
            y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            plt.yticks(y)
            plt.xticks(SNR)
            plt.title(f'{dataset_name} recognition accuracy')
            plt.savefig(f'{save_dir}/{dataset_name}_recognition accuracy')

            # plt.show()
    if plot:
        # label = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM'] #2016
        label = ['16QAM', '64QAM', '8PSK', 'AM-DSB-SC', 'AM-SSB-SC', 'BPSK', 'GMSK', 'QPSK']
        # 生成总混淆矩阵
        cm = confusion_matrix(all_y_true, all_y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        disp.plot(cmap=plt.cm.Blues)
        plt.xticks(rotation=45)
        plt.title(f'{dataset_name} Total Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{dataset_name}_total_confusion.png')

        # 生成每个SNR下的混淆矩阵
        for snr_value in sorted(snr_accuracy.keys()):
            data = snr_accuracy[snr_value]
            cm_snr = confusion_matrix(data['y_true'], data['y_pred'])
            disp_snr = ConfusionMatrixDisplay(confusion_matrix=cm_snr, display_labels=label)
            disp_snr.plot(cmap=plt.cm.Blues)
            filename = f"{save_dir}/{dataset_name}_confusion_SNR_{int(snr_value)}dB.png"
            plt.xticks(rotation=45)
            plt.title(f'{dataset_name} Confusion Matrix at SNR={snr_value}dB')
            plt.tight_layout()
            plt.savefig(filename)


    return overall_acc



def objective(trial):
    # 1.选择小波基函数
    wavelet_basis = trial.suggest_categorical('wavelet_basis', ['db1','db3','db4', 'db6','db8', 'sym2', 'sym4', 'sym8','coif1', 'coif2','coif3', 'bior1.3', 'bior3.5', 'rbio1.3'])


    train_loader, val_loader, test_loader = create_dataloader_with_wavelet_transform(wavelet_basis,snr_to_layer,SNR, X_i, X_q, encoded_labels)

    model = MultiModalNet(len(label_encoder.classes_)).to(device)

    num_epochs = 200
    learning_rate = 0.001
    patience = 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, threshold=0.1,verbose=True)

    acc = train_and_evaluate(model, train_loader, val_loader, test_loader,
                             num_epochs, optimizer, criterion, scheduler, patience,
                             trial=trial)

    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_trial = study.best_trial
best_params = best_trial.params
print("Best parameters:", best_params)

best_wavelet = best_params['wavelet_basis']


train_loader, val_loader, test_loader = create_dataloader_with_wavelet_transform(
    best_wavelet, snr_to_layer, SNR, X_i, X_q, encoded_labels
)

model = MultiModalNet(len(label_encoder.classes_)).to(device)


# model_path = f"best_model_trial_{best_trial.number}.pth"
model_path = os.path.join("best_model_20180", f"best_model_trial_{best_trial.number}.pth")


model.load_state_dict(torch.load(model_path))
print("Best model:", model_path)

evaluate_model(model, test_loader, "Test",plot=True,value =True)


