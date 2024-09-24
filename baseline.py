# подключение необходимых библиотек
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import os
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from collections import Counter

class BaseDataset(Dataset):
# загружает пути к изображениям и их метки
    def __init__(self, root_dir, transform=None, balance=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {cls_name: idx for idx, cls_name in enumerate(os.listdir(root_dir))}
        self.image_paths = []
        self.labels = []
        self.numeric_label = None 

        for cls_name in os.listdir(root_dir):
            cls_folder = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        label = self.classes.get(cls_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)

                        # Проверяем, состоит ли имя файла только из цифр
                        if img_name.split('.')[0].isdigit():
                            self.numeric_label = label

        if balance:
        # использование RandomOverSampler для балансировки классов
            self.image_paths = np.array(self.image_paths)
            self.labels = np.array(self.labels)

            ros = RandomOverSampler(random_state=42)
            self.image_paths, self.labels = ros.fit_resample(self.image_paths.reshape(-1, 1), self.labels)
            self.image_paths = self.image_paths.flatten()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def compute_binary_metrics(preds, labels, numeric_label, numeric_pred):
    metrics = {'precision': [], 'recall': [], 'f1': []}

    # Получаем предсказанные классы (максимум по вероятности)
    predicted_classes = preds.argmax(dim=1)

    # Бинаризация предсказаний: 0, если предсказанный класс равен numeric_label, иначе 1
    binary_preds = (predicted_classes != numeric_pred).int()

    # Бинаризация меток: 0, если истинная метка равна numeric_label, иначе 1
    binary_labels = (labels != numeric_label).int()

    # Рассчёт метрик
    precision = precision_score(binary_labels.cpu(), binary_preds.cpu(), average='binary')
    recall = recall_score(binary_labels.cpu(), binary_preds.cpu(), average='binary')
    f1 = f1_score(binary_labels.cpu(), binary_preds.cpu(), average='binary')

    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1'].append(f1)

    return metrics

def init_efficientnet_model(device, num_classes, dropout_rate=0.5):

    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)

    # Проверяем индекс слоя для получения количества входных признаков
    num_features = model.classifier[1].in_features

    # Изменяем финальный классификационный слой
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_features, num_classes)
    )


    model = model.to(device)

    return model


def most_common_class_for_numeric_names(image_names, predicted_classes):
    numeric_labels = []

    # Проходим по списку имен изображений и соответствующим предсказанным классам
    for img_name, pred_class in zip(image_names, predicted_classes):
        # Проверяем, состоит ли имя файла только из чисел
        if img_name.split('.')[0].isdigit():
            numeric_labels.append(pred_class)

    if not numeric_labels:
        raise ValueError("Нет изображений с именами, состоящими только из чисел.")

    # Находим наиболее часто встречающийся класс
    most_common_class = Counter(numeric_labels).most_common(1)[0][0]
    return most_common_class

def get_data_loaders(train_dir, val_dir, batch_size=32):
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, shear=10),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    img_size = 224
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = BaseDataset(root_dir=train_dir, transform=train_transform, balance=True)
    val_dataset = BaseDataset(root_dir=val_dir, transform=val_transform, balance=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.01)

    numeric_label = train_loader.dataset.numeric_label
    print(f"Метка для изображений с числовыми именами: {numeric_label}")

    # Сбор имен изображений и предсказанных классов из обучающего набора данных
    image_names = [os.path.basename(path) for path in train_loader.dataset.image_paths]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        print(f'Train learn: Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Loss: {epoch_loss:.4f}')

        model.eval()
        val_running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            print('Valid')
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        print(f'val_Loss: {val_loss:.3f}')

        # Получение наиболее часто присваиваемого класса для числовых имен
        predicted_classes = [pred.argmax().item() for pred in all_preds]
        most_common_class = most_common_class_for_numeric_names(image_names, predicted_classes)
        print(f"Наиболее часто присваиваемый класс для изображений с числовыми именами: {most_common_class}")

        metrics = compute_binary_metrics(torch.tensor(all_preds), torch.tensor(all_labels), numeric_label, most_common_class)

        print(f'Precision: {metrics["precision"][0]:.3f}, Recall: {metrics["recall"][0]:.3f}, F1-Score: {metrics["f1"][0]:.3f}')

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')

    return model


TRAIN_DIR = './data/train/'
VAL_DIR = './data/val/'

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders(TRAIN_DIR, VAL_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Получаем количество классов
    num_classes = len(os.listdir(TRAIN_DIR))
    model = init_efficientnet_model(device, num_classes)

    num_epochs = 10
    # Запускаем обучение
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs)

    torch.save(model.state_dict(), 'baseline.pth')