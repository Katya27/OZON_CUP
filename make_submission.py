import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from baseline import init_efficientnet_model

MODEL_WEIGHTS = "baseline.pth"
TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"
# Определение классов и количества классов
num_classes = 5

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_efficientnet_model(device, num_classes)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
    model.eval()

    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_image_names = os.listdir(TEST_IMAGES_DIR)
    all_preds = []

    for image_name in all_image_names:
        img_path = os.path.join(TEST_IMAGES_DIR, image_name)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            all_preds.extend(output.cpu().numpy())

    predicted_classes = torch.tensor(all_preds).argmax(dim=1)
    
    binary_preds = (predicted_classes != 1).int()

    # Сохранение предсказаний в файл
    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, binary_preds):
            f.write(f"{name}\t{cl_id}\n")