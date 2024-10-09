import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from data_preparation import SeismicDataset, create_dataloader, check_dataset
from data_preparation import PreloadedSeismicDataset, create_dataloader, check_dataset

from config import SEGFAST_FILE_PATH, TARGET_IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
import logging
import os
from datetime import datetime
from tqdm import tqdm


# Создаем директорию для логов, если она не существует
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Настройка логирования
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(log_dir, f"training_log_{current_time}.log")

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Это позволит выводить логи и в консоль, и в файл
    ]
)
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Определение устройства с учетом macOS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Using device: {device}")


# Определение архитектуры модели
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=32):
        super(UNet, self).__init__()
        # Временное вложение
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Энкодер
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Декодер
        self.dec1 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec3 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        # Преобразуем t в float и добавляем размерность
        t = t.float().unsqueeze(1)

        # Временное вложение
        t = self.time_mlp(t)

        # Расширяем t до размерности x
        t = t.view(-1, t.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])

        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))

        d1 = self.dec1(torch.cat([nn.functional.interpolate(e4, scale_factor=2), e3], 1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d1, scale_factor=2), e2], 1))
        d3 = self.dec3(torch.cat([nn.functional.interpolate(d2, scale_factor=2), e1], 1))

        return self.final(d3)


# Определение шагов диффузии
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), alphas_cumprod[:-1]])
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Перемещение всех тензоров на устройство
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
posterior_variance = posterior_variance.to(device)


# Реализация диффузионного процесса
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t.float().unsqueeze(1))
    loss = nn.MSELoss()(noise, predicted_noise) # need to change to MSE
    return loss


# Обновленная функция обучения с одним прогресс-баром tqdm
def train(model, dataloader, optimizer, num_epochs):
    model.train()
    total_batches = len(dataloader) * num_epochs
    progress_bar = tqdm(total=total_batches, desc="Training Progress", unit="batch")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.unsqueeze(1).to(device)  # Добавляем размерность канала
            t = torch.randint(0, num_timesteps, (batch.shape[0],), device=device)
            loss = p_losses(model, batch, t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Обновляем прогресс-бар
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Epoch": f"{epoch + 1}/{num_epochs}",
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (progress_bar.n % len(dataloader) + 1):.4f}"
            })

        # Логируем среднюю потерю за эпоху
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")

        # if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"diffusion_model.pth")
        logging.info(f"Model saved at epoch {epoch + 1}")

    progress_bar.close()


# Обновленная основная функция
def main():
    # Загрузка данных с использованием PreloadedSeismicDataset
    print("Loading dataset...")
    dataset = PreloadedSeismicDataset(SEGFAST_FILE_PATH, target_size=TARGET_IMAGE_SIZE)
    check_dataset(dataset, BATCH_SIZE)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE)

    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Number of batches per epoch: {len(dataloader)}")

    # Инициализация модели
    print("Initializing model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Обучение модели
    print("Starting training...")
    train(model, dataloader, optimizer, NUM_EPOCHS)


if __name__ == "__main__":
    main()