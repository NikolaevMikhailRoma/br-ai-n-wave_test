import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_diffusion_model import UNet
from config import TARGET_IMAGE_SIZE, MODEL_PATH
import pickle
import numpy as np

# Определение устройства с поддержкой MPS для Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Параметры диффузии (должны совпадать с параметрами при обучении)
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

# Перемещаем все тензоры на выбранное устройство
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
posterior_variance = posterior_variance.to(device)


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = betas[t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    sqrt_recip_alphas_t = sqrt_recip_alphas[t]

    # Предсказание шума моделью
    model_output = model(x, t.float())

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = posterior_variance[t]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    b = shape[0]
    # Начинаем с чистого шума
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, num_timesteps)), desc='Sampling loop time step', total=num_timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


def denormalize(seismic_slice, normalization_params):
    # Выбираем максимальное значение из параметров нормализации
    abs_max = max(normalization_params.values())
    return seismic_slice * abs_max

def generate_seismic(model, shape):
    generated = p_sample_loop(model, shape)[-1]
    # Денормализация
    with open('normalization_params.pkl', 'rb') as f:
        normalization_params = pickle.load(f)
    denormalized = denormalize(generated, normalization_params)
    return denormalized


def main():
    # Загрузка модели
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Генерация шума
    noise = torch.randn(1, 1, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1]).to(device)

    # Генерация сейсмической записи
    seismic = generate_seismic(model, noise.shape)

    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(noise[0, 0].cpu().numpy(), cmap='seismic', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title("Input Noise")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2.imshow(seismic[0, 0], cmap='seismic', aspect='auto')
    ax2.set_title("Generated Seismic Record")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig("generated_seismic.png")
    print("Generated seismic record saved as 'generated_seismic.png'")

    # Сохранение сгенерированных данных
    np.save("generated_seismic.npy", seismic[0, 0])
    print("Generated seismic data saved as 'generated_seismic.npy'")


if __name__ == "__main__":
    main()