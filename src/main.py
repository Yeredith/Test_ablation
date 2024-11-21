import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np
from data_utils import TrainsetFromFolder, ValsetFromFolder, TestsetFromFolder
from models import SFCSR, MCNet, SFCCBAM, HYBRID_SE_CBAM

# Clase para convertir diccionarios en objetos con atributos
class ConfigNamespace:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

# Configuración desde JSON
def load_config():
    json_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(json_path, 'r') as file:
        return json.load(file)

# Selección de modelo
def select_model(config):
    model_name = config["model_name"]
    model_config = ConfigNamespace(config["models"][model_name])
    model_config.cuda = config.get("cuda", False)  # Asegurar atributo cuda

    if model_name == "SFCSR":
        return SFCSR(model_config)
    elif model_name == "MCNet":
        return MCNet(model_config)
    elif model_name == "SFCCBAM":
        return SFCCBAM(model_config)
    elif model_name == "HYBRID_SE_CBAM":
        return HYBRID_SE_CBAM(model_config)
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

# Configurar rutas de salida
def setup_output_paths(config, model_name):
    """Configura las rutas de salida para checkpoints y gráficos."""
    base_path = config["output"]["results_path"]
    model_path = os.path.join(base_path, model_name)
    checkpoints_path = os.path.join(model_path, "checkpoints")
    graphs_path = os.path.join(model_path, "graphs")

    # Crear las carpetas necesarias si no existen
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(graphs_path, exist_ok=True)

    return checkpoints_path, graphs_path  # Solo devuelve estos dos valores


def train(train_loader, model, optimizer, criterion, device, model_name):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Entrenando"):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        if model_name == "MCNet":
            # Procesa todas las bandas para MCNet
            outputs = model(inputs)  # MCNet procesa todas las bandas a la vez
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
        else:
            # Inicialización para SFCSR y SFCCBAM
            localFeats = None
            batch_loss = 0
            for i in range(inputs.shape[1]):
                if i == 0:
                    x = inputs[:, 0:3, :, :]
                    y = inputs[:, 0, :, :]
                    new_label = labels[:, 0, :, :]
                elif i == inputs.shape[1] - 1:
                    x = inputs[:, i-2:i+1, :, :]
                    y = inputs[:, i, :, :]
                    new_label = labels[:, i, :, :]
                else:
                    x = inputs[:, i-1:i+2, :, :]
                    y = inputs[:, i, :, :]
                    new_label = labels[:, i, :, :]

                output, localFeats = model(x, y, localFeats, i)
                loss = criterion(output, new_label)
                loss.backward(retain_graph=True)
                batch_loss += loss.item()

            total_loss += batch_loss / inputs.shape[1]

        optimizer.step()

    return total_loss / len(train_loader)

def val(val_loader, model, device, model_name):
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validando"):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            if model_name == "MCNet":
                # Procesa todas las bandas para MCNet
                outputs = model(inputs)
                psnr = 10 * torch.log10(1 / ((outputs - labels) ** 2).mean())
                total_psnr += psnr.item()
            else:
                # Inicialización para SFCSR y SFCCBAM
                localFeats = None
                batch_psnr = 0
                for i in range(inputs.shape[1]):
                    if i == 0:
                        x = inputs[:, 0:3, :, :]
                        y = inputs[:, 0, :, :]
                        new_label = labels[:, 0, :, :]
                    elif i == inputs.shape[1] - 1:
                        x = inputs[:, i-2:i+1, :, :]
                        y = inputs[:, i, :, :]
                        new_label = labels[:, i, :, :]
                    else:
                        x = inputs[:, i-1:i+2, :, :]
                        y = inputs[:, i, :, :]
                        new_label = labels[:, i, :, :]

                    output, localFeats = model(x, y, localFeats, i)
                    psnr = 10 * torch.log10(1 / ((output - new_label) ** 2).mean())
                    batch_psnr += psnr.item()

                total_psnr += batch_psnr / inputs.shape[1]

    return total_psnr / len(val_loader)


def save_checkpoint(model, optimizer, checkpoints_path, epoch):
    """Guarda el estado del modelo y el optimizador en un checkpoint."""
    model_out_path = os.path.join(checkpoints_path, f"model_epoch_{epoch}.pth")
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    os.makedirs(checkpoints_path, exist_ok=True)  # Asegura que la carpeta existe
    torch.save(state, model_out_path)
    print(f"Checkpoint guardado: {model_out_path}")

import matplotlib.pyplot as plt

def load_last_checkpoint(model, optimizer, checkpoints_path):
    """Carga el último checkpoint disponible, si existe."""
    if not os.path.exists(checkpoints_path):
        return model, optimizer, 0  # No hay checkpoints, empezar desde el principio

    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoints_path) if f.startswith("model_epoch_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if not checkpoint_files:
        return model, optimizer, 0  # No hay checkpoints válidos

    last_checkpoint_path = os.path.join(checkpoints_path, checkpoint_files[-1])
    print(f"Cargando el último checkpoint: {last_checkpoint_path}")
    checkpoint = torch.load(last_checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    return model, optimizer, start_epoch

def save_plots(loss_values, psnr_values, graphs_path, epoch):
    """Genera y guarda gráficos de pérdida y PSNR."""
    os.makedirs(graphs_path, exist_ok=True)  # Asegura que la carpeta existe

    # Gráfico de pérdida
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, label="Pérdida (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Pérdida por Época")
    plt.legend()
    loss_plot_path = os.path.join(graphs_path, f"loss_plot_epoch_{epoch}.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Gráfico de pérdida guardado: {loss_plot_path}")

    # Gráfico de PSNR
    plt.figure()
    plt.plot(range(1, len(psnr_values) + 1), psnr_values, label="PSNR")
    plt.xlabel("Épocas")
    plt.ylabel("PSNR")
    plt.title("PSNR por Época")
    plt.legend()
    psnr_plot_path = os.path.join(graphs_path, f"psnr_plot_epoch_{epoch}.png")
    plt.savefig(psnr_plot_path)
    plt.close()
    print(f"Gráfico de PSNR guardado: {psnr_plot_path}")
    
from scipy.io import savemat  # Importar savemat para guardar imágenes en formato .mat
from eval import SAM, EPI, SSIM  # Asegúrate de importar las funciones correctamente

def test_model(test_loader, model, model_name, device, test_path):
    model.eval()
    test_results = {"PSNR": [], "SSIM": [], "SAM": [], "EPI": [], "Time": []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testeando {model_name}")):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # Ajustar dinámicamente band_mean solo para MCNet
            if model_name == "MCNet" and hasattr(model, 'band_mean'):
                if model.band_mean.size(1) != inputs.size(1):
                    print(f"[MCNet] Ajustando band_mean de {model.band_mean.size(1)} a {inputs.size(1)} bandas.")
                    model.band_mean = nn.Parameter(torch.zeros(1, inputs.size(1), 1, 1), requires_grad=False).to(device)

            # Realizar predicción
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            if model_name == "MCNet":
                outputs = model(inputs)
            else:  # Para SFCSR y SFCCBAM
                localFeats = None
                outputs = torch.zeros_like(labels)
                for i in range(inputs.shape[1]):
                    if i == 0:
                        x = inputs[:, 0:3, :, :]
                        y = inputs[:, 0, :, :]
                    elif i == inputs.shape[1] - 1:
                        x = inputs[:, i-2:i+1, :, :]
                        y = inputs[:, i, :, :]
                    else:
                        x = inputs[:, i-1:i+2, :, :]
                        y = inputs[:, i, :, :]
                    output, localFeats = model(x, y, localFeats, i)
                    outputs[:, i, :, :] = output

            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)

            # Convertir a formato numpy para las métricas SAM, EPI y SSIM
            outputs_np = outputs.cpu().numpy().squeeze()
            labels_np = labels.cpu().numpy().squeeze()

            # Calcular métricas
            psnr = 10 * torch.log10(1 / ((outputs - labels) ** 2).mean())
            sam = SAM(outputs_np, labels_np)
            epi = EPI(outputs_np, labels_np)

            # Calcular SSIM por banda y promediar
            ssim_per_band = [SSIM(outputs_np[i], labels_np[i]) for i in range(outputs_np.shape[0])]
            ssim = np.mean(ssim_per_band)

            # Convertir métricas a tipo float estándar
            test_results["PSNR"].append(float(psnr.item()))
            test_results["SSIM"].append(float(ssim))
            test_results["SAM"].append(float(sam))
            test_results["EPI"].append(float(epi))
            test_results["Time"].append(float(elapsed_time))

            # Guardar imagen generada como archivo .mat
            output_file = os.path.join(test_path, f"output_image_{batch_idx + 1}.mat")
            savemat(output_file, {'generated': outputs_np, 'ground_truth': labels_np})
            print(f"Imagen guardada en {output_file}")

            print(f"Imagen procesada con {model_name}: PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, SAM: {sam:.4f}, EPI: {epi:.4f}, Tiempo: {elapsed_time:.2f}ms")

    # Guardar métricas
    metrics_path = os.path.join(test_path, "metrics.json")
    with open(metrics_path, "w") as metrics_file:
        json.dump(test_results, metrics_file)

    print(f"Pruebas completadas para {model_name}. Resultados guardados en {metrics_path}.")


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["training"]["seed"])

    train_dataset = TrainsetFromFolder(os.path.join(config["database"]["base_path"], "Train", config["database"]["name"], "4"))
    val_dataset = ValsetFromFolder(os.path.join(config["database"]["base_path"], "Validation", config["database"]["name"], "4"))
    test_dataset = TestsetFromFolder(os.path.join(config["database"]["base_path"], "Test", config["database"]["name"], "4"))

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["gpu"]["num_threads"])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config["gpu"]["num_threads"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config["gpu"]["num_threads"])

    model_name = config["model_name"]
    model = select_model(config).to(device)

    if config["gpu"]["use_multi_gpu"] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=config["gpu"]["gpu_ids"])

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.L1Loss()

    checkpoints_path, graphs_path = setup_output_paths(config, model_name)
    test_path = os.path.join(config["output"]["results_path"], model_name, "test_results")
    os.makedirs(test_path, exist_ok=True)

    model, optimizer, start_epoch = load_last_checkpoint(model, optimizer, checkpoints_path)

    if start_epoch >= config["training"]["epochs"]:
        print(f"El entrenamiento para {model_name} ya está completo. Pasando directamente a las pruebas.")
        test_model(test_loader, model, model_name, device, test_path)
        return

    loss_values = []
    psnr_values = []

    for epoch in range(start_epoch + 1, config["training"]["epochs"] + 1):
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        train_loss = train(train_loader, model, optimizer, criterion, device, model_name)
        val_psnr = val(val_loader, model, device, model_name)

        loss_values.append(train_loss)
        psnr_values.append(val_psnr)

        save_checkpoint(model, optimizer, checkpoints_path, epoch)
        save_plots(loss_values, psnr_values, graphs_path, epoch)

        print(f"Epoch {epoch}, Loss: {train_loss:.4f}, PSNR: {val_psnr:.4f}")

    test_model(test_loader, model, model_name, device, test_path)



if __name__ == "__main__":
    main()
