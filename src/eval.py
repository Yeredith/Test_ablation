import math
import numpy as np
from scipy.signal import convolve2d

import pdb

def PSNR(pred, gt):
      
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))   
    
    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr

def SSIM(pred, gt):
    """Función para calcular SSIM entre dos imágenes bidimensionales (una banda)."""
    ssim = compute_ssim(pred, gt)  # Aquí pred y gt ya son imágenes 2D
    return ssim

    	
    
import numpy as np

def SAM(pred, gt):
    eps = 2.2204e-16
    pred = np.clip(pred, eps, None)  # Reemplazar valores cercanos a 0 para evitar problemas de división
    gt = np.clip(gt, eps, None)      # Reemplazar valores cercanos a 0 para evitar problemas de división

    nom = np.sum(pred * gt)
    denom1 = np.sqrt(np.sum(pred ** 2))
    denom2 = np.sqrt(np.sum(gt ** 2))
    
    # Calcular SAM (en radianes)
    sam = np.real(np.arccos(nom / (denom1 * denom2 + eps)))
    
    # Asegurarse de que SAM sea un arreglo antes de reemplazar valores
    if np.isnan(sam):
        sam = 0.0

    # Convertir el SAM de radianes a grados
    sam_deg = sam * 180 / np.pi
    
    return sam_deg


def SAM_pixeles(pred, gt):
    """
    Calcula el ángulo espectral (SAM) entre dos píxeles individuales.
    pred y gt deben ser arreglos de numpy de forma [bandas].
    """

    eps = 2.2204e-16  # Evitar división por cero

    # Reemplazar valores 0 por epsilon para evitar problemas de división
    pred[np.where(pred == 0)] = eps
    gt[np.where(gt == 0)] = eps

    # Calcular el producto punto entre los espectros de los dos píxeles
    nom = np.sum(pred * gt)
    
    # Calcular la magnitud de los vectores espectrales
    denom1 = np.sqrt(np.sum(pred ** 2))
    denom2 = np.sqrt(np.sum(gt ** 2))
    
    # Calcular el SAM (en radianes)
    sam = np.real(np.arccos(nom.astype(np.float32) / (denom1 * denom2 + eps)))
    
    # Reemplazar NaN por 0
    if np.isnan(sam):
        sam = 0
    
    # Convertir el SAM de radianes a grados
    sam_deg = sam * 180 / np.pi
    
    return sam_deg



def matlab_style_gauss2D(shape=np.array([11,11]),sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    siz = (shape-np.array([1,1]))/2
    std = sigma
    eps = 2.2204e-16
    x = np.arange(-siz[1], siz[1]+1, 1)
    y = np.arange(-siz[0], siz[1]+1, 1)
    m,n = np.meshgrid(x, y)
    
    h = np.exp(-(m*m + n*n).astype(np.float32) / (2.*sigma*sigma))    
    h[ h < eps*h.max() ] = 0    	
    sumh = h.sum()   	

    if sumh != 0:
        h = h.astype(np.float32) / sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=np.array([win_size,win_size]), sigma=1.5)
    window = window.astype(np.float32)/np.sum(np.sum(window))
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)).astype(np.float32) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))
 