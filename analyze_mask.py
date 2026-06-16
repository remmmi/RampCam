#!/usr/bin/env python3
import cv2
import numpy as np

# Charger le masque
mask_img = cv2.imread('masque.png')
print(f"Taille du masque: {mask_img.shape}")

# Convertir en RGB pour analyser les couleurs
mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

# Créer les masques pour les zones vertes et rouges
green_mask = np.all(mask_rgb == [0, 255, 0], axis=2)  # Vert pur
red_mask = np.all(mask_rgb == [255, 0, 0], axis=2)    # Rouge pur

print(f"Pixels verts (à surveiller): {np.sum(green_mask)}")
print(f"Pixels rouges (à ignorer): {np.sum(red_mask)}")
print(f"Autres pixels: {mask_img.shape[0] * mask_img.shape[1] - np.sum(green_mask) - np.sum(red_mask)}")

# Créer un masque binaire final (1 = surveiller, 0 = ignorer)
surveillance_mask = green_mask.astype(np.uint8)
cv2.imwrite('surveillance_mask.png', surveillance_mask * 255)
print("Masque de surveillance sauvé dans surveillance_mask.png")
