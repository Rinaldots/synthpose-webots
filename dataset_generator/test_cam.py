"""Verifica se a camera fica dentro da arena (4x4m) para todas as combinacoes
de raio/elevacao. Imprime casos que saem da arena."""
import numpy as np

radius_min = 0.8
radius_max = 1.5
elevation_min_deg = 15
elevation_max_deg = 50
arena_half = 2.0  # arena 4x4 -> paredes em +-2m

print(f"Arena: {arena_half*2}x{arena_half*2}m  |  raio [{radius_min}, {radius_max}]  "
      f"|  elevacao [{elevation_min_deg}, {elevation_max_deg}] deg")
print()

worst_inside = None
worst_outside = None

for r in np.linspace(radius_min, radius_max, 20):
    for el in np.linspace(np.deg2rad(elevation_min_deg), np.deg2rad(elevation_max_deg), 20):
        h_dist = r * np.cos(el)          # distancia horizontal maxima
        z_cam  = 0.333 + r * np.sin(el)  # altura da camera (NAO em Z=0.333)
        inside = h_dist <= arena_half
        if not inside:
            if worst_outside is None or h_dist > worst_outside[0]:
                worst_outside = (h_dist, r, np.degrees(el), z_cam)
        else:
            if worst_inside is None or h_dist > worst_inside[0]:
                worst_inside = (h_dist, r, np.degrees(el), z_cam)

if worst_outside:
    h, r, el, z = worst_outside
    print(f"FORA  pior caso: r={r:.2f}m  el={el:.1f}deg  h={h:.2f}m > {arena_half}m  z_cam={z:.2f}m")
else:
    print("OK - nenhuma posicao fora da arena")

if worst_inside:
    h, r, el, z = worst_inside
    print(f"DENTRO maior h: r={r:.2f}m  el={el:.1f}deg  h={h:.2f}m  z_cam={z:.2f}m")

# Verifica FOV: a 0.8m o NAO (0.58m alto) cabe no frame 480px?
fov_v = 2 * np.arctan(480 / 640 * np.tan(np.deg2rad(30)))
frame_h_at_080 = 2 * 0.8 * np.tan(fov_v / 2)
print(f"\nFOV vertical: {np.degrees(fov_v):.1f}deg")
print(f"Altura visivel a 0.8m: {frame_h_at_080:.2f}m  (NAO ~0.58m -> {'OK' if frame_h_at_080 > 0.58 else 'PEQUENO'})")
