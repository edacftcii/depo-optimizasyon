import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

# Veri yükle
file_path = os.path.join("data", "Warehouse_and_Retail_Sales.csv")
df = pd.read_csv(file_path)

# İlk 50 ürün
product_freq = df.groupby("ITEM DESCRIPTION")["WAREHOUSE SALES"].sum().reset_index()
product_freq = product_freq.sort_values(by="WAREHOUSE SALES", ascending=False).reset_index(drop=True)
top_products = product_freq.head(50)

rack_positions = [(x, y) for y in range(5) for x in range(10)]
frequencies = top_products["WAREHOUSE SALES"].tolist()
product_names = top_products["ITEM DESCRIPTION"].tolist()

# Mesafe hesaplama
def calculate_distance(pos1, pos2=(0, 0)):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def total_weighted_distance(solution, frequencies, rack_positions):
    return sum(
        calculate_distance(rack_positions[i]) * frequencies[product_index]
        for i, product_index in enumerate(solution)
    )

# ACO parametreleri
NUM_ANTS = 30
NUM_ITERATIONS = 100
EVAPORATION_RATE = 0.3
ALPHA = 1.0  # feromon etkisi
BETA = 2.0   # sipariş sıklığı etkisi

# Feromon matrisi
tau = np.ones((50, 50))  # başlangıç feromon seviyesi

# Heuristik bilgi: sıklıkla ters orantılı (frekans yüksekse daha önemli)
eta = np.array(frequencies)
eta = eta / np.max(eta)
eta = eta + 0.01  # sıfır bölmeyi önle

best_solution = None
best_distance = float("inf")

start = time.time()

for iteration in range(NUM_ITERATIONS):
    all_solutions = []
    all_distances = []

    for ant in range(NUM_ANTS):
        unvisited = list(range(50))
        solution = []

        for i in range(50):
            probs = []
            for j in unvisited:
                if i == 0:
                    tau_ij = 1.0
                else:
                    tau_ij = tau[solution[-1]][j]
                eta_j = eta[j]
                prob = (tau_ij ** ALPHA) * (eta_j ** BETA)
                probs.append(prob)

            probs = np.array(probs)
            probs = probs / np.sum(probs)
            selected = np.random.choice(unvisited, p=probs)
            solution.append(selected)
            unvisited.remove(selected)

        dist = total_weighted_distance(solution, frequencies, rack_positions)
        all_solutions.append(solution)
        all_distances.append(dist)

        if dist < best_distance:
            best_solution = solution
            best_distance = dist

    # Feromon güncelle
    tau *= (1 - EVAPORATION_RATE)
    for sol, dist in zip(all_solutions, all_distances):
        for i in range(len(sol) - 1):
            a, b = sol[i], sol[i + 1]
            tau[a][b] += 1.0 / dist

end = time.time()
runtime = round(end - start, 2)

# Sonuç yazdır
print("\n✅ ACO - En iyi çözüm bulundu:")
print("Toplam mesafe:", round(best_distance, 2))
print("Çözüm süresi (saniye):", runtime)

for i, index in enumerate(best_solution):
    print(f"Raf {rack_positions[i]} → {product_names[index]} (Frekans: {frequencies[index]})")

# Görselleştirme
rack_layout = [["" for _ in range(10)] for _ in range(5)]
for i, index in enumerate(best_solution):
    x, y = rack_positions[i]
    rack_layout[y][x] = product_names[index][:10]

fig, ax = plt.subplots(figsize=(12, 6))
table = ax.table(cellText=rack_layout, loc='center', cellLoc='center', colWidths=[0.1]*10)
table.auto_set_font_size(False)
table.set_fontsize(8)
ax.axis('off')
plt.title("ACO ile En İyi Ürün Yerleşimi")
os.makedirs("results", exist_ok=True)
plt.savefig("results/aco_result.png", dpi=300)
plt.show()

# Özet
summary_path = os.path.join("results", "aco_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Toplam Mesafe: {round(best_distance, 2)}\n")
    f.write(f"Çözüm Süresi: {runtime} saniye\n")
    f.write("\nEn İyi Yerleşim:\n")
    for i, index in enumerate(best_solution):
        f.write(f"Raf {rack_positions[i]} → {product_names[index]} (Frekans: {frequencies[index]})\n")

print("📄 ACO sonucu başarıyla kaydedildi: results/aco_result.png ve results/aco_summary.txt")
