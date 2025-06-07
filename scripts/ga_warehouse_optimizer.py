import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import os
import matplotlib.pyplot as plt
import time

# Veri seti yolu
data_path = os.path.join("data", "Warehouse_and_Retail_Sales.csv")
df = pd.read_csv(data_path)

# Kontrol
print("✅ Veri başarıyla yüklendi.")
print(df.head())
print("Sütunlar:", df.columns)
print("Eksik veriler:\n", df.isnull().sum())

# Sipariş frekansları
product_freq = df.groupby("ITEM DESCRIPTION")["WAREHOUSE SALES"].sum().reset_index()
product_freq = product_freq.sort_values(by="WAREHOUSE SALES", ascending=False).reset_index(drop=True)
top_products = product_freq.head(50)

# Raf pozisyonları
rack_positions = [(x, y) for y in range(5) for x in range(10)]
frequencies = top_products["WAREHOUSE SALES"].tolist()
product_names = top_products["ITEM DESCRIPTION"].tolist()

def calculate_distance(pos1, pos2=(0, 0)):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def total_weighted_distance(solution, frequencies, rack_positions):
    total = 0
    for i, product_index in enumerate(solution):
        distance = calculate_distance(rack_positions[i])
        freq = frequencies[product_index]
        total += distance * freq
    return total

# DEAP ayarları
for attr in ["FitnessMin", "Individual"]:
    if attr in creator.__dict__:
        delattr(creator, attr)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def run_ga_with_crossover(crossover_operator, title_suffix, output_filename):
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(50), 50)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return (total_weighted_distance(individual, frequencies, rack_positions),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", crossover_operator)

    def swap_mutation(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual,

    toolbox.register("mutate", swap_mutation, indpb=0.05)

    population = toolbox.population(n=100)
    start = time.time()
    result_population, _ = algorithms.eaSimple(
        population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, verbose=True
    )
    end = time.time()

    best = tools.selBest(result_population, k=1)[0]
    best_distance = evaluate(best)[0]
    runtime = round(end - start, 2)

    print(f"\n✅ {title_suffix} - En iyi yerleşim çözümü:")
    print("Toplam ağırlıklı mesafe:", best_distance)
    print("Çözüm süresi (saniye):", runtime)

    for i, index in enumerate(best):
        print(f"Raf {rack_positions[i]} → {product_names[index]} (Sipariş Frekansı: {frequencies[index]})")

    # Görselleştirme
    rack_layout = [["" for _ in range(10)] for _ in range(5)]
    for i, index in enumerate(best):
        x, y = rack_positions[i]
        rack_layout[y][x] = product_names[index][:10]

    fig, ax = plt.subplots(figsize=(12, 6))
    table = ax.table(cellText=rack_layout, loc='center', cellLoc='center', colWidths=[0.1]*10)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.axis('off')
    plt.title(f"En İyi Ürün Yerleşimi ({title_suffix})")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{output_filename}", dpi=300)
    plt.show()

    return title_suffix, runtime, best_distance

# GA'yi OX ve PMX ile çalıştır, sonuçları topla
results = []
results.append(run_ga_with_crossover(tools.cxOrdered, "Order Crossover (OX)", "OX.png"))
results.append(run_ga_with_crossover(tools.cxPartialyMatched, "Partially Mapped Crossover (PMX)", "PMX.png"))

# Karşılaştırma tablosu oluştur ve kaydet
df_comp = pd.DataFrame(results, columns=["Yöntem", "Çözüm Süresi (saniye)", "Toplam Mesafe"])
os.makedirs("results", exist_ok=True)
df_comp.to_csv("results/crossover_karsilastirma.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_comp.values, colLabels=df_comp.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.title("Crossover Yöntem Karşılaştırması", fontweight="bold")
plt.savefig("results/crossover_karsilastirma.png", dpi=300)
plt.show()

print("✅ Karşılaştırma tamamlandı ve dosyalar 'results/' klasörüne kaydedildi.")
