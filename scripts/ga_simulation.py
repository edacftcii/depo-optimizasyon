from deap import base, creator, tools, algorithms
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

# Ana dizin ve sonuç klasörü ayarları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(BASE_DIR, "..", "results")
os.makedirs(results_path, exist_ok=True)

def run_ga_simulation_set():
    # Veri seti
    data_path = os.path.join(BASE_DIR, "..", "data", "Warehouse_and_Retail_Sales.csv")
    df = pd.read_csv(data_path)

    # İlk 50 ürün
    product_freq = df.groupby("ITEM DESCRIPTION")["WAREHOUSE SALES"].sum().reset_index()
    product_freq = product_freq.sort_values(by="WAREHOUSE SALES", ascending=False).reset_index(drop=True)
    top_products = product_freq.head(50)

    rack_positions = [(x, y) for y in range(5) for x in range(10)]
    frequencies = top_products["WAREHOUSE SALES"].tolist()
    product_names = top_products["ITEM DESCRIPTION"].tolist()

    def calculate_distance(pos1, pos2=(0, 0)):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def total_weighted_distance(solution, frequencies, rack_positions):
        return sum(calculate_distance(rack_positions[i]) * frequencies[product_index]
                   for i, product_index in enumerate(solution))

    if "FitnessMin" in creator.__dict__: del creator.FitnessMin
    if "Individual" in creator.__dict__: del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    def swap_mutation(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual,

    def run_single_ga(pop_size, mutation_prob, generations=50):
        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(50), 50)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: (total_weighted_distance(ind, frequencies, rack_positions),))
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", swap_mutation, indpb=mutation_prob)

        population = toolbox.population(n=pop_size)

        start = time.time()
        final_pop, _ = algorithms.eaSimple(
            population, toolbox, cxpb=0.7, mutpb=mutation_prob,
            ngen=generations, verbose=False
        )
        end = time.time()

        best = tools.selBest(final_pop, k=1)[0]
        return {
            "mesafe": total_weighted_distance(best, frequencies, rack_positions),
            "sure": end - start
        }

    # Simülasyon senaryoları
    population_sizes = [50, 100, 200]
    mutation_rates = [0.01, 0.05, 0.1]
    trials_per_config = 5

    methods = []
    times = []
    distances = []

    for pop_size in population_sizes:
        for mut_rate in mutation_rates:
            all_distances = []
            all_times = []
            for _ in range(trials_per_config):
                result = run_single_ga(pop_size, mut_rate)
                all_distances.append(result["mesafe"])
                all_times.append(result["sure"])
            avg_distance = np.mean(all_distances)
            methods.append(f"GA | Pop: {pop_size} | Mut: {int(mut_rate * 100)}%")
            distances.append(all_distances)
            times.append(all_times)
            print(f"Pop: {pop_size}, Mut: {mut_rate:.2f} -> Avg Mesafe: {avg_distance:.2f}")

    # Grafik: Ortalama mesafe
    labels = methods
    values = [np.mean(d) for d in distances]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='darkcyan')
    plt.xticks(rotation=45, ha='right')
    plt.title("GA Simülasyon Sonuçları (Ortalama Mesafe)")
    plt.ylabel("Ortalama Ağırlıklı Mesafe")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "ga_simulasyon_sonuclari.png"), dpi=300)
    plt.show()

    # Sonuçları CSV ve PNG olarak kaydet
    df_results = pd.DataFrame({
        "Yöntem": methods,
        "Ortalama Süre (saniye)": [round(np.mean(t), 2) for t in times],
        "Ortalama Mesafe": [round(np.mean(m), 2) for m in distances],
        "En İyi Çözüm": [round(np.min(m), 2) for m in distances]
    })

    df_results.to_csv(os.path.join(results_path, "karsilastirma_sonuclari.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_results.values, colLabels=df_results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title("Yöntem Karşılaştırma Tablosu", fontweight="bold")
    plt.savefig(os.path.join(results_path, "karsilastirma_sonuclari.png"), dpi=300)
    plt.show()

    print("✅ Simülasyon ve karşılaştırma başarıyla tamamlandı. Dosyalar 'results/' klasörüne kaydedildi.")

if __name__ == "__main__":
    run_ga_simulation_set()
