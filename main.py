import random
import csv, statistics
from dataclasses import dataclass

@dataclass(frozen=True)
class Food:
    name: str
    kcal: float       # per 100 g
    protein: float    # g per 100 g
    fat: float        # g per 100 g
    carbs: float      # g per 100 g
    cost: float       # PLN per 100 g
    vegetarian: bool
    vegan: bool
    gluten: bool
    nuts: bool

FOODS = [
    #    name                   kcal  prot  fat   carbs cost  veget  vegan  gluten nuts
    Food("Chicken breast",       165, 31.0,  3.6,  0.0, 1.20, False, False, False, False),
    Food("White rice (cooked)",  130,  2.7,  0.3, 28.0, 0.35, True,  True,  False, False),
    Food("Broccoli",              34,  2.8,  0.4,  7.0, 0.60, True,  True,  False, False),
    Food("Egg",                  155, 13.0, 11.0,  1.1, 1.00, True,  False, False, False),
    Food("Whole-grain bread",    250,  9.0,  3.5, 45.0, 0.60, True,  True,  True,  False),
    Food("Olive oil",            884,  0.0,100.0,  0.0, 2.50, True,  True,  False, False),
    Food("Salmon",               208, 20.0, 13.0,  0.0, 3.50, False, False, False, False),
    Food("Pasta (cooked)",       158,  5.8,  0.9, 31.0, 0.40, True,  True,  True,  False),
    Food("Plain yogurt",          61,  3.5,  3.3,  4.7, 0.45, True,  False, False, False),
    Food("Banana",                89,  1.1,  0.3, 23.0, 0.50, True,  True,  False, False),
    Food("Cheese",               350, 25.0, 27.0,  1.3, 2.50, True,  False, False, False),
    Food("Potatoes",              77,  2.0,  0.1, 17.0, 0.25, True,  True,  False, False),
    Food("Turkey breast",        135, 30.0,  1.5,  0.0, 1.50, False, False, False, False),
    Food("Ground beef (lean)",   176, 20.0, 10.0,  0.0, 2.00, False, False, False, False),
    Food("Pork tenderloin",      143, 26.0,  3.5,  0.0, 1.40, False, False, False, False),
    Food("Shrimp",                99, 24.0,  0.3,  0.2, 5.00, False, False, False, False),
    Food("Tofu",                  76,  8.0,  4.8,  1.9, 0.90, True,  True,  False, False),
    Food("Sweet potato",          86,  1.6,  0.1, 20.1, 0.50, True,  True,  False, False),
    Food("Avocado",              160,  2.0, 14.7,  8.5, 1.60, True,  True,  False, False),
    Food("Almonds",              579, 21.2, 49.9, 21.6, 4.00, True,  True,  False, True),
    Food("Peanut butter",        588, 25.0, 50.0, 20.0, 1.20, True,  True,  False, True),
    Food("Milk",                  61,  3.2,  3.3,  4.8, 0.30, True,  False, False, False),
    Food("Cottage cheese",        98, 11.1,  4.3,  3.4, 0.90, True,  False, False, False),
    Food("Apple",                 52,  0.3,  0.2, 13.8, 0.35, True,  True,  False, False),
    Food("Spinach",               23,  2.9,  0.4,  3.6, 0.80, True,  True,  False, False),
]

TARGET = {
    "kcal":       800, 
    "protein":    80, 
    "fat":        25, 
    "carbs":      110, 
    "cost":       15,
    "vegetarian": False,
    "vegan":      False,
    "gluten":     True,
    "nuts":       True,
} # meal requirements

W_COST = 3.0             # weight of the cost deviation in the fitness penalty
W_KCAL = 2.0             # weight of the calorie deviation in the fitness penalty
W_MACROS = 1.0           # weight of the protein/fat/carbs deviation in the fitness penalty
POP_SIZE = 100           # number of chromosomes in the population
INIT_MAX_GRAMS = 300     # max grams of a gene in the initial population only (not a limit later)
P_CROSS = 0.8            # probability of crossover for a selected pair of parents
P_MUT = 0.01             # probability of mutating each gene
MUT_DELTA = 50           # mutation range: gene changes by uniform value in e.g. [-50, +50] g
MAX_GEN = 500            # stop: max number of generations
PATIENCE = 50            # stop: no improvement of the best fitness for this many generations

DEFAULTS = {
    "pop_size":       POP_SIZE,
    "p_cross":        P_CROSS,
    "p_mut":          P_MUT,
    "mut_delta":      MUT_DELTA,
    "init_max_grams": INIT_MAX_GRAMS,
    "max_gen":        MAX_GEN,
    "patience":       PATIENCE,
} # default GA parameters

# Values tested for each studied parameter (others stay at DEFAULTS)
STUDY = {
    "p_mut":   [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "p_cross": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
}

RUNS = 20


def filter_foods(foods):
    return [fd for fd in foods
            if (not TARGET["vegetarian"] or fd.vegetarian)
            and (not TARGET["vegan"] or fd.vegan)
            and (TARGET["gluten"] or not fd.gluten)
            and (TARGET["nuts"] or not fd.nuts)]


FOODS = filter_foods(FOODS)
N = len(FOODS)


def macros(ch):
    k = p = f = c = cost = 0
    for g, food in zip(ch, FOODS):
        m = g / 100                    # multiplier relative to 100 g
        k += m * food.kcal
        p += m * food.protein
        f += m * food.fat
        c += m * food.carbs
        cost += m * food.cost
    return k, p, f, c, cost


def fitness(ch):
    k, p, f, c, cost = macros(ch)
    penalty = W_KCAL * abs(k - TARGET["kcal"]) / TARGET["kcal"]
    penalty += W_MACROS * (abs(p - TARGET["protein"]) / TARGET["protein"]
                           + abs(f - TARGET["fat"]) / TARGET["fat"]
                           + abs(c - TARGET["carbs"]) / TARGET["carbs"])
    penalty += W_COST * abs(cost - TARGET["cost"]) / TARGET["cost"]
    return 1 / (1 + penalty)


def random_chromosome(prm):
    return [random.uniform(0, prm["init_max_grams"]) if random.random() < 0.3 else 0.0
            for _ in range(N)]


def rank_selection(pop, fits):
    order = sorted(range(len(pop)), key=lambda i: fits[i])
    weights = [0] * len(pop)
    for rank, i in enumerate(order, start=1):
        weights[i] = rank
    return random.choices(pop, weights=weights, k=2)


def crossover(a, b, prm):
    if random.random() < prm["p_cross"]:
        pt = random.randint(1, N - 1)
        return a[:pt] + b[pt:], b[:pt] + a[pt:]
    return a[:], b[:]


def mutate(ch, prm):
    for i in range(N):
        if random.random() < prm["p_mut"]:
            ch[i] = max(0.0, ch[i] + random.uniform(-prm["mut_delta"], prm["mut_delta"]))
    return ch


def run_ga(prm):
    pop_size = prm["pop_size"]
    pop = [random_chromosome(prm) for _ in range(pop_size)]
    best, best_fit, stale = None, -1, 0

    for gen in range(prm["max_gen"]):
        fits = [fitness(ch) for ch in pop]

        gen_best = max(range(pop_size), key=lambda i: fits[i])
        if fits[gen_best] > best_fit:
            best, best_fit, stale = pop[gen_best][:], fits[gen_best], 0
        else:
            stale += 1
        if stale >= prm["patience"]:
            break

        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = rank_selection(pop, fits)
            c1, c2 = crossover(p1, p2, prm)
            new_pop += [mutate(c1, prm), mutate(c2, prm)]
        pop = new_pop[:pop_size]

    return best, best_fit, gen + 1


def run_study(study=STUDY, runs=RUNS, base=DEFAULTS,
              raw_file="results_raw.csv", summary_file="results_summary.csv"):
    raw_rows, summary_rows = [], []

    for param, values in study.items():
        for value in values:
            prm = {**base, param: value}       # change one parameter only
            fits, gens = [], []
            for run in range(runs):
                best, fit, g = run_ga(prm)
                k, p, f, c, cost = macros(best)
                fits.append(fit)
                gens.append(g)
                raw_rows.append([param, value, run, fit, g, k, p, f, c, cost])

            summary_rows.append([
                param, value, runs,
                statistics.mean(fits), statistics.stdev(fits) if runs > 1 else 0.0,
                max(fits), min(fits), statistics.mean(gens),
            ])
            print(f"{param}={value}: mean fitness {statistics.mean(fits):.4f}")

    with open(raw_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["param", "value", "run", "fitness", "generations",
                    "kcal", "protein", "fat", "carbs", "cost"])
        w.writerows(raw_rows)

    with open(summary_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["param", "value", "runs", "mean_fitness", "std_fitness",
                    "max_fitness", "min_fitness", "mean_generations"])
        w.writerows(summary_rows)


if __name__ == "__main__":
    run_study()
