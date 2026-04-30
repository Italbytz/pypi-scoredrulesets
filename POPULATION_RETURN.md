# Population Return Feature

## Overview

Both C# and Python versions of logicGP/GPAS now return the **final population** in addition to the best model.

### C# (LogicGP)

**GeneticProgram.Run() now returns:**
```csharp
Task<(IIndividual bestIndividual, IIndividualList population)>
```

**Usage:**
```csharp
var (bestIndividual, population) = await geneticProgram.Run();

// Access best individual
var bestFitness = bestIndividual.Fitness;

// Iterate over population
foreach (var individual in population)
{
    var genotype = individual.Genotype; // TinyGpGenotype
    var fitness = individual.Fitness;
}
```

### Python (LogicGP)

**LogicGPClassifier now stores:**
- `self._final_population`: List of `(polynomial, fitness)` tuples after `fit()`

**Usage:**
```python
clf = LogicGPClassifier(max_generations=10_000, ...)
clf.fit(X_train, y_train)

# Access final population
if hasattr(clf, '_final_population'):
    for poly, fit in clf._final_population:
        # poly: _Polynomial (the individual)
        # fit: _Fitness or _FitnessRLCW (the fitness value)
        print(f"Polynomial: {poly}, Fitness: {fit.consolidated}")
```

## Use Cases

1. **Analysis:** Examine all Pareto-optimal candidates, not just the selected model
2. **Diagnostics:** Compare fitness values across the population
3. **Ensemble:** Use multiple candidates from the final population
4. **Research:** Track diversity or convergence behavior

## Implementation Details

- **C#:** Best individual extracted via `OrderByDescending(p => p.Fitness).FirstOrDefault()`
- **Python:** Final population captured from `evaluated` list before model selection
- **Backward compatible:** Existing code continues to work (use first element of tuple in C#, ignore `_final_population` in Python)
