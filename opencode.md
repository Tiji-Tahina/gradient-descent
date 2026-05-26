# Gradient Descent — Centralized Architecture

## Current Problems

| Issue | Examples |
|---|---|
| **Duplicated code everywhere** | `Sigmoid`, `LinearCombination`, `Predict`, `Shuffle`, `GradientOneSample` copied across 5+ files |
| **Missing definitions** | `gradientD_sigmoid.cs` references `Predict` & `BCE` but doesn't define them — relies on same-class-name merging at compile time (fragile) |
| **Broken code** | `utilities.cs:43` calls `Math.Pow(yPred - y[i])` with one argument (missing exponent); missing `using System;` throughout |
| **Hardcoded data** | Every file embeds its own copy of the dataset inline |
| **No project file** | These are standalone `.cs` scripts, not a proper project |
| **Mixed naming** | `GDWithMomentum.cs`, `gradientD_Stochastic.cs` — inconsistent casing |

---

## Proposed File Layout

```
GradientDescent/
├── GradientDescent.csproj      # .NET project (target net8.0+)
├── Program.cs                  # Entry point — runs all demos
├── Utils.cs                    # Sigmoid, Shuffle, Dot product, helpers
├── Models.cs                   # IModel, LinearModel (MSE), LogisticModel (BCE)
├── Optimizers.cs               # BaseOptimizer, GD, SGD, Momentum, NAG
└── Data.cs                     # Static sample datasets
```

---

## Key Abstractions

| Abstraction | What it encapsulates |
|---|---|
| `IModel` | `Predict(w, x)`, `Loss(w, X, Y)`, `Gradient(w, X, Y)`, `GradientOne(w, x, y)` |
| `BaseOptimizer` | Common training loop: iterate epochs, shuffle, call gradient, update weights, log |
| `GD` / `SGD` / `Momentum` / `NAG` | Only the **weight update rule** — everything else inherited |

---

## File-by-File Specification

### 1. `GradientDescent.csproj`
Standard .NET console project file targeting `net8.0`.

### 2. `Utils.cs`
- `static double Sigmoid(double z)`
- `static void Shuffle<T>(T[] array, Random rng)` — Fisher-Yates
- `static double Dot(double[] w, double[] x)` — generic n-dimensional dot product (replaces hardcoded 3-weight `LinearCombination`)

### 3. `Models.cs`
```csharp
interface IModel {
    double Predict(double[] w, double[] x);
    double Loss(double[] w, double[][] X, double[] Y);
    double[] Gradient(double[] w, double[][] X, double[] Y);
    double[] GradientOne(double[] w, double[] x, double y);
}

class LinearModel : IModel {
    // Predict = Dot(w, x)
    // Loss    = MSE
}

class LogisticModel : IModel {
    // Predict = Sigmoid(Dot(w, x))
    // Loss    = BCE
}
```

### 4. `Optimizers.cs`
```csharp
abstract class BaseOptimizer {
    double[] Run(IModel model, double[][] X, double[] Y, double lr, int epochs);
    // common loop: init w, loop epochs, shuffle, loop samples, gradient, Update, log
    protected abstract void Step(double[] w, double[] grad, ref double[] state, double lr);
}

class GD : BaseOptimizer {
    // Step: w[j] -= lr * grad[j]  (full-batch gradient)
}

class SGD : BaseOptimizer {
    // Step: w[j] -= lr * grad[j]  (per-sample)
}

class Momentum : BaseOptimizer {
    // Step: v[j] = β*v[j] + (1-β)*grad[j]; w[j] -= lr * v[j]
}

class NAG : BaseOptimizer {
    // Step: look-ahead w[j] - β*v[j]; grad at look-ahead; then momentum update
}
```

### 5. `Data.cs`
Static class `SampleDatasets` returning `(double[][] X, double[] Y)` tuples:
- `AffineRegression` — 4 samples, target weights [3, 2, -1]
- `LogisticClassification` — 8 samples, binary labels

### 6. `Program.cs`
Entry point instantiating each model+optimizer combo:
- `LinearModel` + `GD` for regression demo
- `LogisticModel` + `GD` for batch logistic
- `LogisticModel` + `SGD` for stochastic logistic
- `LogisticModel` + `Momentum` for momentum demo
- `LogisticModel` + `NAG` for Nesterov demo

---

## How Duplication Disappears

| What | Before | After |
|---|---|---|
| `Sigmoid` | 5 copies | 1 copy in `Utils.cs` |
| `Shuffle` | 3 copies | 1 copy in `Utils.cs` |
| `LinearCombination` | 5 copies | 1 `Dot()` in `Utils.cs` |
| `Predict` | 4 copies | 1 per model (2 total) |
| `BCE` / `MSE` | 3 copies | 1 per loss (2 total) |
| Training loop | 5 copies | 1 in `BaseOptimizer` |
| Gradient func | 4 copies | 1 per model (2 total) |
| Dataset defs | 5 copies | 1 per dataset (2 total) |
