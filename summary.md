# Codebase Audit — 2026-06-01

## Cleanup Completed

Deleted `ToBeRestructured/` directory (6 old monolithic scripts) and removed the `<Compile Remove>` line from `.csproj`.

## Current Files

| File | Lines | Purpose |
|---|---|---|
| `Program.cs` | 42 | Entry point — runs all 5 demos |
| `Utilities.cs` | 45 | `Sigmoid`, `Dot`, `Shuffle`, `MSE`, `BCE` |
| `Models.cs` | 87 | `IModel`, `LinearModel` (MSE), `LogisticModel` (BCE) |
| `Optimizers.cs` | 121 | `BaseOptimizer`, `GD`, `SGD`, `Momentum`, `NAG` |
| `data.cs` | 26 | `SampleDatasets.Linear`, `SampleDatasets.Logistic` |

---

## Assessment

**Build status:** ✅ 0 warnings, 0 errors
**Runtime:** ✅ All 5 demos converge correctly

### Issues Found

#### 1. Dead code — `Utilities.MSE()` and `Utilities.BCE()`

`Utilities.cs:18-35` defines:

```csharp
public static double MSE(double[] yTrue, double[] yPred) { ... }
public static double BCE(double[] yTrue, double[] yPred) { ... }
```

Neither method is called anywhere. The models compute loss directly in their own `Loss()` methods (`LinearModel.Loss` computes MSE inline; `LogisticModel.Loss` computes BCE inline). The utilities are vestigial — perhaps intended for external use or for the base class, but currently unused.

**Fix options:**
- Remove them if they're truly dead code
- Refactor models to delegate to `Utilities.MSE` / `Utilities.BCE` to avoid duplication

---

#### 2. Empty `NAG.Step()` override

`Optimizers.cs:120`:

```csharp
protected override void Step(double[] w, double[] grad, ref double[] state, double lr) { }
```

`NAG` overrides the entire `Run()` method to implement its look-ahead logic, so `Step` is never called by the base class. The base class contract (`BaseOptimizer.Run` → `Step`) is effectively ignored. This is fragile — if someone later modifies `BaseOptimizer.Run`, `NAG` won't benefit because it doesn't use it.

**Fix options:**
- Refactor Nesterov's look-ahead into `Step()` and remove the `Run()` override (more work, cleaner OO)
- Replace the no-op with `throw new NotImplementedException()` to signal the intentional override

---

#### 3. Filename casing mismatch — `data.cs` vs `Data.cs`

The `architecture.md` specifies `Data.cs` (PascalCase), but the actual file is `data.cs`. This is purely cosmetic — the C# compiler doesn't care about filenames — but it's inconsistent with the convention used in other files (`Program.cs`, `Models.cs`, `Optimizers.cs`, `Utilities.cs` all use PascalCase).

**Fix:** `git mv data.cs Data.cs`

---

#### 4. Missing `using System;` in `data.cs`

`data.cs` omits the `using System;` directive that every other file includes. It compiles because:
- `double` is a built-in alias for `System.Double`
- Tuple syntax (`(T, U)`) is a language feature backed by `System.ValueTuple` (part of the runtime, not a namespace import)

Still inconsistent. Harmless but worth adding for uniformity.

---

### Architecture Doc Comparison

The `architecture.md` layout matches reality:

```
GradientDescent/
├── GradientDescent.csproj      ✅
├── Program.cs                  ✅
├── Utilities.cs                ✅  (though MSE/BCE unused)
├── Models.cs                   ✅
├── Optimizers.cs               ✅  (NAG.Step is a no-op)
└── Data.cs                     ⚠️  file is lowercase `data.cs`
```
