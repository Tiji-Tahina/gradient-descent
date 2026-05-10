using System;

class GradientDescent
{
    // Model: f(x) = w0 + w1*x1 + w2*x2  (affine function)
    static double Predict(double[] w, double[] x)
        => w[0] + w[1] * x[0] + w[2] * x[1];

    // Loss: Mean Squared Error = (1/N) * Σ (y_pred - y_true)²
    static double MSE(double[] w, double[][] X, double[] Y)
    {
        double loss = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double error = Predict(w, X[i]) - Y[i];
            loss += error * error;
        }
        return loss / X.Length;
    }

    // Gradient of MSE w.r.t. each weight:
    // ∂L/∂w0 = (2/N) * Σ (y_pred - y_true)
    // ∂L/∂w1 = (2/N) * Σ (y_pred - y_true) * x1
    // ∂L/∂w2 = (2/N) * Σ (y_pred - y_true) * x2
    static double[] Gradient(double[] w, double[][] X, double[] Y)
    {
        double[] grad = new double[3];
        for (int i = 0; i < X.Length; i++)
        {
            double error = Predict(w, X[i]) - Y[i];
            grad[0] += error;           // ∂L/∂w0  (bias)
            grad[1] += error * X[i][0]; // ∂L/∂w1
            grad[2] += error * X[i][1]; // ∂L/∂w2
        }
        for (int j = 0; j < grad.Length; j++)
            grad[j] *= 2.0 / X.Length;
        return grad;
    }

    static double[] Run(double[][] X, double[] Y, double learningRate, int iterations)
    {
        double[] w = { 0.0, 0.0, 0.0 }; // initialize weights at zero

        for (int i = 0; i < iterations; i++)
        {
            double[] grad = Gradient(w, X, Y);
            for (int j = 0; j < w.Length; j++)
                w[j] -= learningRate * grad[j]; // w ← w − α · ∇L

            if (i % 100 == 0)
                Console.WriteLine($"Step {i,4}: Loss = {MSE(w, X, Y):F6} | w = [{w[0]:F3}, {w[1]:F3}, {w[2]:F3}]");
        }
        return w;
    }

    static void Main()
    {
        // Ground truth: y = 3 + 2*x1 - 1*x2  →  target weights: [3, 2, -1]
        double[][] X = { new[] {1.0, 2.0}, new[] {3.0, 1.0}, new[] {5.0, 4.0}, new[] {2.0, 3.0} };
        double[]   Y = { 3 + 2*1 - 1*2,   3 + 2*3 - 1*1,   3 + 2*5 - 1*4,   3 + 2*2 - 1*3 };

        double[] w = Run(X, Y, learningRate: 0.01, iterations: 1000);
        Console.WriteLine($"\nLearned weights: w0={w[0]:F3}, w1={w[1]:F3}, w2={w[2]:F3}");
        Console.WriteLine(  "Target  weights: w0=3.000, w1=2.000, w2=-1.000");
    }
}
```

---

### What changed
```
Scalar x  →  weight vector  w = [w₀, w₁, w₂]
f'(x)     →  gradient vector  ∇L = [∂L/∂w₀, ∂L/∂w₁, ∂L/∂w₂]
```

The update rule is the same, now applied component-wise:
```
w ← w − α · ∇L