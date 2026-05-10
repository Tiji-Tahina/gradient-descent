using System;

class SGDWithMomentum
{
    static Random rng = new Random(42);

    static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));

    static double LinearCombination(double[] w, double[] x) => w[0] + w[1] * x[0] + w[2] * x[1];
    static double Predict(double[] w, double[] x) => Sigmoid(LinearCombination(w, x));

    static double BCE(double[] w, double[][] X, double[] Y)
    {
        double loss = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double yHat = Predict(w, X[i]);
            loss -= Y[i] * Math.Log(yHat) + (1 - Y[i]) * Math.Log(1 - yHat);
        }
        return loss / X.Length;
    }

    static double[] GradientOneSample(double[] w, double[] x, double y)
    {
        double error = Predict(w, x) - y;
        return new double[] { error, error * x[0], error * x[1] };
    }

    static void Shuffle(int[] indices)
    {
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
    }

    static double[] Run(double[][] X, double[] Y, double learningRate, double beta, int epochs)
    {
        double[] w        = { 0.0, 0.0, 0.0 };
        double[] velocity = { 0.0, 0.0, 0.0 }; // accumulated direction — starts at rest

        int N = X.Length;
        int[] indices = new int[N];
        for (int i = 0; i < N; i++) indices[i] = i;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(indices);

            foreach (int i in indices)
            {
                double[] grad = GradientOneSample(w, X[i], Y[i]);

                for (int j = 0; j < w.Length; j++)
                {
                    velocity[j] = beta * velocity[j] + (1 - beta) * grad[j]; // exponential moving average of gradients
                    w[j] -= learningRate * velocity[j];                       // step follows the accumulated direction
                }
            }

            if (epoch % 500 == 0)
                Console.WriteLine($"Epoch {epoch,5}: Loss = {BCE(w, X, Y):F6} | w = [{w[0]:F3}, {w[1]:F3}, {w[2]:F3}]");
        }
        return w;
    }

    static void Main()
    {
        double[][] X = { new[] {0.5, 1.0}, new[] {1.5, 2.0}, new[] {3.0, 1.0}, new[] {4.0, 3.0},
                         new[] {1.0, 0.5}, new[] {2.0, 1.5}, new[] {3.5, 2.5}, new[] {5.0, 2.0} };
        double[]   Y = { 0, 0, 0, 1,
                         0, 0, 1, 1 };

        double[] w = Run(X, Y, learningRate: 0.1, beta: 0.9, epochs: 5000);
        Console.WriteLine($"\nLearned weights: w0={w[0]:F3}, w1={w[1]:F3}, w2={w[2]:F3}");

        Console.WriteLine("\nPredictions:");
        for (int i = 0; i < X.Length; i++)
        {
            double yHat = Predict(w, X[i]);
            Console.WriteLine($"  x=[{X[i][0]}, {X[i][1]}] → ŷ={yHat:F3}  (label={Y[i]})");
        }
    }
}
```

---

### What changed

One new vector, one new equation:
```
velocity ← β · velocity + (1 − β) · ∇L      // exponential moving average
w        ← w − α · velocity                  // step follows accumulated direction
```

| | SGD | SGD + Momentum |
|---|---|---|
| Driven by | raw gradient | smoothed gradient history |
| Noisy gradients | amplified | dampened |
| Consistent direction | no acceleration | builds up speed |
| New hyperparameter | — | β (typically 0.9) |

**The physical intuition** — a ball rolling down a hill:
```
β = 0.0  →  no memory, pure SGD (ball has no mass)
β = 0.9  →  strong memory, smooth acceleration (ball builds momentum)
β = 1.0  →  infinite memory, gradient ignored (ball never stops)