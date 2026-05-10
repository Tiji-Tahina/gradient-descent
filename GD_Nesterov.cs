using System;

class NesterovAcceleratedGradient
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
        double[] velocity = { 0.0, 0.0, 0.0 };

        int N = X.Length;
        int[] indices = new int[N];
        for (int i = 0; i < N; i++) indices[i] = i;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(indices);

            foreach (int i in indices)
            {
                // 1. Look ahead: project where momentum would carry us
                double[] wLookAhead = new double[w.Length];
                for (int j = 0; j < w.Length; j++)
                    wLookAhead[j] = w[j] - beta * velocity[j];

                // 2. Evaluate gradient at the look-ahead position, not at w
                double[] grad = GradientOneSample(wLookAhead, X[i], Y[i]);

                // 3. Update velocity and weights as usual
                for (int j = 0; j < w.Length; j++)
                {
                    velocity[j] = beta * velocity[j] + (1 - beta) * grad[j];
                    w[j]       -= learningRate * velocity[j];
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

One conceptual shift — **where** the gradient is evaluated:
```
Momentum:  grad = ∇L( w )                  // gradient at current position
Nesterov:  grad = ∇L( w − β · velocity )   // gradient at look-ahead position
```

The three-step loop makes the idea explicit:
```
1.  wLookAhead ← w − β · velocity          // project where momentum leads
2.  grad       ← ∇L( wLookAhead )          // evaluate gradient there
3.  velocity   ← β · velocity + (1−β) · grad
    w          ← w − α · velocity