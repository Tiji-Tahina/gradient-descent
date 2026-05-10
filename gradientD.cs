using System;

class GradientDescent
{
    // The function to minimize: f(x) = x² (minimum at x = 0)
    static double F(double x) => x * x;

    // Its derivative: f'(x) = 2x
    static double dF(double x) => 2 * x;

    static double Run(double start, double learningRate, int iterations)
    {
        double x = start;

        for (int i = 0; i < iterations; i++)
        {
            double gradient = dF(x);
            x -= learningRate * gradient;   // step opposite to the gradient

            Console.WriteLine($"Step {i + 1,3}: x = {x:F6}, f(x) = {F(x):F6}");
        }

        return x;
    }

    static void Main()
    {
        double minimum = Run(start: 10.0, learningRate: 0.1, iterations: 30);
        Console.WriteLine($"\nFound minimum at x ≈ {minimum:F6}");
    }
}
```

---

### How it works
```
x ← x − α · f'(x)