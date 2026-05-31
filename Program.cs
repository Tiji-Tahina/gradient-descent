using System;

class Program
{
    static void Main()
    {
        Console.WriteLine("=== Linear Regression + GD ===");
        {
            var (X, Y) = SampleDatasets.Linear;
            var w = new GD().Run(new LinearModel(), X, Y, 0.01, 1000);
            Console.WriteLine($"Weights: [{string.Join(", ", w)}]");
        }

        Console.WriteLine("\n=== Logistic Regression + GD ===");
        {
            var (X, Y) = SampleDatasets.Logistic;
            var w = new GD().Run(new LogisticModel(), X, Y, 0.1, 2000);
            Console.WriteLine($"Weights: [{string.Join(", ", w)}]");
        }

        Console.WriteLine("\n=== Logistic Regression + SGD ===");
        {
            var (X, Y) = SampleDatasets.Logistic;
            var w = new SGD().Run(new LogisticModel(), X, Y, 0.1, 2000);
            Console.WriteLine($"Weights: [{string.Join(", ", w)}]");
        }

        Console.WriteLine("\n=== Logistic Regression + Momentum ===");
        {
            var (X, Y) = SampleDatasets.Logistic;
            var w = new Momentum().Run(new LogisticModel(), X, Y, 0.1, 2000);
            Console.WriteLine($"Weights: [{string.Join(", ", w)}]");
        }

        Console.WriteLine("\n=== Logistic Regression + NAG ===");
        {
            var (X, Y) = SampleDatasets.Logistic;
            var w = new NAG().Run(new LogisticModel(), X, Y, 0.1, 2000);
            Console.WriteLine($"Weights: [{string.Join(", ", w)}]");
        }
    }
}
