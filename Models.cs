using System.Security.Cryptography.X509Certificates;

interface IModel
{
    double Predicate(double[] w, double[] x);
    double Loss(double[] w, double[][] x, double[] y);
    double[] Gradient(double[] w, double[][] x, double[] y);
    double[] GradientOne(double[] w, double[] x, double y);
}

class LinearModel : IModel
{
    // Predict = Dot(w, x)
    // Loss = MSE
}

class LogisticModel : IModel
{
    // Predict = Sigmoid(Dot(w, x))
    // Loss = BCE
}