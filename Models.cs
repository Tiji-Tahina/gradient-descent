using System;

// The models are used to make the gradients. They then define the surface we are going to traverse

interface IModel
{
    double Predict(double[] w, double[] x);
    double Loss(double[] w, double[][] x, double[] y);
    double[] Gradient(double[] w, double[][] x, double[] y);
    double[] GradientOne(double[] w, double[] x, double y);
}

class LinearModel : IModel
{
    public double Predict(double[] w, double[] x) => Utilities.Dot(w, x);

    public double Loss(double[] w, double[][] X, double[] Y)
    {
        double sum = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double err = Predict(w, X[i]) - Y[i];
            sum += err * err;
        }
        return sum / X.Length;
    }

    public double[] Gradient(double[] w, double[][] X, double[] Y)
    {
        double[] grad = new double[w.Length];
        for (int i = 0; i < X.Length; i++)
        {
            double err = Predict(w, X[i]) - Y[i];
            for (int j = 0; j < w.Length; j++)
                grad[j] += err * X[i][j];
        }
        for (int j = 0; j < w.Length; j++)
            grad[j] *= 2.0 / X.Length;
        return grad;
    }

    public double[] GradientOne(double[] w, double[] x, double y)
    {
        double err = Predict(w, x) - y;
        // simple declaration de la variable grad que l'on va utiliser plus tard
        double[] grad = new double[w.Length];
        for (int j = 0; j < w.Length; j++)
            grad[j] = 2.0 * err * x[j];
        return grad;
    }
}

class LogisticModel : IModel
{
    public double Predict(double[] w, double[] x) => Utilities.Sigmoid(Utilities.Dot(w, x));

    public double Loss(double[] w, double[][] X, double[] Y)
    {
        double sum = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double yHat = Predict(w, X[i]);
            sum -= Y[i] * Math.Log(yHat) + (1 - Y[i]) * Math.Log(1 - yHat);
        }
        return sum / X.Length;
    }

    public double[] Gradient(double[] w, double[][] X, double[] Y)
    {
        double[] grad = new double[w.Length];
        for (int i = 0; i < X.Length; i++)
        {
            double err = Predict(w, X[i]) - Y[i];
            for (int j = 0; j < w.Length; j++)
                grad[j] += err * X[i][j];
        }
        for (int j = 0; j < w.Length; j++)
            grad[j] /= X.Length;
        return grad;
    }

    public double[] GradientOne(double[] w, double[] x, double y)
    {
        double err = Predict(w, x) - y;
        double[] grad = new double[w.Length];
        for (int j = 0; j < w.Length; j++)
            grad[j] = err * x[j];
        return grad;
    }
}
