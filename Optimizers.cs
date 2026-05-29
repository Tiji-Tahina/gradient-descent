abstract class BaseOptimizer
{
    double[] Run(IModel model, double[] w, double[][] X, double[] grad, ref double[] state, double lr);
    // common loop: init w, loop epochs, shuffle, loop samples, gradient, Update, log
    protected abstract void Step(double[] w, double[] grad, ref double[] state, double lr);
}

class GD : BaseOptimizer
{
    // Step: w[j] -= lr * grad[j] (full batch gradient)
}

class SGD : BaseOptimizer
{
    // Step: w[j] -= lr * grad[j] (per-sample gradient)
}

class Momentum : BaseOptimizer
{
    // Step: v[j] = momentum * v[j] - lr * grad[j]; 
    // w[j] += v[j]
}

class NAG : BaseOptimizer
{
    // Step: look-ahead w[j] - β*v[j]; grad at look-ahead; then momentum update
}