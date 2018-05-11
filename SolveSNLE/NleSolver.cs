using System;

namespace SolveSNLE
{
    class NleSolver
    {
        public static double SolveNewton(Func<double, double> fun, double a, double b, double eps)
        {
            int N = 40;
            double tmpa = 0.0, tmpb = 0.0;
            double x = Double.MaxValue;
            while (tmpa != a | tmpb != b)
            {
                tmpa = a;
                tmpb = b;
                Localize(fun, ref a, ref b, N);
                double newx = Newton(fun, a, b, eps);
                x = Math.Abs(newx) < Math.Abs(x) ? newx : x;
                a = b;
                b = tmpb;
            }
            return x;
        }
        private static double Newton(Func<double, double> fun, double a, double b, double eps)
        {
            double dFun(double arg, double dx) => (fun(arg + dx) - fun(arg - dx)) / (2 * dx);
            double ddFun(double arg, double dx) => (dFun(arg + dx, dx) - dFun(arg - dx, dx)) / (2 * dx);

            double x = fun(a) * ddFun(a, eps) > 0 ? a : b;

            double y = 0.0;
            do
            {
                y = x;
                x = x - fun(x) / dFun(x, eps);
            } while (Math.Abs(y - x) >= eps);
            return x;
        }
        private static void Localize(Func<double, double> fun, ref double a, ref double b, int N)
        {
            double h = (b - a) / N;
            double tx = a, tx1 = a + h;
            for (int i = 2; i < N; ++i)
            {
                tx = tx1;
                tx1 = a + i * h;
                if (fun(tx) * fun(tx1) < 0)
                {
                    a = tx;
                    b = tx1;
                    break;
                }
            }
        }
    }
}
