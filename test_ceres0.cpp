#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;

struct Curve_Factor
{
        Curve_Factor(double x, double y) : _x(x), _y(y) {}
        template <typename T>
        //操作符()是一个模板方法，返回值为bool型，接受参数依次为待优化变量和残差变量
        bool operator()(const T *const abc, T *residual) const
        {
                residual[0] = T(_y) - (abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
                return true;
        }
        const double _x, _y;
};

int main(int, char **)
{
        double a = 1.0, b = 10.0, c = 1.5;
        int N = 1000;
        double w_sigma = 1.0;
        cv::RNG rng;
        double abc[3] = {0, 0, 0};
        vector<double> x_data, y_data;

        for (int i = 0; i < N; i++)
        {
                double x = i / 100.0;
                x_data.push_back(x);
                y_data.push_back(a * x * x + b * x + c + rng.gaussian(w_sigma));
        }

        //构造最小二乘问题
        ceres::Problem problem;
        for (int i = 0; i < N; i++)
        {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<Curve_Factor, 1, 3>(new Curve_Factor(x_data[i], y_data[i])),
                    nullptr,
                    abc);
        };

        //配置求解器
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        cout << summary.BriefReport() << endl;
        for (auto a : abc)
                cout << a << " " << endl;
}
