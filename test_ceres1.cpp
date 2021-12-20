#include "ceres/ceres.h"
#include "opencv2/core/core.hpp"

using namespace std;

class CeresFactor : public ceres::SizedCostFunction<1, 3>
{
public:
        CeresFactor(const double x, const double y) : x_(x), y_(y) {}
        virtual ~CeresFactor() {}
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
        {
                const double a = parameters[0][0];
                const double b = parameters[0][1];
                const double c = parameters[0][2];

                residuals[0] = (a*x_*x_ + b*x_ + c) - y_;

                if (!jacobians)
                        return true;
                double *jacobian = jacobians[0];
                if (!jacobian)
                        return true;

                jacobian[0] = x_*x_;
                jacobian[1] = x_;
                jacobian[2] = 1;
                return true;
        }

private:
        const double x_;
        const double y_;
};

int main(int argc, char** argv)
{
        double a = 1.0, b = 10.0, c = 1.5; //真实参数值
        int N = 1000;
        double w_sigma = 1;
        cv::RNG rng;
        double abc[3] = {0, 0, 0};
        vector<double> x_data, y_data;
        for(int i = 0; i < N; i++)
        {
                double x = i / 100.0;
                double y = a*x*x + b*x + c + rng.gaussian(w_sigma);
                x_data.push_back(x);
                y_data.push_back(y);
        }
        ceres::Problem problem;
        for(int i = 0; i < N; i++)
        {
                ceres::CostFunction* cost_function = new CeresFactor(x_data[i], y_data[i]);
                problem.AddResidualBlock(cost_function, NULL, abc);
        }
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cout << abc[0] << "  " << abc[1] << "  " << abc[2] << endl;
        return 0;
}