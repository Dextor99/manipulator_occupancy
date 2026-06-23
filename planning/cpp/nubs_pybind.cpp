#include "NUBSTrajectory.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace
{

using Trajectory = nubs::QuinticNUBS<6>;

class PyNUBSTrajectory6D
{
private:
    Trajectory trajectory_;
    bool generated_ = false;

    void requireGenerated() const
    {
        if (!generated_)
        {
            throw std::runtime_error("trajectory has not been generated");
        }
    }

public:
    void generate(const Eigen::MatrixXd &inner_points,
                  const Eigen::MatrixXd &head_state,
                  const Eigen::MatrixXd &tail_state,
                  const Eigen::VectorXd &durations)
    {
        Eigen::MatrixXd control_points;
        trajectory_.generate(inner_points, head_state, tail_state,
                             durations, control_points);
        generated_ = true;
    }

    Eigen::VectorXd evaluate(const double time, const int derivative_order) const
    {
        requireGenerated();
        if (!std::isfinite(time))
        {
            throw std::invalid_argument("time must be finite");
        }
        if (derivative_order < 0 || derivative_order > 5)
        {
            throw std::invalid_argument("derivative_order must be in [0, 5]");
        }
        return trajectory_.evaluate(time, derivative_order);
    }

    py::tuple sample(const Eigen::VectorXd &times, const int max_derivative) const
    {
        requireGenerated();
        if (max_derivative < 0 || max_derivative > 3)
        {
            throw std::invalid_argument("max_derivative must be in [0, 3]");
        }
        const Eigen::Index count = times.size();
        Eigen::MatrixXd q = Eigen::MatrixXd::Zero(count, 6);
        Eigen::MatrixXd qd = Eigen::MatrixXd::Zero(count, 6);
        Eigen::MatrixXd qdd = Eigen::MatrixXd::Zero(count, 6);
        Eigen::MatrixXd jerk = Eigen::MatrixXd::Zero(count, 6);

        for (Eigen::Index i = 0; i < count; ++i)
        {
            const double t = times(i);
            if (!std::isfinite(t))
            {
                throw std::invalid_argument("all sample times must be finite");
            }
            q.row(i) = trajectory_.evaluate(t, 0).transpose();
            if (max_derivative >= 1)
            {
                qd.row(i) = trajectory_.evaluate(t, 1).transpose();
            }
            if (max_derivative >= 2)
            {
                qdd.row(i) = trajectory_.evaluate(t, 2).transpose();
            }
            if (max_derivative >= 3)
            {
                jerk.row(i) = trajectory_.evaluate(t, 3).transpose();
            }
        }
        return py::make_tuple(q, qd, qdd, jerk);
    }

    py::tuple energyAndGradient() const
    {
        requireGenerated();
        double energy = 0.0;
        Eigen::MatrixXd grad_points;
        Eigen::VectorXd grad_times;
        trajectory_.getEnergyAndFiniteDiffGrad(
            energy, grad_points, grad_times);
        return py::make_tuple(energy, grad_points, grad_times);
    }

    py::tuple energyAndGradientFull() const
    {
        requireGenerated();
        double energy = 0.0;
        Eigen::MatrixXd grad_points;
        Eigen::VectorXd grad_times;
        trajectory_.getEnergyAndFiniteDiffGradFull(
            energy, grad_points, grad_times);
        return py::make_tuple(energy, grad_points, grad_times);
    }

    double energy() const
    {
        requireGenerated();
        return trajectory_.getEnergy();
    }

    double totalDuration() const
    {
        requireGenerated();
        return trajectory_.getTotalDuration();
    }

    Eigen::VectorXd durations() const
    {
        requireGenerated();
        return trajectory_.getDurations();
    }

    Eigen::MatrixXd controlPoints() const
    {
        requireGenerated();
        return trajectory_.getControlPoints();
    }
};

} // namespace

PYBIND11_MODULE(_nubs_cpp, module)
{
    module.doc() = "6-DOF minimum-jerk NUBSTrajectory bindings for CCRO-NUBS";
    module.attr("SYSTEM_ORDER") = 3;
    module.attr("DEGREE") = 5;
    module.attr("DIMENSION") = 6;

    py::class_<PyNUBSTrajectory6D>(module, "NUBSTrajectory6D")
        .def(py::init<>())
        .def("generate", &PyNUBSTrajectory6D::generate,
             py::arg("inner_points"), py::arg("head_state"),
             py::arg("tail_state"), py::arg("durations"))
        .def("evaluate", &PyNUBSTrajectory6D::evaluate,
             py::arg("time"), py::arg("derivative_order") = 0)
        .def("sample", &PyNUBSTrajectory6D::sample,
             py::arg("times"), py::arg("max_derivative") = 3)
        .def("energy", &PyNUBSTrajectory6D::energy)
        .def("energy_and_gradient", &PyNUBSTrajectory6D::energyAndGradient)
        .def("energy_and_gradient_full",
             &PyNUBSTrajectory6D::energyAndGradientFull)
        .def_property_readonly("total_duration",
                               &PyNUBSTrajectory6D::totalDuration)
        .def_property_readonly("durations", &PyNUBSTrajectory6D::durations)
        .def_property_readonly("control_points",
                               &PyNUBSTrajectory6D::controlPoints);
}
