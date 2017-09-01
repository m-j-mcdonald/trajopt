#pragma once
#include <string>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <openrave/openrave.h>

#include "utils/basic_array.hpp"
#include "macros.h"

namespace trajopt {


namespace OR = OpenRAVE;
using OR::KinBody;
using OR::RobotBase;
using std::string;
using std::vector;
using std::map;
using namespace util;



typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DblMatrix;

typedef vector<double> DblVec;
typedef vector<int> IntVec;

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> TrajArray;
using Eigen::MatrixXd;
using Eigen::Matrix3d;

}
