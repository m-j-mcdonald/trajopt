#include "utils.hpp"
#include <Eigen/Geometry>
using namespace Eigen;

namespace trajopt {

Eigen::Matrix3d toRot(const OR::Vector& rq) {
  Eigen::Affine3d T;
  T = Eigen::Quaterniond(rq[0], rq[1], rq[2], rq[3]);
  return T.rotation();
}

}
