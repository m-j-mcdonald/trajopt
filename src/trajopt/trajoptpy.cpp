#include <boost/python.hpp>
#include "trajopt/collision_checker.hpp"
#include <boost/foreach.hpp>
#include "macros.h"
#include "openrave_userdata_utils.hpp"
#include "numpy_utils.hpp"
#include <limits>
#include "utils/eigen_conversions.hpp"
#include "trajopt/rave_utils.hpp"
using namespace trajopt;
using namespace Eigen;
using namespace OpenRAVE;
using std::vector;

namespace py = boost::python;

bool gInteractive = false;
py::object PyNone = py::object();

EnvironmentBasePtr GetCppEnv(py::object py_env) {
  py::object openravepy = py::import("openravepy");
  int id = py::extract<int>(openravepy.attr("RaveGetEnvironmentId")(py_env));
  EnvironmentBasePtr cpp_env = RaveGetEnvironment(id);
  return cpp_env;
}
KinBodyPtr GetCppKinBody(py::object py_kb, EnvironmentBasePtr env) {
  KinBodyPtr cpp_kb;
  if (PyObject_HasAttrString(py_kb.ptr(), "GetEnvironmentId")) {
    int id = py::extract<int>(py_kb.attr("GetEnvironmentId")());
    cpp_kb = env->GetBodyFromEnvironmentId(id);
  }
  return cpp_kb;
}
KinBody::LinkPtr GetCppLink(py::object py_link, EnvironmentBasePtr env) {
  KinBody::LinkPtr cpp_link;
  if (PyObject_HasAttrString(py_link.ptr(), "GetParent")) {
    KinBodyPtr parent = GetCppKinBody(py_link.attr("GetParent")(), env);
    int idx = py::extract<int>(py_link.attr("GetIndex")());
    cpp_link = parent->GetLinks()[idx];
  }
  return cpp_link;
}
RobotBasePtr GetCppRobot(py::object py_robot, EnvironmentBasePtr env) {
  return boost::dynamic_pointer_cast<RobotBase>(GetCppKinBody(py_robot, env));
}
RobotBase::ManipulatorPtr GetCppManip(py::object py_manip, EnvironmentBasePtr env) {
  RobotBase::ManipulatorPtr cpp_manip;
  if (PyObject_HasAttrString(py_manip.ptr(), "GetRobot")) {
    RobotBasePtr robot = GetCppRobot(py_manip.attr("GetRobot")(), env);
    cpp_manip = robot->GetManipulator(py::extract<string>(py_manip.attr("GetName")()));
  }
  return cpp_manip;
}
vector<KinBody::LinkPtr> GetCppLinks(py::object py_obj, EnvironmentBasePtr env) {
  vector<KinBody::LinkPtr> links;
  KinBodyPtr cpp_kb = GetCppKinBody(py_obj, env);
  if (!!cpp_kb) links.insert(links.end(), cpp_kb->GetLinks().begin(), cpp_kb->GetLinks().end());
  KinBody::LinkPtr cpp_link = GetCppLink(py_obj, env);
  if (!!cpp_link) links.push_back(cpp_link);
  return links;
}

py::list toPyList2(const std::vector< OpenRAVE::Vector >& x) {
  py::list out;
  for (int i=0; i < x.size(); ++i) out.append(toNdarray1<double>((double*)&x[i],3) );
  return out;
}

class PyCollision {
public:
  Collision m_c;
  PyCollision(const Collision& c) : m_c(c) {}
  float GetDistance() {return m_c.distance;}
  py::object GetNormal() {return toNdarray1<double>((double*)&m_c.normalB2A,3);}
  py::object GetPtA() {return toNdarray1<double>((double*)&m_c.ptA,3);}
  py::object GetPtB() {return toNdarray1<double>((double*)&m_c.ptB,3);}
  string GetLinkAName() {return m_c.linkA->GetName();}
  string GetLinkBName() {return m_c.linkB->GetName();}
  string GetLinkAParentName() {return m_c.linkA->GetParent()->GetName();}
  string GetLinkBParentName() {return m_c.linkB->GetParent()->GetName();}
  py::object GetCastAlphas() {return toNdarray1<float>(&m_c.mi.alpha[0], m_c.mi.alpha.size());}
  py::object GetCastSupportVertices() {return toPyList2(m_c.mi.supportPtsWorld);}
  py::object GetMultiCastAlphas() {return toNdarray1<float>(&m_c.mi.alpha[0], m_c.mi.alpha.size());}
  py::object GetMultiCastIndices() {return toNdarray1<int>(&m_c.mi.instance_ind[0], m_c.mi.instance_ind.size());}
  py::object GetMultiCastSupportVertices() {return toPyList2(m_c.mi.supportPtsWorld);}
};

py::list toPyList(const vector<Collision>& collisions) {
  py::list out;
  BOOST_FOREACH(const Collision& c, collisions) {
    out.append(PyCollision(c));
  }
  return out;
}

bool compareCollisions(const Collision& c1,const Collision& c2) { return c1.distance < c2.distance; }

class PyCollisionMatrix {
public:
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> m_col_mat;
  PyCollisionMatrix(const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& col_mat) : m_col_mat(col_mat) {}
};

class PyCollisionChecker {
public:
  py::object AllVsAll() {
    vector<Collision> collisions;
    m_cc->AllVsAll(collisions);
    return toPyList(collisions);
  }
  py::object BodyVsAll(py::object& py_kb, bool sort=true) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    KinBodyPtr cpp_kb = GetCppKinBody(py_kb, env);
    if (!cpp_kb) {
      throw openrave_exception("body isn't part of environment!");
    }
    vector<Collision> collisions;
    m_cc->BodyVsAll(*cpp_kb, collisions);
    if (sort)
      std::sort(collisions.begin(), collisions.end(), compareCollisions);
    return toPyList(collisions);
  }
  py::object BodyVsBody(py::object& py_kb1, py::object& py_kb2, bool sort=true) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    KinBodyPtr cpp_kb1 = GetCppKinBody(py_kb1, env);
    KinBodyPtr cpp_kb2 = GetCppKinBody(py_kb2, env);
    if (!cpp_kb1) {
      throw openrave_exception("body 1 isn't part of environment!");
    }
    if (!cpp_kb2) {
      throw openrave_exception("body 2 isn't part of environment!");
    }
    vector<Collision> collisions;
    m_cc->BodyVsBody(*cpp_kb1, *cpp_kb2, collisions);
    if (sort)
      std::sort(collisions.begin(), collisions.end(), compareCollisions);
    return toPyList(collisions);
  }
  py::object BodiesVsBodies(py::list& py_kbs1, py::list& py_kbs2) {
    //py::list py_kbs vs np::list py_kbs
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());

    //Convert Bodies
    int n_kbs1 = py::extract<int>(py_kbs1.attr("__len__")());
    vector<KinBodyPtr> cpp_kbs1(n_kbs1);
    for (int i=0; i < n_kbs1; ++i) {
      cpp_kbs1[i] = GetCppKinBody(py_kbs1[i], env);
      if (!cpp_kbs1[i]) {
        throw openrave_exception("One of the bodies isn't part of environment!");
      }
    }

    int n_kbs2 = py::extract<int>(py_kbs2.attr("__len__")());
    vector<KinBodyPtr> cpp_kbs2(n_kbs2);
    for (int i=0; i < n_kbs2; ++i) {
      cpp_kbs2[i] = GetCppKinBody(py_kbs2[i], env);
      if (!cpp_kbs2[i]) {
        throw openrave_exception("One of the bodies isn't part of environment!");
      }
    }

    vector< vector<Collision> > collisions(n_kbs1);
    for (int i=0; i < n_kbs1; ++i) { 
      for (int j=0; j < n_kbs2; ++j) {
        m_cc->BodyVsBody(*cpp_kbs1[i], *cpp_kbs2[j], collisions[i]);
      }
    }

    vector<Collision> f_collisions;
    for (int i=0; i < n_kbs1; ++i) {
      std::sort(collisions[i].begin(), collisions[i].end(), compareCollisions);
      if (!collisions[i].empty()) {
        f_collisions.push_back(collisions[i].front());
      }
    }
    
    return toPyList(f_collisions);
  }
  py::object KinBodyCastVsAll(py::object& py_kb, py::object& py_dv0, py::object& py_dv1, int which_dofs=DOF_X|DOF_Y|DOF_RotationAxis, py::object& rot_axis = PyNone, bool sort=true) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    KinBodyPtr cpp_kb = GetCppKinBody(py_kb, env);
    if (!cpp_kb) {
      throw openrave_exception("Kinbody isn't part of environment!");
    }
    if (cpp_kb->IsRobot()) {
      throw openrave_exception("Use RobotMultiCastVsAll for Robots!");
    }
    OR::Vector rotation_axis;
    if (rot_axis == PyNone)
      rotation_axis = OR::Vector(0,0,1);
    else
      rotation_axis = OR::Vector(py::extract<double>(rot_axis[0]), py::extract<double>(rot_axis[1]), py::extract<double>(rot_axis[2]));
    ConfigurationPtr rad = KinBodyAndDOFPtr(new KinBodyAndDOF(cpp_kb, which_dofs, rotation_axis));
    vector<Collision> collisions;
    CastVsAll(rad, py_dv0, py_dv1, collisions, sort);
    return toPyList(collisions);
  }
  py::object RobotCastVsAll(py::object& py_kb, py::object& py_dv0, py::object& py_dv1, string which_dofs="active", bool sort=true) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    KinBodyPtr cpp_kb = GetCppKinBody(py_kb, env);
    if (!cpp_kb) {
      throw openrave_exception("Robot isn't part of environment!");
    }
    if (!cpp_kb->IsRobot()) {
      throw openrave_exception("Use KinBodyMultiCastVsAll for KinBodies!");
    }
    RobotBasePtr robot = boost::dynamic_pointer_cast<RobotBase>(cpp_kb);
    ConfigurationPtr rad = RADFromName(which_dofs, robot);
    vector<Collision> collisions;
    CastVsAll(rad, py_dv0, py_dv1, collisions, sort);
    return toPyList(collisions);
  }
  void CastVsAll(ConfigurationPtr rad, py::object& py_dv0, py::object& py_dv1, vector<Collision>& collisions, bool sort) {
    int n_dofs0 = py::extract<int>(py_dv0.attr("__len__")());
    DblVec dofvals0(n_dofs0);
    for (unsigned i=0; i < n_dofs0; ++i) dofvals0[i] = py::extract<double>(py_dv0[i]);

    int n_dofs1 = py::extract<int>(py_dv1.attr("__len__")());
    DblVec dofvals1(n_dofs1);
    for (unsigned i=0; i < n_dofs1; ++i) dofvals1[i] = py::extract<double>(py_dv1[i]);

    vector<int> inds;
    std::vector<KinBody::LinkPtr> links;
    rad->GetAffectedLinks(links,true,inds);

    m_cc->CastVsAll(*rad, links, dofvals0, dofvals1, collisions, -1);
    if (sort)
      std::sort(collisions.begin(), collisions.end(), compareCollisions);
  }
  py::object KinBodyMultiCastVsAll(py::object& py_kb, py::object& py_dof_list, int which_dofs=DOF_X|DOF_Y|DOF_RotationAxis, py::object& rot_axis = PyNone, bool sort=true) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    KinBodyPtr cpp_kb = GetCppKinBody(py_kb, env);
    if (!cpp_kb) {
      throw openrave_exception("Kinbody isn't part of environment!");
    }
    if (cpp_kb->IsRobot()) {
      throw openrave_exception("Use RobotMultiCastVsAll for Robots!");
    }
    OR::Vector rotation_axis;
    if (rot_axis == PyNone)
      rotation_axis = OR::Vector(0,0,1);
    else
      rotation_axis = OR::Vector(py::extract<double>(rot_axis[0]), py::extract<double>(rot_axis[1]), py::extract<double>(rot_axis[2]));
    ConfigurationPtr rad = KinBodyAndDOFPtr(new KinBodyAndDOF(cpp_kb, which_dofs, rotation_axis));
    vector<Collision> collisions;
    MultiCastVsAll(rad, py_dof_list, collisions, sort);
    return toPyList(collisions);
  }
  py::object RobotMultiCastVsAll(py::object& py_kb, py::object& py_dof_list, string which_dofs="active", bool sort=true) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    KinBodyPtr cpp_kb = GetCppKinBody(py_kb, env);
    if (!cpp_kb) {
      throw openrave_exception("Robot isn't part of environment!");
    }
    if (!cpp_kb->IsRobot()) {
      throw openrave_exception("Use KinBodyMultiCastVsAll for KinBodies!");
    }
    RobotBasePtr robot = boost::dynamic_pointer_cast<RobotBase>(cpp_kb);
    ConfigurationPtr rad = RADFromName(which_dofs, robot);
    vector<Collision> collisions;
    MultiCastVsAll(rad, py_dof_list, collisions, sort);
    return toPyList(collisions);
  }
  void MultiCastVsAll(ConfigurationPtr rad, py::object& py_dof_list, vector<Collision>& collisions, bool sort) {
    int n_sigma_pts = py::extract<int>(py_dof_list.attr("__len__")());
    int n_dofs = 0;
    if (n_sigma_pts > 0)
      n_dofs = py::extract<int>(py_dof_list[0].attr("__len__")());
    vector<DblVec> dofvals(n_sigma_pts, DblVec(n_dofs));
    for (int i=0; i<n_sigma_pts; i++) {
      for (int j=0; j<n_dofs; j++) {
        dofvals[i][j] = py::extract<double>(py_dof_list[i][j]);
      }
    }

    vector<int> inds;
    std::vector<KinBody::LinkPtr> links;
    rad->GetAffectedLinks(links,true,inds);
    m_cc->MultiCastVsAll(*rad, links, dofvals, collisions, -1);
    if (sort)
      std::sort(collisions.begin(), collisions.end(), compareCollisions);
  }
  void SetContactDistance(float dist) {
    m_cc->SetContactDistance(dist);
  }
  double GetContactDistance() {
    return m_cc->GetContactDistance();
  }
  void ExcludeCollisionPair(py::object py_obj0, py::object py_obj1) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    
    vector<KinBody::LinkPtr> links0 = GetCppLinks(py_obj0, env);
    vector<KinBody::LinkPtr> links1 = GetCppLinks(py_obj1, env);

    BOOST_FOREACH(const KinBody::LinkPtr& link0, links0) {
      BOOST_FOREACH(const KinBody::LinkPtr& link1, links1) {
        m_cc->ExcludeCollisionPair(*link0, *link1);
      }
    }
  }
  void IncludeCollisionPair(py::object py_obj0, py::object py_obj1) {
    EnvironmentBasePtr env = boost::const_pointer_cast<EnvironmentBase>(m_cc->GetEnv());
    
    vector<KinBody::LinkPtr> links0 = GetCppLinks(py_obj0, env);
    vector<KinBody::LinkPtr> links1 = GetCppLinks(py_obj1, env);

    BOOST_FOREACH(const KinBody::LinkPtr& link0, links0) {
      BOOST_FOREACH(const KinBody::LinkPtr& link1, links1) {
        m_cc->IncludeCollisionPair(*link0, *link1);
      }
    }
  }
  PyCollisionMatrix SaveCollisionMatrix() {
    return PyCollisionMatrix(m_cc->GetCollisionMatrix());
  }
  void RestoreCollisionMatrix(PyCollisionMatrix py_col_mat) {
    m_cc->SetCollisionMatrix(py_col_mat.m_col_mat);
  }
  PyCollisionChecker(CollisionCheckerPtr cc) : m_cc(cc) {}
private:
  PyCollisionChecker();
  CollisionCheckerPtr m_cc;
};

PyCollisionChecker PyGetCollisionChecker(py::object py_env) {
  CollisionCheckerPtr cc = CollisionChecker::GetOrCreate(*GetCppEnv(py_env));
  return PyCollisionChecker(cc);
}

void CallPyFunc(py::object f) {
  f();
}

void translate_runtime_error(std::runtime_error const& e)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(BodyVsAllDefaults, PyCollisionChecker::BodyVsAll, 1, 2);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(BodyVsBodyDefaults, PyCollisionChecker::BodyVsBody, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(RobotCastVsAllDefaults, PyCollisionChecker::RobotCastVsAll, 3, 5);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(KinBodyCastVsAllDefaults, PyCollisionChecker::KinBodyCastVsAll, 3, 6);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(RobotMultiCastVsAllDefaults, PyCollisionChecker::RobotMultiCastVsAll, 2, 4);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(KinBodyMultiCastVsAllDefaults, PyCollisionChecker::KinBodyMultiCastVsAll, 2, 5);

BOOST_PYTHON_MODULE(ctrajoptpy) {

  np_mod = py::import("numpy");

  py::object openravepy = py::import("openravepy");

  string pyversion = py::extract<string>(openravepy.attr("__version__"));
  if (OPENRAVE_VERSION_STRING != pyversion) {
    PRINT_AND_THROW("the openrave on your pythonpath is different from the openrave version that trajopt links to!");
  }

  py::register_exception_translator<std::runtime_error>(&translate_runtime_error);

  py::class_<PyCollisionMatrix>("PyCollisionMatrix", py::no_init);

  py::class_<PyCollisionChecker>("CollisionChecker", py::no_init)
      .def("AllVsAll", &PyCollisionChecker::AllVsAll)
      .def("BodyVsAll", &PyCollisionChecker::BodyVsAll, BodyVsAllDefaults())
      .def("BodyVsBody", &PyCollisionChecker::BodyVsBody, BodyVsBodyDefaults())
      .def("BodiesVsBodies", &PyCollisionChecker::BodiesVsBodies)
      .def("RobotCastVsAll", &PyCollisionChecker::RobotCastVsAll, RobotCastVsAllDefaults())
      .def("KinBodyCastVsAll", &PyCollisionChecker::KinBodyCastVsAll, KinBodyCastVsAllDefaults())
      .def("RobotMultiCastVsAll", &PyCollisionChecker::RobotMultiCastVsAll, RobotMultiCastVsAllDefaults())
      .def("KinBodyMultiCastVsAll", &PyCollisionChecker::KinBodyMultiCastVsAll, KinBodyMultiCastVsAllDefaults())
      .def("ExcludeCollisionPair", &PyCollisionChecker::ExcludeCollisionPair)
      .def("IncludeCollisionPair", &PyCollisionChecker::IncludeCollisionPair)
      .def("SaveCollisionMatrix", &PyCollisionChecker::SaveCollisionMatrix)
      .def("RestoreCollisionMatrix", &PyCollisionChecker::RestoreCollisionMatrix)
      .def("SetContactDistance", &PyCollisionChecker::SetContactDistance)
      .def("GetContactDistance", &PyCollisionChecker::GetContactDistance)
      ;
  py::def("GetCollisionChecker", &PyGetCollisionChecker);
  py::class_<PyCollision>("Collision", py::no_init)
     .def("GetDistance", &PyCollision::GetDistance)
     .def("GetNormal", &PyCollision::GetNormal)
     .def("GetPtA", &PyCollision::GetPtA)
     .def("GetPtB", &PyCollision::GetPtB)
     .def("GetLinkAName", &PyCollision::GetLinkAName)
     .def("GetLinkBName", &PyCollision::GetLinkBName)
     .def("GetLinkAParentName", &PyCollision::GetLinkAParentName)
     .def("GetLinkBParentName", &PyCollision::GetLinkBParentName)
     .def("GetCastAlphas", &PyCollision::GetCastAlphas)
     .def("GetCastSupportVertices", &PyCollision::GetCastSupportVertices)
     .def("GetMultiCastAlphas", &PyCollision::GetMultiCastAlphas)
     .def("GetMultiCastIndices", &PyCollision::GetMultiCastIndices)
     .def("GetMultiCastSupportVertices", &PyCollision::GetMultiCastSupportVertices)
    ;
}
