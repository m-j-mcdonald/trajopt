#include <vector>

PyCollisionChecker PyGetCollisionChecker(py::object py_env) {
  CollisionCheckerPtr cc = CollisionChecker::GetOrCreate(*GetCppEnv(py_env));
  return PyCollisionChecker(cc);
}

class PyCollisionProcessor {
public:

  py::object ValAndGrad() {

  }

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

/*
def _calc_grad_and_val(self, robot_body, obj_body, collisions):
        """
            This function is helper function of robot_obj_collision(self, x)
            It calculates collision distance and gradient between each robot's link and object

            robot_body: OpenRAVEBody containing body information of pr2 robot
            obj_body: OpenRAVEBody containing body information of object
            collisions: list of collision objects returned by collision checker
            Note: Needs to provide attr_dim indicating robot pose's total attribute dim
        """
        # Initialization
        links = []
        robot = self.params[self.ind0]
        obj = self.params[self.ind1]
        col_links = robot.geom.col_links
        obj_links = obj.geom.col_links
        obj_pos = OpenRAVEBody.obj_pose_from_transform(obj_body.env_body.GetTransform())
        Rz, Ry, Rx = OpenRAVEBody._axis_rot_matrices(obj_pos[:3], obj_pos[3:])
        rot_axises = [[0,0,1], np.dot(Rz, [0,1,0]),  np.dot(Rz, np.dot(Ry, [1,0,0]))]
        link_pair_to_col = {}
        for c in collisions:
            # Identify the collision points
            linkA, linkB = c.GetLinkAName(), c.GetLinkBName()
            linkAParent, linkBParent = c.GetLinkAParentName(), c.GetLinkBParentName()
            linkRobot, linkObj = None, None
            sign = 0
            if linkAParent == robot_body.name and linkBParent == obj_body.name:
                ptRobot, ptObj = c.GetPtA(), c.GetPtB()
                linkRobot, linkObj = linkA, linkB
                sign = -1
            elif linkBParent == robot_body.name and linkAParent == obj_body.name:
                ptRobot, ptObj = c.GetPtB(), c.GetPtA()
                linkRobot, linkObj = linkB, linkA
                sign = 1
            else:
                continue

            if linkRobot not in col_links or linkObj not in obj_links:
                continue
            # Obtain distance between two collision points, and their normal collision vector
            distance = c.GetDistance()
            normal = c.GetNormal()
            # Calculate robot jacobian
            robot = robot_body.env_body
            robot_link_ind = robot.GetLink(linkRobot).GetIndex()
            robot_jac = robot.CalculateActiveJacobian(robot_link_ind, ptRobot)
            grad = np.zeros((1, self.attr_dim+6))
            grad[:, :self.attr_dim] = np.dot(sign * normal, robot_jac)
            # robot_grad = np.dot(sign * normal, robot_jac).reshape((1,20))
            col_vec = -sign*normal
            # Calculate object pose jacobian
            # obj_jac = np.array([-sign*normal])
            grad[:, self.attr_dim:self.attr_dim+3] = np.array([-sign*normal])
            torque = ptObj - obj_pos[:3]
            # Calculate object rotation jacobian
            rot_vec = np.array([[np.dot(np.cross(axis, torque), col_vec) for axis in rot_axises]])
            # obj_jac = np.c_[obj_jac, rot_vec]
            grad[:, self.attr_dim+3:self.attr_dim+6] = rot_vec
            # Constructing gradient matrix
            # robot_grad = np.c_[robot_grad, obj_jac]
            # TODO: remove robot.GetLink(linkRobot) from links (added for debugging purposes)
            link_pair_to_col[(linkRobot, linkObj)] = [self.dsafe - distance, grad, robot.GetLink(linkRobot), robot.GetLink(linkObj)]
            # import ipdb; ipdb.set_trace()
            if self._debug:
                self.plot_collision(ptRobot, ptObj, distance)

        vals, greds = [], []
        for robot_link, obj_link in self.col_link_pairs:
            col_infos = link_pair_to_col.get((robot_link, obj_link), [self.dsafe - const.MAX_CONTACT_DISTANCE, np.zeros((1, self.attr_dim+6)), None, None])
            vals.append(col_infos[0])
            greds.append(col_infos[1])


        # # arrange gradients in proper link order
        # max_dist = self.dsafe - const.MAX_CONTACT_DISTANCE
        # vals, robot_grads = max_dist*np.ones((len(self.col_links),1)), np.zeros((len(self.col_links), self.attr_dim+6))
        #
        #
        # links = sorted(links, key = lambda x: x[0])
        #
        # links_pair = [(link[3].GetName(), link[4].GetName()) for link in links]
        #
        # vals[:len(links),0] = np.array([link[1] for link in links])
        # robot_grads[:len(links), range(self.attr_dim+6)] = np.array([link[2] for link in links]).reshape((len(links), self.attr_dim+6))
        # TODO: remove line below which was added for debugging purposes
        # self.links = [(ind, val, limb) for ind, val, grad, limb in links]
        # self.col = collisions
        # return vals, robot_grads
        return np.array(vals).reshape((len(vals), 1)), np.array(greds).reshape((len(greds), self.attr_dim+6))
*/
