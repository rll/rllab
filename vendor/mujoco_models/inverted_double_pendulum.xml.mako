<!-- Cartpole Model

    The state space is populated with joints in the order that they are
    defined in this file. The actuators also operate on joints.

    State-Space (name/joint/parameter):
        - cart      slider      position (m)
        - pole      hinge       angle (rad)
        - cart      slider      velocity (m/s)
        - pole      hinge       angular velocity (rad/s)

    Actuators (name/actuator/parameter):
        - cart      motor       force x (N)

-->
<%
    noise = opts.get("noise", False)
    pole1_height = 0.6
    pole2_height = 0.6
    if noise:
        import numpy as np
        pole1_height = pole1_height + np.random.uniform(-0.1, 0.4)
        pole2_height = pole2_height + np.random.uniform(-0.1, 0.4)
%>


<mujoco model='cartpole'>
    <compiler inertiafromgeom='true'
              coordinate='local'/>

    <custom>
        <numeric name="frame_skip" data="2" />
    </custom>

    <default>
        <joint damping='0.05' />
        <geom contype='0'
              friction='1 0.1 0.1'
              rgba='0.7 0.7 0 1' />
    </default>

    <option timestep='0.01'
            gravity='1e-5 0 -9.81'
            integrator="RK4"
    />

    <size nstack='3000'/>

    <worldbody>
        <geom name='floor'
              pos='0 0 -3.0'
              size='40 40 40'
              type='plane'
              rgba='0.8 0.9 0.8 1' />
        <geom name='rail'
              type='capsule'
              pos='0 0 0'
              quat='0.707 0 0.707 0'
              size='0.02 1'
              rgba='0.3 0.3 0.7 1' />
        <body name='cart' pos='0 0 0'>
            <joint name='slider'
                   type='slide'
                   limited='true'
                   pos='0 0 0'
                   axis='1 0 0'
                   range='-10 10'
                   margin='0.01'/>
            <geom name='cart'
                  type='capsule'
                  pos='0 0 0'
                  quat='0.707 0 0.707 0'
                  size='0.1 0.1' />
            <body name='pole' pos='0 0 0'>
                <joint name='hinge'
                       type='hinge'
                       pos='0 0 0'
                       axis='0 1 0'/>
                <geom name='cpole'
                      type='capsule'
                      fromto='0 0 0 0 0 ${pole1_height}'
                      size='0.045 ${pole1_height/2}'
                      rgba='0 0.7 0.7 1' />
                <body name='pole2' pos='0 0 ${pole1_height}'>
                    <joint name='hinge2'
                           type='hinge'
                           pos='0 0 0'
                           axis='0 1 0'/>
                    <geom name='cpole2'
                          type='capsule'
                          fromto='0 0 0 0 0 ${pole2_height}'
                          size='0.045 ${pole2_height/2}'
                          rgba='0 0.7 0.7 1' />
                    <site name='tip'
                          size='0.01 0.01'
                          pos='0 0 ${pole2_height}'/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name='slide'
               joint='slider'
               gear='500'
               ctrlrange='-1 1'
               ctrllimited='true'/>
    </actuator>
</mujoco>

