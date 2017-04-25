<%
    difficulty = opts.get("difficulty", 1.0)
    texturedir = opts.get("texturedir", "/tmp/mujoco_textures")
    hfield_file = opts.get("hfield_file", "/tmp/mujoco_terrains/hills.png")
%>
<mujoco model="walker2d">
  <compiler inertiafromgeom="true" angle="degree" coordinate="global" texturedir="${texturedir}"/>
  <option timestep="0.005" />
  <default>
    <joint limited='true' damping='.1' armature='0.01' />
    <geom contype='1' conaffinity='0' condim='3' friction='.7 .1 .1' rgba='0.8 0.6 .4 1' density='1000' />
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" width="100" height="100" rgb1="1 1 1" rgb2="0 0 0" />
    <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01" />
    <texture name="texplane" type="2d" builtin="checker" rgb1="0 0 0" rgb2="0.8 0.8 0.8" width="100" height="100" />
    <texture name="hilltexture" file="hills_texture.png" height="40" rgb1="0.62 0.81 0.55" rgb2="0.62 0.81 0.55" type="2d" width="40"/>
    <material name="MatPlane" reflectance="0.0" shininess="1" specular="1" texrepeat="1 1" texture="hilltexture"/>
    <material name='geom' texture="texgeom" texuniform="true" />
    <hfield name="hill" file="${hfield_file}" size="40 40 ${difficulty} 0.1"/>
  </asset>
  <worldbody>
    <light directional="true" cutoff="100" exponent="1" diffuse="1 1 1" specular=".1 .1 .1" pos="0 0 1.3" dir="-0 0 -1.3" />
    <geom name="floor" conaffinity="1" condim="3" material="MatPlane" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="hfield" hfield="hill"/>
    <body name="torso" pos="0 0 1.25">
      <joint name='rootz' type='slide' axis='0 0 1' pos='0 0 0' limited='false' damping='0' armature='0' stiffness='0' ref="1.25" />
      <joint name='rootx' type='slide' axis='1 0 0' pos='0 0 0' limited='false' damping='0' armature='0' stiffness='0' />
      <joint name='rooty' type='hinge' axis='0 1 0' pos='0 0 1.25' limited='false' damping='0' armature='0' stiffness='0' />
      <geom name="torso_geom" type="capsule" fromto="0 0 1.45 0 0 1.05" size="0.05" friction="0.9" />
      <body name="thigh" pos="0 0 1.05">
        <joint name="thigh_joint" type="hinge" pos="0 0 1.05" axis="0 -1 0" range="-150 0" />
        <geom name="thigh_geom" type="capsule" fromto="0 0 1.05 0 0 0.6" size="0.05" friction="0.9" />
        <body name="leg" pos="0 0 0.35">
          <joint name="leg_joint" type="hinge" pos="0 0 0.6" axis="0 -1 0" range="-150 0" />
          <geom name="leg_geom" type="capsule" fromto="0 0 0.6 0 0 0.1" size="0.04" friction="0.9" />
          <body name="foot" pos="0.2/2 0 0.1">
            <joint name="foot_joint" type="hinge" pos="0 0 0.1" axis="0 -1 0" range="-45 45" />
            <geom name="foot_geom" type="capsule" fromto="-0.0 0 0.1 0.2 0 0.1" size="0.06" friction="0.9" />
          </body>
        </body>
      </body>
      <!-- copied and then replace thigh->thigh_left, leg->leg_left, foot->foot_right -->
      <body name="thigh_left" pos="0 0 1.05">
        <joint name="thigh_left_joint" type="hinge" pos="0 0 1.05" axis="0 -1 0" range="-150 0" />
        <geom name="thigh_left_geom" type="capsule" fromto="0 0 1.05 0 0 0.6" size="0.05" friction="0.9" rgba=".7 .3 .6 1" />
        <body name="leg_left" pos="0 0 0.35">
          <joint name="leg_left_joint" type="hinge" pos="0 0 0.6" axis="0 -1 0" range="-150 0" />
          <geom name="leg_left_geom" type="capsule" fromto="0 0 0.6 0 0 0.1" size="0.04" friction="0.9" rgba=".7 .3 .6 1" />
          <body name="foot_left" pos="0.2/2 0 0.1">
            <joint name="foot_left_joint" type="hinge" pos="0 0 0.1" axis="0 -1 0" range="-45 45" />
            <geom name="foot_left_geom" type="capsule" fromto="-0.0 0 0.1 0.2 0 0.1" size="0.06" friction="1.9" rgba=".7 .3 .6 1" />
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- <motor joint="torso_joint" ctrlrange="-100.0 100.0" isctrllimited="true"/> -->
    <motor joint="thigh_joint" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="leg_joint" ctrlrange="-100.0 100.0" ctrllimited="true" />
    <motor joint="foot_joint" ctrlrange="-100.0 100.0" ctrllimited="true" />
    <motor joint="thigh_left_joint" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="leg_left_joint" ctrlrange="-100.0 100.0" ctrllimited="true" />
    <motor joint="foot_left_joint" ctrlrange="-100.0 100.0" ctrllimited="true" />
    <!-- <motor joint="finger2_rot" ctrlrange="-20.0 20.0" isctrllimited="true"/> -->
  </actuator>
</mujoco>
