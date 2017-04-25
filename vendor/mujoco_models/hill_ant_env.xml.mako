<%
    difficulty = opts.get("difficulty", 1.0)
    texturedir = opts.get("texturedir", "/tmp/mujoco_textures")
    hfield_file = opts.get("hfield_file", "/tmp/mujoco_terrains/hills.png")
%>
<mujoco model="ant">
  <compiler inertiafromgeom="true" angle="degree" coordinate="local" texturedir="${texturedir}"/>
  <option timestep="0.02" integrator="RK4" />
  <custom>
    <numeric name="init_qpos" data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" />
  </custom>
  <default>
    <joint limited="true" armature="1" damping="1" />
    <geom condim="3" conaffinity="0" margin="0.01" friction="1 0.5 0.5" solref=".02 1" solimp=".8 .8 .01" rgba="0.8 0.6 0.4 1" density="5.0" />
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
    <body name="torso" pos="0 0 0.75">
      <geom name="torso_geom" type="sphere" size="0.25" pos="0 0 0" />
      <joint name="root" type="free" limited="false" pos="0 0 0" axis="0 0 1" margin="0.01" armature="0" damping="0" />
      <body name="front_left_leg" pos="0 0 0">
        <geom name="aux_1_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 0.2 0.2 0.0" />
        <body name="aux_1" pos="0.2 0.2 0">
          <joint name="hip_1" type="hinge" pos="0.0 0.0 0.0" axis="0 0 1" range="-30 30" />
          <geom name="left_leg_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 0.2 0.2 0.0" />
          <body pos="0.2 0.2 0">
            <joint name="ankle_1" type="hinge" pos="0.0 0.0 0.0" axis="-1 1 0" range="30 70" />
            <geom name="left_ankle_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 0.4 0.4 0.0" />
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom name="aux_2_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 -0.2 0.2 0.0" />
        <body name="aux_2" pos="-0.2 0.2 0">
          <joint name="hip_2" type="hinge" pos="0.0 0.0 0.0" axis="0 0 1" range="-30 30" />
          <geom name="right_leg_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 -0.2 0.2 0.0" />
          <body pos="-0.2 0.2 0">
            <joint name="ankle_2" type="hinge" pos="0.0 0.0 0.0" axis="1 1 0" range="-70 -30" />
            <geom name="right_ankle_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 -0.4 0.4 0.0" />
          </body>
        </body>
      </body>
      <body name="back_leg" pos="0 0 0">
        <geom name="aux_3_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" />
        <body name="aux_3" pos="-0.2 -0.2 0">
          <joint name="hip_3" type="hinge" pos="0.0 0.0 0.0" axis="0 0 1" range="-30 30" />
          <geom name="back_leg_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" />
          <body pos="-0.2 -0.2 0">
            <joint name="ankle_3" type="hinge" pos="0.0 0.0 0.0" axis="-1 1 0" range="-70 -30" />
            <geom name="third_ankle_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" />
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom name="aux_4_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 0.2 -0.2 0.0" />
        <body name="aux_4" pos="0.2 -0.2 0">
          <joint name="hip_4" type="hinge" pos="0.0 0.0 0.0" axis="0 0 1" range="-30 30" />
          <geom name="rightback_leg_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 0.2 -0.2 0.0" />
          <body pos="0.2 -0.2 0">
            <joint name="ankle_4" type="hinge" pos="0.0 0.0 0.0" axis="1 1 0" range="30 70" />
            <geom name="fourth_ankle_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 0.4 -0.4 0.0" />
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hip_4" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="ankle_4" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="hip_1" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="ankle_1" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="hip_2" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="ankle_2" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="hip_3" ctrlrange="-150.0 150.0" ctrllimited="true" />
    <motor joint="ankle_3" ctrlrange="-150.0 150.0" ctrllimited="true" />
  </actuator>
</mujoco>
