<%
    difficulty = opts.get("difficulty", 1.0)
    texturedir = opts.get("texturedir", "/tmp/mujoco_textures")
    hfield_file = opts.get("hfield_file", "/tmp/mujoco_terrains/hills.png")
%>
<mujoco model="swimmer">
  <compiler inertiafromgeom="true" angle="degree" coordinate="local" texturedir="${texturedir}"/>
  <custom>
    <numeric name="frame_skip" data="50" />
  </custom>
  <option timestep="0.001" density="4000" viscosity="0.1" integrator="Euler" iterations="1000">
    <flag warmstart="disable" />
  </option>
  <default>
    <geom contype='1' conaffinity='1' condim='1' rgba='0.8 0.6 .4 1' material="geom" />
    <!--<joint armature='1'  />-->
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" width="100" height="100" rgb1="1 1 1" rgb2="0 0 0" />
    <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01" />
    <texture name="texplane" type="2d" builtin="checker" rgb1="0 0 0" rgb2="0.8 0.8 0.8" width="100" height="100" />
    <texture name="hilltexture" file="hills_texture.png" height="40" rgb1="0.62 0.81 0.55" rgb2="0.62 0.81 0.55" type="2d" width="40"/>
    <material name="MatPlane" reflectance="0.0" shininess="1" specular="1" texrepeat="1 1" texture="hilltexture"/>
    <material name='geom' texture="texgeom" texuniform="true" />
    <hfield name="hill" file="${hfield_file}" size="40 40 ${difficulty/2.0} 0.1"/>
  </asset>
  <worldbody>
    <light directional="true" cutoff="100" exponent="1" diffuse="1 1 1" specular=".1 .1 .1" pos="0 0 1.3" dir="-0 0 -1.3" />
    <geom name="floor" conaffinity="1" condim="1" material="MatPlane" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="hfield" hfield="hill"/>
    <!--  ================= SWIMMER ================= /-->
    <body name="torso" pos="0 0 0">
      <geom name="torso" type="capsule" fromto="1.5 0 0 0.5 0 0" size="0.1" density="1000" />
      <joint name="root" armature="0" damping="0" limited="false"  pos="0 0 0" axis="0 0 1" stiffness="0" type="free"/>
      <body name="mid" pos="0.5 0 0">
        <geom name="mid" type="capsule" fromto="0 0 0 -1 0 0" size="0.1" density="1000" />
        <joint name="rot2" type="hinge" pos="0 0 0" axis="0 0 1" range="-100 100" limited="true" />
        <body name="back" pos="-1 0 0">
          <geom name="back" type="capsule" fromto="0 0 0 -1 0 0" size="0.1" density="1000" />
          <joint name="rot3" type="hinge" pos="0 0 0" axis="0 0 1" range="-100 100" limited="true" />
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="rot2" ctrllimited="true" ctrlrange="-50 50" />
    <motor joint="rot3" ctrllimited="true" ctrlrange="-50 50" />
  </actuator>
</mujoco>
