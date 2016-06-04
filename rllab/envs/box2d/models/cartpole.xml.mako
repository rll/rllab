<%
    from rllab.misc.mako_utils import compute_rect_vertices
    cart_width = 4.0 / (12 ** 0.5)
    cart_height = 3.0 / (12 ** 0.5)

    pole_width = 0.1
    pole_height = 1.0
    noise = opts.get("noise", False)
    if noise:
        import numpy as np
        pole_height += (np.random.rand()-0.5) * pole_height * 1

    cart_friction = 0.0005
    pole_friction = 0.000002
%>

<box2d>
  <world timestep="0.05">
    <body name="cart" type="dynamic" position="0,${cart_height/2}">
      <fixture
              density="1"
              friction="${cart_friction}"
              shape="polygon"
              box="${cart_width/2},${cart_height/2}"
      />
    </body>
    <body name="pole" type="dynamic" position="0,${cart_height}">
      <fixture
              density="1"
              friction="${pole_friction}"
              group="-1"
              shape="polygon"
              vertices="${compute_rect_vertices((0, 0), (0, pole_height), pole_width/2)}"
      />
    </body>
    <body name="track" type="static" position="0,${cart_height/2}">
      <fixture friction="${pole_friction}" group="-1" shape="polygon" box="100,0.1"/>
    </body>
    <joint type="revolute" name="pole_joint" bodyA="cart" bodyB="pole" anchor="0,${cart_height}"/>
    <joint type="prismatic" name="track_cart" bodyA="track" bodyB="cart"/>
    <state type="xpos" body="cart"/>
    <state type="xvel" body="cart"/>
    <state type="apos" body="pole"/>
    <state type="avel" body="pole"/>
    <control type="force" body="cart" anchor="0,0" direction="1,0" ctrllimit="-10,10"/>
  </world>
</box2d>

