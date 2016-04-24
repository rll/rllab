<%
    from rllab.misc.mako_utils import compute_rect_vertices
    link_len = opts['link_len']
    link_width = 0.1
%>

<box2d>
  <world timestep="0.01" velitr="20" positr="20">
    <body name="link1" type="dynamic" position="0,0">
      <fixture
              density="5.0"
              group="-1"
              shape="polygon"
              vertices="${compute_rect_vertices([0,0], [0, -link_len], link_width/2)}"
      />
    </body>
    <body name="link2" type="dynamic" position="0,${-link_len}">
      <fixture
              density="5.0"
              group="-1"
              shape="polygon"
              vertices="${compute_rect_vertices([0,0], [0,-link_len], link_width/2)}"
      />
    </body>
    <body name="track" type="static" position="0,-0.1">
      <fixture group="-1" shape="polygon" box="100,0.1"/>
    </body>
    <joint type="revolute" name="link_joint_1" bodyA="track" bodyB="link1" anchor="0,0"/>
    <joint type="revolute" name="link_joint_2" bodyA="link1" bodyB="link2" anchor="0,${-link_len}"/>
    <!-- <control type="torque" joint="link_joint_1" ctrllimit="-3,3"/> -->
    <control type="torque" joint="link_joint_2" ctrllimit="-50,50" />
    <state type="apos" body="link1" transform="sin"/>
    <state type="apos" body="link1" transform="cos"/>
    <state type="avel" body="link1"/>
    <state type="apos" body="link2" transform="sin"/>
    <state type="apos" body="link2" transform="cos"/>
    <state type="avel" body="link2"/>
  </world>
</box2d>

