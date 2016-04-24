<%
    noise = opts.get("noise", False)
    track_width = 4
    if noise:
        import numpy as np
        track_width += np.random.uniform(-1, 1)
%>

<box2d>
  <world timestep="0.05">
    <body name="cart" type="dynamic" position="0,0.05">
      <fixture density="5" friction="0.0005" shape="polygon" box="0.2, 0.1"/>
    </body>
    <body name="track" type="static" position="0,1">
      <fixture shape="sine_chain" height="1" width="${track_width}"/>
    </body>
    <state type="xpos" body="cart"/>
    <state type="xvel" body="cart"/>
    <control type="force" body="cart" anchor="0,0" direction="1,0" ctrllimit="-1,1"/>
  </world>
</box2d>
