car_height = 1.0
car_width = 0.6
car_mass = 1
car_density = car_mass / car_height / car_width

wheel_height = 0.3
wheel_width = 0.1
wheel_mass = 0.1
wheel_density = wheel_mass / wheel_height / wheel_width
wheel_max_deg = 30

phantom_group = -1
common = { group: phantom_group }

box2d {
  world(timestep: 0.05, gravity: [0, 0]) {
    body(name: :goal, type: :static, position: [0, 0]) {
      fixture(common.merge(shape: :circle, radius: 1))
    }

    car_pos = [3, 4]
    body(name: :car, type: :dynamic, position: car_pos) {
      rect(
           box: [car_width / 2, car_height / 2],
           density: car_density,
           group: phantom_group,
           )
    }
    [:left_front_wheel, :right_front_wheel, :left_rear_wheel, :right_rear_wheel].each do |wheel|
      x_pos = car_width / 2
      x_pos *= wheel =~ /left/ ? -1 : 1
      y_pos = wheel =~ /front/ ? 0.2 : -0.3
      body(name: wheel, type: :dynamic, position: [car_pos[0] + x_pos, car_pos[1] + y_pos]) {
        rect(
             box: [wheel_width / 2, wheel_height / 2],
             density: wheel_density,
             group: phantom_group,
             )
      }
      # limit = wheel =~ /front/ ? [-wheel_max_deg, wheel_max_deg] : [0, 0]
      limit = [0, 0]
      joint(
            type: :revolute,
            name: "#{wheel}_joint",
            bodyA: :car,
            bodyB: wheel,
            localAnchorA: [x_pos, y_pos],
            localAnchorB: [0, 0],
            limit: limit,
            )
    end
    control(
            type: :force,
            bodies: [:left_front_wheel, :right_front_wheel],
            anchor: [0, 0],
            direction: [0, 1],
            ctrllimit: [-10.N, 10.N],
            )
    state body: :car, type: :xvel
    state body: :car, type: :yvel
    state body: :car, type: :dist, to: :goal
    state body: :car, type: :angle, to: :goal, transform: :cos
    state body: :car, type: :angle, to: :goal, transform: :sin
  }
}
