
def compute_rect_vertices(fromp, to, radius):
    x1, y1 = fromp
    x2, y2 = to
    if abs(y1 - y2) < 1e-6:
        dx = 0
        dy = radius
    else:
        dx = radius * 1.0 / (((x1 - x2) / (y1 - y2)) ** 2 + 1) ** 0.5
        # equivalently dx = radius * (y2-y1).to_f / ((x2-x1)**2 + (y2-y1)**2)**0.5
        dy = (radius**2 - dx**2) ** 0.5
        dy *= -1 if (x1 - x2) * (y1 - y2) > 0 else 1

    return ";".join([",".join(map(str, r)) for r in [
      [x1 + dx, y1 + dy],
      [x2 + dx, y2 + dy],
      [x2 - dx, y2 - dy],
      [x1 - dx, y1 - dy],
    ]])

