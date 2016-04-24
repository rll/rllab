from Box2D import b2ContactListener, b2Vec2, b2DrawExtended
import pygame
from pygame import (QUIT, KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEMOTION)


class PygameDraw(b2DrawExtended):
    """
    This debug draw class accepts callbacks from Box2D (which specifies what to
    draw) and handles all of the rendering.

    If you are writing your own game, you likely will not want to use debug
    drawing.  Debug drawing, as its name implies, is for debugging.
    """
    surface = None
    axisScale = 50.0

    def __init__(self, test=None, **kwargs):
        b2DrawExtended.__init__(self, **kwargs)
        self.flipX = False
        self.flipY = True
        self.convertVertices = True
        self.test = test
        self.flags = dict(
            drawShapes=True,
            convertVertices=True,
        )

    def StartDraw(self):
        self.zoom = self.test.viewZoom
        self.center = self.test.viewCenter
        self.offset = self.test.viewOffset
        self.screenSize = self.test.screenSize

    def EndDraw(self):
        pass

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a pixel size and color.
        """
        self.DrawCircle(p, size / self.zoom, color, drawwidth=0)

    def DrawAABB(self, aabb, color):
        """
        Draw a wireframe around the AABB with the given color.
        """
        points = [(aabb.lowerBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.upperBound.y),
                  (aabb.lowerBound.x, aabb.upperBound.y)]

        pygame.draw.aalines(self.surface, color, True, points)

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        pygame.draw.aaline(self.surface, color.bytes, p1, p2)

    def DrawTransform(self, xf):
        """
        Draw the transform xf on the screen
        """
        p1 = xf.position
        p2 = self.to_screen(p1 + self.axisScale * xf.R.x_axis)
        p3 = self.to_screen(p1 + self.axisScale * xf.R.y_axis)
        p1 = self.to_screen(p1)
        pygame.draw.aaline(self.surface, (255, 0, 0), p1, p2)
        pygame.draw.aaline(self.surface, (0, 255, 0), p1, p3)

    def DrawCircle(self, center, radius, color, drawwidth=1):
        """
        Draw a wireframe circle given the center, radius, axis of orientation
        and color.
        """
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        pygame.draw.circle(self.surface, color.bytes,
                           center, radius, drawwidth)

    def DrawSolidCircle(self, center, radius, axis, color):
        """
        Draw a solid circle given the center, radius, axis of orientation and
        color.
        """
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        pygame.draw.circle(
            self.surface,
            (color / 2).bytes + [127], center, radius, 0)
        pygame.draw.circle(
            self.surface,
            color.bytes, center, radius, 1)

        pygame.draw.aaline(self.surface, (255, 0, 0), center,
                           (center[0] - radius * axis[0],
                            center[1] + radius * axis[1]))

    def DrawSolidCapsule(self, p1, p2, radius, color):
        pass

    def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the screen vertices with the specified
        color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes,
                               vertices[0], vertices)
        else:
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the screen vertices with the specified
        color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes,
                               vertices[0], vertices[1])
        else:
            pygame.draw.polygon(
                self.surface, (color / 2).bytes + [127], vertices, 0)
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)


class Box2DViewer(b2ContactListener):

    def __init__(self, world):
        super(Box2DViewer, self).__init__()

        self.world = world
        self.world.contactListener = self

        self._reset()
        pygame.init()
        caption = "Box2D Simulator"
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((800, 600))
        self.screenSize = b2Vec2(*self.screen.get_size())
        self.renderer = PygameDraw(surface=self.screen, test=self)
        self.world.renderer = self.renderer

        # FIXME, commented to avoid Linux error due to font.
#         try:
#             self.font = pygame.font.Font(None, 15)
#         except IOError:
#             try:
#                 self.font = pygame.font.Font("freesansbold.ttf", 15)
#             except IOError:
#                 print("Unable to load default font or 'freesansbold.ttf'")
#                 print("Disabling text drawing.")
#                 self.Print = lambda *args: 0
#                 self.DrawStringAt = lambda *args: 0

        self.viewCenter = (0, 20.0)
        self._viewZoom = 100

    def _reset(self):
        self._viewZoom = 10.0
        self._viewCenter = None
        self._viewOffset = None
        self.screenSize = None
        self.rMouseDown = False
        self.textLine = 30
        self.font = None

    def setCenter(self, value):
        """
        Updates the view offset based on the center of the screen.

        Tells the debug draw to update its values also.
        """
        self._viewCenter = b2Vec2(*value)
        self._viewCenter *= self._viewZoom
        self._viewOffset = self._viewCenter - self.screenSize / 2

    def setZoom(self, zoom):
        self._viewZoom = zoom

    viewZoom = property(lambda self: self._viewZoom, setZoom,
                        doc='Zoom factor for the display')
    viewCenter = property(lambda self: self._viewCenter / self._viewZoom,
                          setCenter, doc='Screen center in camera coordinates')
    viewOffset = property(lambda self: self._viewOffset,
                          doc='The offset of the top-left corner of the '
                              'screen')

    def checkEvents(self):
        """
        Check for pygame events (mainly keyboard/mouse events).
        Passes the events onto the GUI also.
        """
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key ==
                                      pygame.K_ESCAPE):
                return False
            elif event.type == KEYDOWN:
                self._Keyboard_Event(event.key, down=True)
            elif event.type == KEYUP:
                self._Keyboard_Event(event.key, down=False)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.viewZoom *= 1.1
                elif event.button == 5:
                    self.viewZoom /= 1.1
            elif event.type == MOUSEMOTION:
                if self.rMouseDown:
                    self.viewCenter -= (event.rel[0] /
                                        5.0, -event.rel[1] / 5.0)

        return True

    def _Keyboard_Event(self, key, down=True):
        """
        Internal keyboard event, don't override this.

        Checks for the initial keydown of the basic testbed keys. Passes the
        unused ones onto the test via the Keyboard() function.
        """
        if down:
            if key == pygame.K_z:       # Zoom in
                self.viewZoom = min(2 * self.viewZoom, 500.0)
            elif key == pygame.K_x:     # Zoom out
                self.viewZoom = max(0.9 * self.viewZoom, 0.02)

    def CheckKeys(self):
        """
        Check the keys that are evaluated on every main loop iteration.
        I.e., they aren't just evaluated when first pressed down
        """

        pygame.event.pump()
        self.keys = keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.viewCenter -= (0.5, 0)
        elif keys[pygame.K_RIGHT]:
            self.viewCenter += (0.5, 0)

        if keys[pygame.K_UP]:
            self.viewCenter += (0, 0.5)
        elif keys[pygame.K_DOWN]:
            self.viewCenter -= (0, 0.5)

        if keys[pygame.K_HOME]:
            self.viewZoom = 1.0
            self.viewCenter = (0.0, 20.0)

    def ConvertScreenToWorld(self, x, y):
        return b2Vec2((x + self.viewOffset.x) / self.viewZoom,
                      ((self.screenSize.y - y + self.viewOffset.y) /
                          self.viewZoom))

    def loop_once(self):
        self.checkEvents()
        # self.CheckKeys()
        self.screen.fill((0, 0, 0))

        if self.renderer is not None:
            self.renderer.StartDraw()

        self.world.DrawDebugData()
        self.renderer.EndDraw()

        pygame.display.flip()

    def finish(self):
        pygame.quit()
