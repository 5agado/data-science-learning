# adapted from https://lsandig.org/blog/2014/08/apollon-python/en/

from cmath import sqrt
import math

class Circle(object):
    """
    A circle represented by center point as complex number and radius.
    """
    def __init__ ( self, mx, my, r ):
        """
        @param mx: x center coordinate
        @param my: y center coordinate
        @param r: radius
        """
        self.r = r
        self.mx = mx
        self.my = my
        self.m = (mx +my*1j)

    def curvature (self):
        return 1/self.r

def outerTangentCircle( circle1, circle2, circle3 ):
    """
    Takes three externally tangent circles and calculates the fourth one enclosing them.
    """
    cur1 = circle1.curvature()
    cur2 = circle2.curvature()
    cur3 = circle3.curvature()
    m1 = circle1.m
    m2 = circle2.m
    m3 = circle3.m
    cur4 = -2 * sqrt( cur1*cur2 + cur2*cur3 + cur1 * cur3 ) + cur1 + cur2 + cur3
    m4 = ( -2 * sqrt( cur1*m1*cur2*m2 + cur2*m2*cur3*m3 + cur1*m1*cur3*m3 ) + cur1*m1 + cur2*m2 + cur3*m3 ) /  cur4
    circle4 = Circle( m4.real, m4.imag, 1/cur4 )

    return circle4


def tangentCirclesFromRadii( r2, r3, r4 ):
    """
    Takes three radii and calculates the corresponding externally
    tangent circles as well as a fourth one enclosing them. The
    enclosing circle is the first one.
    """
    circle2 = Circle( 0, 0, r2 )
    circle3 = Circle( r2 + r3, 0, r3 )
    m4x = (r2*r2 + r2*r4 + r2*r3 - r3*r4) / (r2 + r3)
    m4y = sqrt( (r2 + r4) * (r2 + r4) - m4x*m4x )
    circle4 = Circle( m4x, m4y, r4 )
    circle1 = outerTangentCircle( circle2, circle3, circle4 )
    return ( circle1, circle2, circle3, circle4 )

def secondSolution( fixed, c1, c2, c3 ):
    """
    If given four tangent circles, calculate the other one that is tangent
    to the last three.

    @param fixed: The fixed circle touches the other three, but not
    the one to be calculated.

    @param c1, c2, c3: Three circles to which the other tangent circle
    is to be calculated.
    """
    curf = fixed.curvature()
    cur1 = c1.curvature()
    cur2 = c2.curvature()
    cur3 = c3.curvature()
    curn = 2 * (cur1 + cur2 + cur3) - curf
    mn = (2 * (cur1*c1.m + cur2*c2.m + cur3*c3.m) - curf*fixed.m ) / curn
    return Circle( mn.real, mn.imag, 1/curn )

def recurse(circles, depth, maxDepth, genCircles):
    """Recursively calculate the smaller circles of the AG up to the
    given depth. Note that for depth n we get 2*3^{n+1} circles.

    @param maxDepth: Maximal depth of the recursion.
    @type maxDepth: int

    @param circles: 4-Tuple of circles for which the second
    solutions are calculated
    @type circles: (L{Circle}, L{Circle}, L{Circle}, L{Circle})

    @param depth: Current depth
    @type depth: int
    """
    if( depth == maxDepth):
        return
    (c1, c2, c3, c4) = circles
    if( depth == 0 ):
        # First recursive step, this is the only time we need to
        # calculate 4 new circles.
        #del genCircles[4:]
        cspecial = secondSolution( c1, c2, c3, c4 )
        genCircles.append( cspecial )
        recurse( (cspecial, c2, c3, c4), 1, maxDepth, genCircles )

    cn2 = secondSolution( c2, c1, c3, c4 )
    genCircles.append( cn2 )
    cn3 = secondSolution( c3, c1, c2, c4 )
    genCircles.append( cn3 )
    cn4 = secondSolution( c4, c1, c2, c3 )
    genCircles.append( cn4 )

    recurse( (cn2, c1, c3, c4), depth+1, maxDepth, genCircles )
    recurse( (cn3, c1, c2, c4), depth+1, maxDepth, genCircles )
    recurse( (cn4, c1, c2, c3), depth+1, maxDepth, genCircles )

def get_circle(circle, segments):
    # Define stroke geometry
    points = []
    angle = 2*math.pi/segments  # angle in radians
    r = float(circle.r.real + circle.r.imag)
    mx = float(circle.mx.real + circle.mx.imag)
    my = float(circle.my.real + circle.my.imag)
    for i in range(segments):
        x = r*math.cos(angle*i)
        y = r*math.sin(angle*i)
        z = 0
        points.append((x, y, z))
    return points

start = tangentCirclesFromRadii( 1/c1, 1/c2, 1/c3 )
gen_circles =list(start)
recurse(list(start), 0, max_depth, gen_circles)
res_circles = [get_circle(c, segments) for c in gen_circles]
centers = [(float(circle.mx.real + circle.mx.imag), float(circle.my.real + circle.my.imag), 0) for circle in gen_circles]
radiuses = [float(circle.r.real + circle.r.imag) for circle in gen_circles]