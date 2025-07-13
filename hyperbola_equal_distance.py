import math

# returns n+1 pairs of equal-spaced function values on [a,b]
# between a and b for the function f so that their distances
# are roughly equal, first and last argument being a and b

def dots(a,b,n,f):
    # how much more detail (discretize more finely)
    boostfactor = 10 
    bign = n * boostfactor
    ssum = 0.0
    skip = (0.0+b-a) / bign
    skipsq = skip*skip
    # approximate total length
    for i in range(bign):
        x = a+i*skip
        y1 = f(x)
        y2 = f(x+skip)
        dist = math.sqrt(skipsq + (y2-y1)**2)
        ssum += dist 
    L = [[a,f(a)]]
    wantdist = ssum / n 
    # now the same again
    ssum = 0.0
    x = a
    k = 1
    for i in range(bign-1):
        x = a+i*skip
        y1 = f(x)
        y2 = f(x+skip)
        dist = math.sqrt(skipsq + (y2-y1)**2)
        # print i, x, dist, ssum
        nextsum = ssum + dist 
        if nextsum > k * wantdist: # next segment
            # interpolate linearly 
            xx = x + (k * wantdist - ssum)/dist * skip
            k += 1
            L.append([xx,f(xx)])
        ssum = nextsum
    if k != n: 
        # print( "hyperbola_dist: why not??", k, n)
        pass
    L.append([b,f(b)])
    return L

# scale x in [a,b] to [c,d]
def affinescale(a,b,c,d,x):
    return (0.0+x-a)*(d-c)/(b-a)+c

# stretch all elements in list of lists from [0,1] to [-1,1]
def stretch(L):
    LL = []
    for a in L:
        s = []
        for x in a:
            s.append(affinescale(0,1,-1,1,x))
        LL.append(s)
    return LL

# print 2-element list in 3d for y,z
def yzprint(name, L):
    LL = stretch(L)
    print( "$verts_%s=new Matrix<Rational>([" % name,)
    flag = False
    for s in LL:
        if flag: print( ",",)
        print( "[1,-1,%6.4f,%6.4f]," % (s[0],s[1]),)
        # print "[1,1,%6.4f,%6.4f]" % (s[0],s[1]),
        print( "[1,1,%6.4f,%6.4f]" % (s[0],s[1]))
        flag = True
    print( "]);")
    print( \
      "$poly_%s=new Polytope<Rational>(VERTICES=>$verts_%s);" % \
      (name,name) )
    print( '$poly_%s->VISUAL(VertexLabels=>"",' % name,)
    print( 'VertexThickness=>0.01,EdgeThickness=>0.01,',)
    print( 'EdgeColor=>"red",',)
    print( 'FacetColor=>"red",FacetTransparency=>0.8),')
    return

# print 2-element list in 3d for x,z
def xzprint(name, L):
    LL = stretch(L)
    print( "$verts_%s=new Matrix<Rational>([" % name,)
    flag = False
    for s in LL:
        if flag: print( ",",)
        print( "[1,%6.4f,-1,%6.4f]," % (s[0],s[1]),)
        # print "[1,%6.4f,1,%6.4f]" % (s[0],s[1]),
        print( "[1,%6.4f,1,%6.4f]" % (s[0],s[1]))
        flag = True
    print( "]);")
    print( \
      "$poly_%s=new Polytope<Rational>(VERTICES=>$verts_%s);" % \
      (name,name) )
    print( '$poly_%s->VISUAL(VertexLabels=>"",' % name,)
    print( 'VertexThickness=>0.01,EdgeThickness=>0.01,',)
    print( 'EdgeColor=>"blue",',)
    print( 'FacetColor=>"blue",FacetTransparency=>0.8),')
    return

# print 2-element list in 3d for x,y
def xyprint(name, L):
    LL = stretch(L)
    print( "$verts_%s=new Matrix<Rational>([" % name,)
    flag = False
    for s in LL:
        if flag: print( ",",)
        print( "[1,%6.4f,%6.4f,-1]," % (s[0],s[1]),)
        # print "[1,%6.4f,%6.4f,1]" % (s[0],s[1]),
        print( "[1,%6.4f,%6.4f,1]" % (s[0],s[1]))
        flag = True
    print( "]);")
    print( \
      "$poly_%s=new Polytope<Rational>(VERTICES=>$verts_%s);" % \
      (name,name) )
    print( '$poly_%s->VISUAL(VertexLabels=>"",' % name,)
    print( 'VertexThickness=>0.01,EdgeThickness=>0.01,',)
    print( 'EdgeColor=>"green",',)
    print( 'FacetColor=>"green",FacetTransparency=>0.8),')
    return

# print cube
def cube():
    print( '$c=cube(3);')
    print( '$c->VISUAL(VertexLabels=>"",',)
    print( 'VertexThickness=>0.01,EdgeThickness=>0.01,',)
    print( 'FacetColor=>"white",FacetTransparency=>1),')
    return

#------------------------------------------------------
# L = dots(0.2,1,10, lambda x: 0.2/x)
# print( L)
# exit()

# L = dots(2.0/3,1,10, lambda x: 3-2.0/x)
# L.append([1,0])
# #  print L
# #  print stretch(L)
# xyprint("xy",L)

# print

# #fake
# L = dots(1.0/3,1,10, lambda x: 13./12+1/(16*x-52./3))
# yzprint("fakeyz",L)

# print

# L = dots(1.0/3,1,10, lambda x: x/(4*x-1.0))
# L.append([1,1]) 
# yzprint("yz",L)

# L=[[0,0], [0,1]]
# xzprint("p2up",L)

# L=[[0,0.25], [1,0.25]]
# xzprint("p2across",L)

# cube()
