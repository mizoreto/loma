# helper functions
class Vec3:
    x: float
    y: float
    z: float

class Vec4:
    x: float
    y: float
    z: float
    w: float

def make_vec3(x : In[float], y : In[float], z : In[float]) -> Vec3:
    ret : Vec3
    ret.x = x
    ret.y = y
    ret.z = z
    return ret

def make_vec4(x: In[float], y: In[float], z: In[float], w: In[float]) -> Vec4:
    ret: Vec4
    ret.x = x
    ret.y = y
    ret.z = z
    ret.w = w
    return ret

def add(a : In[Vec3], b : In[Vec3]) -> Vec3:
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z)

def addVec4(a : In[Vec4], b : In[Vec4]) -> Vec4:
    return make_vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)

def sub(a : In[Vec3], b : In[Vec3]) -> Vec3:
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z)

def mul_scalar_vec3(a : In[float], b : In[Vec3]) -> Vec3:
    return make_vec3(a * b.x, a * b.y, a * b.z)

def mul_scalar_vec4(a : In[float], b : In[Vec4]) -> Vec4:
    return make_vec4(a * b.x, a * b.y, a * b.z, a * b.w)

def mul_vec4_vec4(a : In[Vec4], b : In[Vec4]) -> Vec4:
    return make_vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w)

def dot(a : In[Vec3], b : In[Vec3]) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z

def cross(a: In[Vec3], b: In[Vec3]) -> Vec3:
    return make_vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

def length(v: In[Vec3]) -> float:
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

def normalize(v: In[Vec3]) -> Vec3:
    len: float = length(v)
    return make_vec3(v.x / len, v.y / len, v.z / len)

def abs_val(x: In[float]) -> float:
    ret: float
    if x < 0.0:
        ret = -x
    else:
        ret = x
    return ret

def mod(x: In[float], y: In[float]) -> float:
    div: int = float2int(x / y)
    ret: float = x - y * int2float(div)

    return ret

def atan2_approx(y: In[float], x: In[float]) -> float:
    ret: float
    z: float
    atan_approx: float
    if abs_val(x) > abs_val(y):
        z = y / x
        atan_approx = z / (1.0 + 0.28 * z * z)
        if x < 0.0:
            if y < 0.0:
                ret = atan_approx - 3.14159265359
            else:
                ret = atan_approx + 3.14159265359
        else:
            ret = atan_approx
    else:
        z = x / y
        atan_approx = (3.14159265359 / 2.0) - z / (1.0 + 0.28 * z * z)
        if y < 0.0:
            ret = atan_approx - 3.14159265359
        else:
            ret = atan_approx
    
    return ret

def max(a: In[float], b: In[float]) -> float:
    ret: float
    if a > b:
        ret = a
    else:
        ret = b
    return ret

def reflect(I: In[Vec3], N: In[Vec3]) -> Vec3:
    d: float = 2.0 * dot(N, I)
    return sub(I, mul_scalar_vec3(d, N))

def refract(I: In[Vec3], N: In[Vec3], eta: In[float]) -> Vec3:
    cosi: float = dot(N, I)
    k: float = 1.0 - eta * eta * (1.0 - cosi * cosi)
    ret: Vec3 = make_vec3(0.0, 0.0, 0.0)
    if k < 0.0:
        ret = make_vec3(0.0, 0.0, 0.0)
    else:
        ret = sub(mul_scalar_vec3(eta, I), mul_scalar_vec3((eta * cosi + sqrt(k)), N))
    return ret

def clamp(x: In[float], a: In[float], b: In[float]) -> float:
    ret: float
    if x < a:
        ret = a
    else:
        if x > b:
            ret = b
        else:
            ret = x

    return ret

def texture(environment: In[Array[Vec4]], coord: In[Vec3], width: In[int], height: In[int]) -> Vec4:
    u: float = 0.5 * (coord.x + 1.0) * (width - 1)
    v: float = 0.5 * (coord.y + 1.0) * (height - 1)
    x: int = float2int(clamp(u, 0, width - 1))
    y: int = float2int(clamp(v, 0, height - 1))
    return environment[y * width + x]

##################################################
def scene(position: In[Vec3]) -> float:
    height: float = 0.3
    return length(position) - height

def getNormal(pos: In[Vec3], smoothness: In[float]) -> Vec3:
    n: Vec3 = make_vec3(0.0, 0.0, 0.0)
    dx: Vec3 = make_vec3(smoothness, 0.0, 0.0)
    dy: Vec3 = make_vec3(0.0, smoothness, 0.0)
    dz: Vec3 = make_vec3(0.0, 0.0, smoothness)


    n.x = scene(add(pos, dx)) - scene(sub(pos, dx))
    n.y = scene(add(pos, dy)) - scene(sub(pos, dy))
    n.z = scene(add(pos, dz)) - scene(sub(pos, dz))

    return normalize(n)

def raymarch(position: In[Vec3], direction: In[Vec3]) -> float:
    total_distance: float = 0.0
    i: int = 0
    new_pos: Vec3 = make_vec3(0.0, 0.0, 0.0)
    result: float
    while (i < 32, max_iter := 32):
        new_pos = add(position, mul_scalar_vec3(total_distance, direction))
        result = scene(new_pos)
        if result < 0.005:
            i = 32      # loma disallows early return
        else:
            total_distance = total_distance + result
            i = i + 1
    
    ret: float
    if result < 0.005:
        ret = total_distance
    else:
        ret = -1.0
    
    return ret

def calcLookAtMatrix(ro: In[Vec3], ta: In[Vec3], roll: In[float], uu: Out[Vec3], vv: Out[Vec3], ww: Out[Vec3]):
    ww_val: Vec3 = normalize(sub(ta, ro))
    sin_roll: float = sin(roll)
    cos_roll: float = cos(roll)
    up: Vec3 = make_vec3(sin_roll, cos_roll, 0.0)
    uu_val: Vec3 = normalize(cross(ww_val, up))
    vv_val: Vec3 = normalize(cross(uu_val, ww_val))

    uu = uu_val
    vv = vv_val
    ww = ww_val

def mainImage(fragCoord: In[Vec3], iResolutionX: In[float], iResolutionY: In[float], iTime: In[float], environment: In[Array[Vec4]], width: In[int], height: In[int], fragColor: Out[Vec4]):
    uv: Vec3 = make_vec3(0.0, 0.0, 0.0)
    uv.x = fragCoord.x / iResolutionY - 0.5 * iResolutionX / iResolutionY
    uv.y = fragCoord.y / iResolutionY - 0.5
    uv.z = 0.0

    origin: Vec3 = make_vec3(sin(iTime * 0.1) * 2.5, 0.0, cos(iTime * 0.1) * 2.5)

    ta: Vec3 = make_vec3(0.0, 0.0, 0.0)
    uu: Vec3 = make_vec3(0.0, 0.0, 0.0)
    vv: Vec3 = make_vec3(0.0, 0.0, 0.0)
    ww: Vec3 = make_vec3(0.0, 0.0, 0.0)
    calcLookAtMatrix(origin, ta, 0.0, uu, vv, ww)

    uv_x: float = uv.x
    uv_y: float = uv.y

    direction: Vec3 = normalize(add(add(mul_scalar_vec3(uv_x, uu), mul_scalar_vec3(uv_y, vv)), mul_scalar_vec3(2.5, ww)))

    dist: float = raymarch(origin, direction)

    fragPosition: Vec3 = make_vec3(0.0, 0.0, 0.0)
    N: Vec3 = make_vec3(0.0, 0.0, 0.0)
    ballColor: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)
    ref: Vec3 = make_vec3(0.0, 0.0, 0.0)
    P: float
    angle: float
    tmp: float
    starVal: float
    uv_length: float
    edge: float
    starColor: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)
    rim: float
    refr: Vec3 = make_vec3(0.0, 0.0, 0.0)

    baseColor: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)
    reflection: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)
    rim_vec4: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)

    temp1: float
    temp_vec: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)

    if dist < 0.0:
        fragColor = texture(environment, direction, width, height)
    else:
        fragPosition = add(origin, mul_scalar_vec3(dist, direction))
        N = getNormal(fragPosition, 0.01)
        ballColor = make_vec4(0.75, 0.6, 0.0, 0.75)       # (1.0, 0.8, 0.0, 1.0) * 0.75
        ref = reflect(direction, N)

        P = 3.14159265359 / 5.0
        angle = atan2_approx(uv_x, uv_y) + 3.14159265359
        tmp = mod(angle, 2.0 * P)
        starVal = (1.0/P) * (P - abs_val(tmp - P))

        uv_length = length(uv)
        edge = 0.06 - (starVal * 0.03)

        if uv_length < edge:
            starColor = make_vec4(2.8, 1.0, 0.0, 1.0)
        else:
            starColor = make_vec4(0.0, 0.0, 0.0, 0.0)
        
        rim = max(0.0, (0.7 + dot(N, direction)))
        refr = refract(direction, N, 0.7)

        baseColor = texture(environment, refr, width, height)
        baseColor = mul_vec4_vec4(baseColor, ballColor)

        reflection = texture(environment, ref, width, height)
        reflection = mul_scalar_vec4(0.3, reflection)

        rim_vec4 = make_vec4(rim, rim * 0.5, 0.0, 1.0)

        temp1 = max(0.0, 1.0 - uv_length) * 4.0 * (0.2 + abs_val(sin(iTime)) * 0.8)
        temp_vec = make_vec4(0.6, 0.2, 0.0, 1.0)
        temp_vec = mul_scalar_vec4(temp1, temp_vec)

        fragColor = addVec4(baseColor, addVec4(temp_vec, addVec4(starColor, addVec4(reflection, rim_vec4))))
        

def loss_fn(environment: In[Array[Vec4]], width: In[int], height: In[int], target: In[Array[Vec4]]) -> float:
    loss: float = 0.0
    i: int = 0
    j: int = 0
    fragCoord: Vec3 = make_vec3(0.0, 0.0, 0.0)
    fragColor: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)
    target_pixel: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)
    alpha: float
    bg_color: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)
    blended: Vec4 = make_vec4(0.0, 0.0, 0.0, 0.0)

    dx: float
    dy: float
    dz: float

    while (i < height, max_iter := 720):
        j = 0
        while (j < width, max_iter := 1280):
            fragCoord = make_vec3(int2float(j), int2float(i), 0.0)
            mainImage(fragCoord,
                      int2float(width),
                      int2float(height),
                      0.0,  # iTime
                      environment,
                      width,
                      height,
                      fragColor)
            
            blended = fragColor
            # GT
            target_pixel = target[i * width + j]

            dx = blended.x - target_pixel.x
            dy = blended.y - target_pixel.y
            dz = blended.z - target_pixel.z
            loss = loss + dx * dx + dy * dy + dz * dz

            j = j + 1
        i = i + 1

    return loss


rev_loss_fn = rev_diff(loss_fn)

def grad_loss_fn(environment: In[Array[Vec4]], gEnvironment: Out[Array[Vec4]], width: In[int], height: In[int], target: In[Array[Vec4]], gTarget: Out[Array[Vec4]]):
    gWidth: int = 0
    gHeight: int = 0

    rev_loss_fn(environment, gEnvironment, width, gWidth, height, gHeight, target, gTarget, 1.0)