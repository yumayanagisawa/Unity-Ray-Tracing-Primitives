// https://www.shadertoy.com/view/tl23Rm based on this shader by reinder
Shader "Unlit/Ray-Tracing"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
		iMouseX ("Mouse X", Range(0, 1920)) = 0
		iMouseY("Mouse Y", Range(0, 1080)) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
		//GrabPass{"iChannel0"}
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
			//#include "Unlit/Common.shader"
			#define MAX_DIST 1e10
			float dot2(in float3 v) { return dot(v, v); }

			// Plane 
			float iPlane(in float3 ro, in float3 rd, in float2 distBound, inout float3 normal,
				in float3 planeNormal, in float planeDist) {
				float a = dot(rd, planeNormal);
				float d = -(dot(ro, planeNormal) + planeDist) / a;
				if (a > 0. || d < distBound.x || d > distBound.y) {
					return MAX_DIST;
				}
				else {
					normal = planeNormal;
					return d;
				}
			}

			// Sphere:          https://www.shadertoy.com/view/4d2XWV
			float iSphere(in float3 ro, in float3 rd, in float2 distBound, inout float3 normal,
				float sphereRadius) {
				float b = dot(ro, rd);
				float c = dot(ro, ro) - sphereRadius * sphereRadius;
				float h = b * b - c;
				if (h < 0.) {
					return MAX_DIST;
				}
				else {
					h = sqrt(h);
					float d1 = -b - h;
					float d2 = -b + h;
					if (d1 >= distBound.x && d1 <= distBound.y) {
						normal = normalize(ro + rd * d1);
						return d1;
					}
					else if (d2 >= distBound.x && d2 <= distBound.y) {
						normal = normalize(ro + rd * d2);
						return d2;
					}
					else {
						return MAX_DIST;
					}
				}
			}

			// Box:             https://www.shadertoy.com/view/ld23DV
			float iBox(in float3 ro, in float3 rd, in float2 distBound, inout float3 normal,
				in float3 boxSize) {
				float3 m = sign(rd) / max(abs(rd), 1e-8);
				float3 n = m * ro;
				float3 k = abs(m)*boxSize;

				float3 t1 = -n - k;
				float3 t2 = -n + k;

				float tN = max(max(t1.x, t1.y), t1.z);
				float tF = min(min(t2.x, t2.y), t2.z);

				if (tN > tF || tF <= 0.) {
					return MAX_DIST;
				}
				else {
					if (tN >= distBound.x && tN <= distBound.y) {
						normal = -sign(rd)*step(t1.yzx, t1.xyz)*step(t1.zxy, t1.xyz);
						return tN;
					}
					else if (tF >= distBound.x && tF <= distBound.y) {
						normal = -sign(rd)*step(t1.yzx, t1.xyz)*step(t1.zxy, t1.xyz);
						return tF;
					}
					else {
						return MAX_DIST;
					}
				}
			}

			// Triangle:        https://www.shadertoy.com/view/MlGcDz
			float iTriangle(in float3 ro, in float3 rd, in float2 distBound, inout float3 normal,
							 in float3 v0, in float3 v1, in float3 v2) {
				float3 v1v0 = v1 - v0;
				float3 v2v0 = v2 - v0;
				float3 rov0 = ro - v0;

				float3  n = cross(v1v0, v2v0);
				float3  q = cross(rov0, rd);
				float d = 1.0 / dot(rd, n);
				float u = d * dot(-q, v2v0);
				float v = d * dot(q, v1v0);
				float t = d * dot(-n, rov0);

				if (u < 0. || v<0. || (u + v)>1. || t<distBound.x || t>distBound.y) {
					return MAX_DIST;
				}
			 else {
			  normal = normalize(-n);
				return t;
				}
			}

			// Capsule:         https://www.shadertoy.com/view/Xt3SzX
			float iCapsule(in float3 ro, in float3 rd, in float2 distBound, inout float3 normal,
				in float3 pa, in float3 pb, in float r) {
				float3  ba = pb - pa;
				float3  oa = ro - pa;

				float baba = dot(ba, ba);
				float bard = dot(ba, rd);
				float baoa = dot(ba, oa);
				float rdoa = dot(rd, oa);
				float oaoa = dot(oa, oa);

				float a = baba - bard * bard;
				float b = baba * rdoa - baoa * bard;
				float c = baba * oaoa - baoa * baoa - r * r*baba;
				float h = b * b - a * c;
				if (h >= 0.) {
					float t = (-b - sqrt(h)) / a;
					float d = MAX_DIST;

					float y = baoa + t * bard;

					// body
					if (y > 0. && y < baba) {
						d = t;
					}
					else {
						// caps
						float3 oc = (y <= 0.) ? oa : ro - pb;
						b = dot(rd, oc);
						c = dot(oc, oc) - r * r;
						h = b * b - c;
						if (h > 0.0) {
							d = -b - sqrt(h);
						}
					}
					if (d >= distBound.x && d <= distBound.y) {
						//float3  
						pa = ro + rd * d - pa;
						float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
						normal = (pa - h * ba) / r;
						return d;
					}
				}
				return MAX_DIST;
			}

			// Goursat:         https://www.shadertoy.com/view/3lj3DW
			float cuberoot(float x) { return sign(x)*pow(abs(x), 1.0 / 3.0); }

			float iGoursat(in float3 ro, in float3 rd, in float2 distBound, inout float3 normal,
				in float ra, float rb) {
				// hole: x4 + y4 + z4 - (r2^2)·(x2 + y2 + z2) + r1^4 = 0;
				float ra2 = ra * ra;
				float rb2 = rb * rb;

				float3 rd2 = rd * rd; float3 rd3 = rd2 * rd;
				float3 ro2 = ro * ro; float3 ro3 = ro2 * ro;

				float ka = 1.0 / dot(rd2, rd2);

				float k3 = ka * (dot(ro, rd3));
				float k2 = ka * (dot(ro2, rd2) - rb2 / 6.0);
				float k1 = ka * (dot(ro3, rd) - rb2 * dot(rd, ro) / 2.0);
				float k0 = ka * (dot(ro2, ro2) + ra2 * ra2 - rb2 * dot(ro, ro));

				float c2 = k2 - k3 * (k3);
				float c1 = k1 + k3 * (2.0*k3*k3 - 3.0*k2);
				float c0 = k0 + k3 * (k3*(c2 + k2)*3.0 - 4.0*k1);

				c0 /= 3.0;

				float Q = c2 * c2 + c0;
				float R = c2 * c2*c2 - 3.0*c0*c2 + c1 * c1;
				float h = R * R - Q * Q*Q;


				// 2 intersections
				if (h > 0.0) {
					h = sqrt(h);

					float s = cuberoot(R + h);
					float u = cuberoot(R - h);

					float x = s + u + 4.0*c2;
					float y = s - u;

					float k2 = x * x + y * y*3.0;

					float k = sqrt(k2);

					float d = -0.5*abs(y)*sqrt(6.0 / (k + x))
						- 2.0*c1*(k + x) / (k2 + x * k)
						- k3;

					if (d >= distBound.x && d <= distBound.y) {
						float3 pos = ro + rd * d;
						normal = normalize(4.0*pos*pos*pos - 2.0*pos*rb*rb);
						return d;
					}
					else {
						return MAX_DIST;
					}
				}
				else {
					// 4 intersections
					float sQ = sqrt(Q);
					float z = c2 - 2.0*sQ*cos(acos(-R / (sQ*Q)) / 3.0);

					float d1 = z - 3.0*c2;
					float d2 = z * z - 3.0*c0;

					if (abs(d1) < 1.0e-4) {
						if (d2 < 0.0) return MAX_DIST;
						d2 = sqrt(d2);
					}
					else {
						if (d1 < 0.0) return MAX_DIST;
						d1 = sqrt(d1 / 2.0);
						d2 = c1 / d1;
					}

					//----------------------------------

					float h1 = sqrt(d1*d1 - z + d2);
					float h2 = sqrt(d1*d1 - z - d2);
					float t1 = -d1 - h1 - k3;
					float t2 = -d1 + h1 - k3;
					float t3 = d1 - h2 - k3;
					float t4 = d1 + h2 - k3;

					if (t2 < 0.0 && t4 < 0.0) return MAX_DIST;

					float result = 1e20;
					if (t1 > 0.0) result = t1;
					else if (t2 > 0.0) result = t2;
					if (t3 > 0.0) result = min(result, t3);
					else if (t4 > 0.0) result = min(result, t4);

					if (result >= distBound.x && result <= distBound.y) {
						float3 pos = ro + rd * result;
						normal = normalize(4.0*pos*pos*pos - 2.0*pos*rb*rb);
						return result;
					}
					else {
						return MAX_DIST;
					}
				}
			}

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
				float4 screenPos : TEXCOORD1;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
				float4 screenPos : TEXCOORD1;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
			float iMouseX, iMouseY;
			float2 iMouse;// = float2(iMouseX, iMouseY);
			//sampler2D iChannel0;
			Texture2D iChannel0;

            v2f vert (appdata v)
            {
                /*v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;*/
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				o.screenPos = ComputeScreenPos(o.vertex);
				return o;
            }

			#define PATH_LENGTH 12

			uint baseHash(uint2 p) {
				p = 1103515245U * ((p >> 1U) ^ (p.yx));
				uint h32 = 1103515245U * ((p.x) ^ (p.y >> 3U));
				return h32 ^ (h32 >> 16);
			}

			float hash1(inout float seed) {
				//uint n = baseHash(floatBitsToUint(float2(seed += .1, seed += .1)));
				uint n = baseHash(asuint(float2(seed += .1, seed += .1)));
				return float(n) / float(0xffffffffU);
			}

			float2 hash2(inout float seed) {
				//uint n = baseHash(floatBitsToUint(vec2(seed += .1, seed += .1)));
				uint n = baseHash(asuint(float2(seed += .1, seed += .1)));
				uint2 rz = uint2(n, n * 48271U);
				
				//float2 hash2 = float2(rz.xy & uint2(0x7fffffffU, 0x7fffffffU));// / float(0x7fffffff);
				return float2(rz.xy & uint2(0x7fffffffU, 0x7fffffffU)) / float(0x7fffffff);
			}

			//
			// Ray tracer helper functions
			//

			float FresnelSchlickRoughness(float cosTheta, float F0, float roughness) {
				return F0 + (max((1. - roughness), F0) - F0) * pow(abs(1. - cosTheta), 5.0);
			}

			float3 cosWeightedRandomHemisphereDirection(const float3 n, inout float seed) {
				float2 r = hash2(seed);
				float3  uu = normalize(cross(n, abs(n.y) > .5 ? float3(1., 0., 0.) : float3(0., 1., 0.)));
				float3  vv = cross(uu, n);
				float ra = sqrt(r.y);
				float rx = ra * cos(6.28318530718*r.x);
				float ry = ra * sin(6.28318530718*r.x);
				float rz = sqrt(1. - r.y);
				float3  rr = float3(rx*uu + ry * vv + rz * n);
				return normalize(rr);
			}

			bool modifiedRefract(const in float3 v, const in float3 n, const in float ni_over_nt,
				out float3 refracted) {
				float dt = dot(v, n);
				float discriminant = 1. - ni_over_nt * ni_over_nt*(1. - dt * dt);
				if (discriminant > 0.) {
					refracted = ni_over_nt * (v - n * dt) - n * sqrt(discriminant);
					return true;
				}
				else {
					return false;
				}
			}

			float3 modifyDirectionWithRoughness(const float3 normal, const float3 n, const float roughness, inout float seed) {
				float2 r = hash2(seed);

				float3  uu = normalize(cross(n, abs(n.y) > .5 ? float3(1., 0., 0.) : float3(0., 1., 0.)));
				float3  vv = cross(uu, n);

				float a = roughness * roughness;

				float rz = sqrt(abs((1.0 - r.y) / clamp(1. + (a - 1.)*r.y, .00001, 1.)));
				float ra = sqrt(abs(1. - rz * rz));
				float rx = ra * cos(6.28318530718*r.x);
				float ry = ra * sin(6.28318530718*r.x);
				float3  rr = float3(rx*uu + ry * vv + rz * n);

				float3 ret = normalize(rr);
				return dot(ret, normal) > 0. ? ret : n;
			}

			float2 randomInUnitDisk(inout float seed) {
				float2 h = hash2(seed) * float2(1, 6.28318530718);
				float phi = h.y;
				float r = sqrt(h.x);
				return r * float2(sin(phi), cos(phi));
			}

			//
			// Scene description
			//

			float3 rotateY(const in float3 p, const in float t) {
				float co = cos(t);
				float si = sin(t);
				float2 xz = mul(p.xz, float2x2(co, si, -si, co));// float2x2(co, si, -si, co)*p.xz;
				return float3(xz.x, p.y, xz.y);
			}

			float3 opU(float3 d, float iResult, float mat) {
				return (iResult < d.y) ? float3(d.x, iResult, mat) : d;
			}

			float iMesh(in float3 ro, in float3 rd, in float2 distBound, inout float3 normal) {
				static const float3 tri0 = float3(-2. / 3. * 0.43301270189, 0, 0);
				static const float3 tri1 = float3(1. / 3. * 0.43301270189, 0, .25);
				static const float3 tri2 = float3(1. / 3. * 0.43301270189, 0, -.25);
				static const float3 tri3 = float3(0, 0.41079191812, 0);

				float2 d = distBound;
				d.y = min(d.y, iTriangle(ro, rd, d, normal, tri0, tri1, tri2));
				d.y = min(d.y, iTriangle(ro, rd, d, normal, tri0, tri3, tri1));
				d.y = min(d.y, iTriangle(ro, rd, d, normal, tri2, tri3, tri0));
				d.y = min(d.y, iTriangle(ro, rd, d, normal, tri1, tri3, tri2));

				return d.y < distBound.y ? d.y : MAX_DIST;
			}

			float3 worldhit(in float3 ro, in float3 rd, in float2 dist, out float3 normal) {
				float3 tmp0, tmp1, d = float3(dist, 0.);

				d = opU(d, iPlane(ro, rd, d.xy, normal, float3(0, 1, 0), 0.), 1.);
				d = opU(d, iBox(ro - float3(0, .250, 0), rd, d.xy, normal, float3(.25, .25, .25)), 2.);
				d = opU(d, iSphere(ro - float3(0, .250, 0), rd, d.xy, normal, .25), 3.);
				d = opU(d, iSphere(ro - float3(0, .250, 0), rd, d.xy, normal, 25.), 10.);
				d = opU(d, iSphere(ro - float3(0, .250, 0), rd, d.xy, normal, 15.), 10.);
				d = opU(d, iSphere(ro - float3(1.55, .250, 0.6), rd, d.xy, normal, .25), 12.);
				/*d = opU(d, iCylinder   (ro,                  rd, d.xy, normal, vec3(2.1,.1,-2), vec3(1.9,.5,-1.9), .08 ), 4.);
				d = opU(d, iCylinder   (ro-vec3( 1,.100,-2), rd, d.xy, normal, vec3(0,0,0), vec3(0,.4,0), .1 ), 5.);
				d = opU(d, iTorus      (ro-vec3( 0,.250, 1), rd, d.xy, normal, vec2(.2,.05)), 6.);*/
				d = opU(d, iCapsule    (ro-float3( 1.6,.000, 0), rd, d.xy, normal, float3(-.1,.1,-.1), float3(.2,.4,.2), .1), 7.);
				/*d = opU(d, iCone       (ro-vec3( 2,.200, 0), rd, d.xy, normal, vec3(.1,0,0), vec3(-.1,.3,.1), .15, .05), 8.);
				d = opU(d, iRoundedBox (ro-vec3( 0,.250,-2), rd, d.xy, normal, vec3(.15,.125,.15), .045), 9.);*/
				d = opU(d, iGoursat    (ro-float3( 1,.275, 0), rd, d.xy, normal, .16, .2), 10.);
				/*d = opU(d, iEllipsoid  (ro-vec3(-1,.300, 0), rd, d.xy, normal, vec3(.2,.25, .05)), 11.);
				d = opU(d, iRoundedCone(ro-vec3( 2,.200,-1), rd, d.xy, normal, vec3(.1,0,0), vec3(-.1,.3,.1), 0.15, 0.05), 12.);
				d = opU(d, iRoundedCone(ro-vec3(-1,.200,-2), rd, d.xy, normal, vec3(0,.3,0), vec3(0,0,0), .1, .2), 13.);*/
				d = opU(d, iMesh       (ro-float3( 0.3,.090, 0.6), rd, d.xy, normal), 14.);
				/*d = opU(d, iSphere4    (ro-vec3(-1,.275,-1), rd, d.xy, normal, .225), 15.);
				*/
				tmp1 = opU(d, iBox(rotateY(ro - float3(1.1, .25, -0.8), 0.78539816339), rotateY(rd, 0.78539816339), d.xy, tmp0, float3(.2, .2, .3)), 8.);
				if (tmp1.y < d.y) {
					d = tmp1;
					normal = rotateY(tmp0, -0.78539816339);
				}

				return d;
			}

			//
			// Palette by Íñigo Quílez: 
			// https://www.shadertoy.com/view/ll2GD3
			//
			float3 pal(in float t, in float3 a, in float3 b, in float3 c, in float3 d) {
				return a + b * cos(6.28318530718*(c*t + d));
			}

			// https://stackoverflow.com/questions/7610631/glsl-mod-vs-hlsl-fmod?rq=1 mod in GLSL is different from fmod in HLSL
			float mod(float x, float y)
			{
				return x - y * floor(x / y);
			}

			float checkerBoard(float2 p) {
				return mod(floor(p.x) + floor(p.y), 2.);
			}
			

			float3 getSkyColor(float3 rd) {
				float3 col = lerp(float3(1, 1, 1), float3(.5, .7, 1), .5 + .5*rd.y);
				float sun = clamp(dot(normalize(float3(-.4, .7, -.6)), rd), 0., 1.);
				col += float3(1, .6, .1)*(pow(sun, 4.) + 10.*pow(sun, 32.));
				return col;
			}

			#define LAMBERTIAN 0.
			#define METAL 1.
			#define DIELECTRIC 2.

			float gpuIndepentHash(float p) {
				p = frac(p * .1031);
				p *= p + 19.19;
				p *= p + p;
				return frac(p);
			}

			void getMaterialProperties(in float3 pos, in float mat,
				out float3 albedo, out float type, out float roughness) {
				albedo = pal(mat*.729996323 + .5, float3(.5, .5, .5), float3(.5, .5, .5), float3(1, 1, 1), float3(0, .1, .4));

				if (mat < 1.5) {
					albedo = float3(.25 + .25*checkerBoard(pos.xz * 5.), .25 + .25*checkerBoard(pos.xz * 5.), .25 + .25*checkerBoard(pos.xz * 5.));
					roughness = .75 *albedo.x - .15;
					type = METAL;
				}
				else {
					type = floor(gpuIndepentHash(mat + .3) * 3.);
					roughness = (1. - type * .475) * gpuIndepentHash(mat);
				}
			}

			//
			// Simple ray tracer
			//

			float schlick(float cosine, float r0) {
				return r0 + (1. - r0)*pow((1. - cosine), 5.);
			}
			float3 render(in float3 ro, in float3 rd, inout float seed) {
				float3 albedo, normal, col = float3(1., 1., 1.);
				float roughness, type;

				for (int i = 0; i < PATH_LENGTH; ++i) {
					float3 res = worldhit(ro, rd, float2(.0001, 100), normal);
					if (res.z > 0.) {
						ro += rd * res.y;

						getMaterialProperties(ro, res.z, albedo, type, roughness);

						if (type < LAMBERTIAN + .5) { // Added/hacked a reflection term
							float F = FresnelSchlickRoughness(max(0., -dot(normal, rd)), .04, roughness);
							if (F > hash1(seed)) {
								rd = modifyDirectionWithRoughness(normal, reflect(rd, normal), roughness, seed);
							}
							else {
								col *= albedo;
								rd = cosWeightedRandomHemisphereDirection(normal, seed);
							}
						}
						else if (type < METAL + .5) {
							col *= albedo;
							rd = modifyDirectionWithRoughness(normal, reflect(rd, normal), roughness, seed);
						}
						else { // DIELECTRIC
							float3 normalOut, refracted;
							float ni_over_nt, cosine, reflectProb = 1.;
							if (dot(rd, normal) > 0.) {
								normalOut = -normal;
								ni_over_nt = 1.4;
								cosine = dot(rd, normal);
								cosine = sqrt(1. - (1.4*1.4) - (1.4*1.4)*cosine*cosine);
							}
							else {
								normalOut = normal;
								ni_over_nt = 1. / 1.4;
								cosine = -dot(rd, normal);
							}

							if (modifiedRefract(rd, normalOut, ni_over_nt, refracted)) {
								float r0 = (1. - ni_over_nt) / (1. + ni_over_nt);
								reflectProb = FresnelSchlickRoughness(cosine, r0*r0, roughness);
							}

							rd = hash1(seed) <= reflectProb ? reflect(rd, normal) : refracted;
							rd = modifyDirectionWithRoughness(-normalOut, rd, roughness, seed);
						}
					}
					else {
						col *= getSkyColor(rd);
						return col;
					}
				}
				return float3(0, 0, 0);
			}

			float3x3 setCamera(in float3 ro, in float3 ta, float cr) {
				float3 cw = normalize(ta - ro);
				float3 cp = float3(sin(cr), cos(cr), 0.0);
				float3 cu = normalize(cross(cw, cp));
				float3 cv = (cross(cu, cw));
				return float3x3(cu, cv, cw);
			}


            fixed4 frag (v2f i) : SV_Target
            {
				i.uv.xy = (i.screenPos.xy / i.screenPos.w);
				iMouse = float2(iMouseX, iMouseY);
				//bool reset = iFrame == 0;
				bool reset = int(_Time.y) == 0;

				float2 mo = iMouse.xy == float2(0, 0) ? float2(.125, .125) :
					abs(iMouse.xy) / _ScreenParams.xy - .5;

				//float4 data = texelFetch(iChannel0, int2(0), 0);
				float4 pos = float4(int2(0, 0), 0, 0);
				//float4 data = tex2Dlod(iChannel0, int2(0, 0), 0, 0);
				float4 data = iChannel0.Load(pos); //tex2Dlod(iChannel0, pos);
				float numA = round(mo*_ScreenParams.xy);
				float numB = round(data.yz);
				if (numA != numB || round(data.w) != round(_ScreenParams.x)) {
					reset = true;
				}

				float3 ro = float3(.5 + 2.5*cos(1.5 + 6.*mo.x), 1. + 2.*mo.y, -.5 + 2.5*sin(1.5 + 6.*mo.x));
				float3 ta = float3(.5, -.4, -.5);
				float3x3 ca = setCamera(ro, ta, 0.);
				float3 normal;

				float fpd = data.x;
				//if (all(equal(int2(i.uv.xy * _ScreenParams.xy), int2(0, 0)))) {
				int2 numC = int2(i.uv.xy * _ScreenParams.xy);
				int2 numD = int2(0, 0);
				if (any(numC == numD)) {
					// Calculate focus plane.
					float nfpd = worldhit(ro, normalize(float3(.5, 0, -.5) - ro), float2(0, 100), normal).y;
					return float4(nfpd, mo*_ScreenParams.xy, _ScreenParams.x);
				}
				else {
					float2 p = (-_ScreenParams.xy + 2.*i.uv.xy * _ScreenParams.xy - 1.) / _ScreenParams.y;
					float seed = float(baseHash(asuint(p - _Time.y))) / float(0xffffffffU);

					// AA
					p += 2.*hash2(seed) / _ScreenParams.y;
					float3 rd = mul(normalize(float3(p.xy, 1.6)), ca); // ca * normalize(float3(p.xy, 1.6));

					// DOF
					float3 fp = ro + rd * fpd;
					//ro = ro + ca * float3(randomInUnitDisk(seed), 0.)*.02;
					ro = ro + mul(float3(randomInUnitDisk(seed), 0.), ca)*.0005;
					rd = normalize(fp - ro);

					float3 col = render(ro, rd, seed);

					if (reset) {
						return float4(col, 1);
						//return float4(1, 0, 0, 1);
					}
					else {
						float4 pos = float4(int2(i.uv.xy * _ScreenParams.xy), 0, 0);
						//return float4(col, 1) + tex2Dlod(iChannel0, pos);
						return float4(col, 1) + iChannel0.Load(pos);
					}
				}

            }
            ENDCG
        }
		GrabPass{"iChannel0"}
		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
				float4 screenPos : TEXCOORD1;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
				float4 screenPos : TEXCOORD1;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			Texture2D iChannel0;

			v2f vert(appdata v)
			{
				/*v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				UNITY_TRANSFER_FOG(o,o.vertex);
				return o;*/
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				o.screenPos = ComputeScreenPos(o.vertex);
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				//
				i.uv.xy = (i.screenPos.xy / i.screenPos.w);
				//
				float2 temp = float2(1, 1);
				float4 pos = float4(int2(i.uv.xy * _ScreenParams.xy), 0, 0);
				float4 data = iChannel0.Load(pos);// tex2Dlod(iChannel0, pos); //texelFetch(iChannel0, ivec2(fragCoord), 0);
				float3 col = data.rgb/ data.w;

				// gamma correction
				col = max(float3(0, 0, 0), col - 0.004);
				col = (col*(6.2*col + .5)) / (col*(6.2*col + 1.7) + 0.06);
				return float4(col, 1.0);
				//return tex2D(iChannel0, i.uv);
			}
			ENDCG
		}
		//GrabPass{"iChannel0"}
    }
}
