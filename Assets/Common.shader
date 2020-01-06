Shader "Unlit/Common"
{

    /*Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }*/
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

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
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

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

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);
                return col;
            }
            ENDCG
        }
    }
}
