#pragma once

// -----------------------------------------------------------
// scene.h
// Simple test scene for ray tracing experiments. Goals:
// - Super-fast scene intersection
// - Easy interface: scene.FindNearest / IsOccluded
// - With normals and albedo: GetNormal / GetAlbedo
// - Area light source (animated), for light transport
// - Primitives can be hit from inside - for dielectrics
// - Can be extended with other primitives and/or a BVH
// - Optionally animated - for temporal experiments
// - Not everything is axis aligned - for cache experiments
// - Can be evaluated at arbitrary time - for motion blur
// - Has some high-frequency details - for filtering
// -----------------------------------------------------------

#define FOURLIGHTS

#define PLANE_X(o,i) {t=-(ray.O.x+o)*ray.rD.x;if(t<ray.t&&t>0)ray.t=t,ray.objIdx=i;}
#define PLANE_Y(o,i) {t=-(ray.O.y+o)*ray.rD.y;if(t<ray.t&&t>0)ray.t=t,ray.objIdx=i;}
#define PLANE_Z(o,i) {t=-(ray.O.z+o)*ray.rD.z;if(t<ray.t&&t>0)ray.t=t,ray.objIdx=i;}

namespace Tmpl8 {
__declspec(align(64)) class Ray
{
public:
	Ray() = default;
	Ray( const float3 origin, const float3 direction, const float distance = 1e34f, const int idx = -1 )
	{
		O = origin, D = direction, t = distance;
		// calculate reciprocal ray direction for triangles and AABBs
		rD = float3( 1 / D.x, 1 / D.y, 1 / D.z );
		d0 = 1, d1 = d2 = 0; // ready for SIMD matrix math
		objIdx = idx;
	}
	float3 IntersectionPoint() const { return O + t * D; }
	// ray data
	union { struct { float3 O; float d0; }; __m128 O4; };
	union { struct { float3 D; float d1; }; __m128 D4; };
	union { struct { float3 rD; float d2; }; __m128 rD4; };
	float t = 1e34f;
	int objIdx = -1;
	bool inside = false; // true when in medium
};

// -----------------------------------------------------------
// Sphere primitive
// Basic sphere, with explicit support for rays that start
// inside it. Good candidate for a dielectric material.
// -----------------------------------------------------------
class Sphere
{
public:
	Sphere() = default;
	Sphere( int idx, float3 p, float r ) :
		pos( p ), r2( r* r ), invr( 1 / r ), objIdx( idx ) {}
	void Intersect( Ray& ray ) const
	{
		const __m128 oc = _mm_sub_ps( ray.O4, _mm_set_ps( 1, 1, this->pos.y, -1.8f ) );
		const float b = _mm_dp_ps( oc, ray.D4, 0xFF ).m128_f32[0];
		const float c = _mm_dp_ps( oc, oc, 0xFF ).m128_f32[0];
		float t, d = b * b - c;
		if (d <= 0) return;
		d = sqrtf( d ), t = -b - d;
		bool hit = t < ray.t && t > 0;
		if (hit)
		{
			ray.t = t, ray.objIdx = objIdx;
			return;
		}
		if (c > 0) return; // we're outside; safe to skip option 2
		t = d - b, hit = t < ray.t && t > 0;
		if (hit) ray.t = t, ray.objIdx = objIdx;
	}
	bool IsOccluded( const Ray& ray ) const
	{
		const __m128 oc = _mm_sub_ps( ray.O4, _mm_set_ps( 1, 1, this->pos.y, -1.8f ) );
		const float b = _mm_dp_ps( oc, ray.D4, 0xFF ).m128_f32[0];
		const float c = _mm_dp_ps( oc, oc, 0xFF ).m128_f32[0];
		float t, d = b * b - c;
		if (d <= 0) return false;
		d = sqrtf( d ), t = -b - d;
		bool hit = t < ray.t && t > 0;
		return hit;
	}
	float3 GetNormal( const float3 I ) const
	{
		return (I - this->pos) * invr;
	}
	float3 GetAlbedo( const float3 I ) const
	{
		return float3( 0.93f );
	}
	union
	{
		float3 pos;
		__m128 pos4;
	};
	float r2 = 0, invr = 0;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Plane primitive
// Basic infinite plane, defined by a normal and a distance
// from the origin (in the direction of the normal).
// -----------------------------------------------------------
class Plane
{
public:
	Plane() = default;
	Plane( int idx, float3 normal, float dist ) : N( normal ), d( dist ), objIdx( idx ) {}
	void Intersect( Ray& ray ) const
	{
		float t = -(dot( ray.O, this->N ) + this->d) / (dot( ray.D, this->N ));
		if (t < ray.t && t > 0) ray.t = t, ray.objIdx = objIdx;
	}
	float3 GetNormal( const float3 I ) const
	{
		return N;
	}
	float3 GetAlbedo( const float3 I ) const
	{
		if (N.y == 1)
		{
			// floor albedo: checkerboard
			int ix = (int)(I.x * 2 + 96.01f);
			int iz = (int)(I.z * 2 + 96.01f);
			// add deliberate aliasing to two tile
			if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
			if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
			return float3( ((ix + iz) & 1) ? 1 : 0.3f );
		}
		else if (N.z == -1)
		{
			// back wall: logo
			static Surface logo( "../assets/logo.png" );
			int ix = (int)((I.x + 4) * (128.0f / 8)), iy = (int)((2 - I.y) * (64.0f / 3));
			uint p = logo.pixels[(ix & 127) + (iy & 63) * 128];
			uint3 i3( (p >> 16) & 255, (p >> 8) & 255, p & 255 );
			return float3( i3 ) * (1.0f / 255.0f);
		}
		else if (N.x == 1)
		{
			// left wall: red
			static Surface red( "../assets/red.png" );
			int ix = (int)((I.z - 4) * (512.0f / 7)), iy = (int)((2 - I.y) * (512.0f / 3));
			uint p = red.pixels[(ix & 511) + (iy & 511) * 512];
			uint3 i3( (p >> 16) & 255, (p >> 8) & 255, p & 255 );
			return float3( i3 ) * (1.0f / 255.0f);
		}
		else if (N.x == -1)
		{
			// right wall: blue
			static Surface blue( "../assets/blue.png" );
			int ix = (int)((I.z - 4) * (512.0f / 7)), iy = (int)((2 - I.y) * (512.0f / 3));
			uint p = blue.pixels[(ix & 511) + (iy & 511) * 512];
			uint3 i3( (p >> 16) & 255, (p >> 8) & 255, p & 255 );
			return float3( i3 ) * (1.0f / 255.0f);
		}
		return float3( 0.93f );
	}
	float3 N;
	float d;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Cube primitive
// Oriented cube. Unsure if this will also work for rays that
// start inside it; maybe not the best candidate for testing
// dielectrics.
// -----------------------------------------------------------
class Cube
{
public:
	Cube() = default;
	Cube( int idx, float3 pos, float3 size, mat4 transform = mat4::Identity() )
	{
		objIdx = idx;
		b[0] = float4( pos - 0.5f * size, 1 );
		b[1] = float4( pos + 0.5f * size, 1 );
	}
	void Intersect( Ray& ray ) const
	{
		// AABB test
		__m128 t1 = _mm_mul_ps( _mm_sub_ps( bmin4, ray.O4 ), ray.rD4 );
		__m128 t2 = _mm_mul_ps( _mm_sub_ps( bmax4, ray.O4 ), ray.rD4 );
		__m128 vmax4 = _mm_max_ps( t1, t2 ), vmin4 = _mm_min_ps( t1, t2 );
		float tmax = min( vmax4.m128_f32[0], min( vmax4.m128_f32[1], vmax4.m128_f32[2] ) );
		float tmin = max( vmin4.m128_f32[0], max( vmin4.m128_f32[1], vmin4.m128_f32[2] ) );
		if (tmin < tmax) if (tmin > 0)
		{
			if (tmin < ray.t) ray.t = tmin, ray.objIdx = objIdx;
		}
		else if (tmax > 0)
		{
			if (tmax < ray.t) ray.t = tmax, ray.objIdx = objIdx;
		}
	}
	bool IsOccluded( const Ray& ray ) const
	{
		// AABB test
		__m128 t1 = _mm_mul_ps( _mm_sub_ps( bmin4, ray.O4 ), ray.rD4 );
		__m128 t2 = _mm_mul_ps( _mm_sub_ps( bmax4, ray.O4 ), ray.rD4 );
		__m128 vmax4 = _mm_max_ps( t1, t2 ), vmin4 = _mm_min_ps( t1, t2 );
		float tmax = min( vmax4.m128_f32[0], min( vmax4.m128_f32[1], vmax4.m128_f32[2] ) );
		float tmin = max( vmin4.m128_f32[0], max( vmin4.m128_f32[1], vmin4.m128_f32[2] ) );
		return tmax > 0 && tmin < tmax && tmin < ray.t;
	}
	float3 GetNormal( const float3 I ) const
	{
		// determine normal in object space
		float3 N = float3( -1, 0, 0 );
		float d0 = fabs( I.x - b[0].x ), d1 = fabs( I.x - b[1].x );
		float d2 = fabs( I.y - b[0].y ), d3 = fabs( I.y - b[1].y );
		float d4 = fabs( I.z - b[0].z ), d5 = fabs( I.z - b[1].z );
		float minDist = d0;
		if (d1 < minDist) minDist = d1, N.x = 1;
		if (d2 < minDist) minDist = d2, N = float3( 0, -1, 0 );
		if (d3 < minDist) minDist = d3, N = float3( 0, 1, 0 );
		if (d4 < minDist) minDist = d4, N = float3( 0, 0, -1 );
		if (d5 < minDist) minDist = d5, N = float3( 0, 0, 1 );
		// return normal in world space
		return N;
	}
	float3 GetAlbedo( const float3 I ) const
	{
		return float3( 1, 1, 1 );
	}
	union { float4 b[2]; struct { __m128 bmin4, bmax4; }; };
	int objIdx = -1;
};

// -----------------------------------------------------------
// Quad primitive
// Oriented quad, intended to be used as a light source.
// -----------------------------------------------------------
class Quad
{
public:
	Quad() = default;
	Quad( int idx, float s, mat4 transform = mat4::Identity() )
	{
		objIdx = idx;
		size = s * 0.5f;
		T = transform, invT = transform.FastInvertedTransformNoScale();
	}
	void Intersect( Ray& ray ) const
	{
		const float Oy = invT.cell[4] * ray.O.x + invT.cell[5] * ray.O.y + invT.cell[6] * ray.O.z + invT.cell[7];
		const float Dy = invT.cell[4] * ray.D.x + invT.cell[5] * ray.D.y + invT.cell[6] * ray.D.z;
		const float t = Oy / -Dy;
		if (t < ray.t && t > 0)
		{
			const float Ox = invT.cell[0] * ray.O.x + invT.cell[1] * ray.O.y + invT.cell[2] * ray.O.z + invT.cell[3];
			const float Oz = invT.cell[8] * ray.O.x + invT.cell[9] * ray.O.y + invT.cell[10] * ray.O.z + invT.cell[11];
			const float Dx = invT.cell[0] * ray.D.x + invT.cell[1] * ray.D.y + invT.cell[2] * ray.D.z;
			const float Dz = invT.cell[8] * ray.D.x + invT.cell[9] * ray.D.y + invT.cell[10] * ray.D.z;
			const float Ix = Ox + t * Dx, Iz = Oz + t * Dz;
			if (Ix > -size && Ix < size && Iz > -size && Iz < size)
				ray.t = t, ray.objIdx = objIdx;
		}
	}
	bool IsOccluded( const Ray& ray ) const
	{
		const float Oy = invT.cell[4] * ray.O.x + invT.cell[5] * ray.O.y + invT.cell[6] * ray.O.z + invT.cell[7];
		const float Dy = invT.cell[4] * ray.D.x + invT.cell[5] * ray.D.y + invT.cell[6] * ray.D.z;
		const float t = Oy / -Dy;
		if (t < ray.t && t > 0)
		{
			const float Ox = invT.cell[0] * ray.O.x + invT.cell[1] * ray.O.y + invT.cell[2] * ray.O.z + invT.cell[3];
			const float Oz = invT.cell[8] * ray.O.x + invT.cell[9] * ray.O.y + invT.cell[10] * ray.O.z + invT.cell[11];
			const float Dx = invT.cell[0] * ray.D.x + invT.cell[1] * ray.D.y + invT.cell[2] * ray.D.z;
			const float Dz = invT.cell[8] * ray.D.x + invT.cell[9] * ray.D.y + invT.cell[10] * ray.D.z;
			const float Ix = Ox + t * Dx, Iz = Oz + t * Dz;
			return Ix > -size && Ix < size && Iz > -size && Iz < size;
		}
		return false;
	}
	float3 GetNormal( const float3 I ) const
	{
		return float3( -T.cell[1], -T.cell[5], -T.cell[9] );
	}
	float3 GetAlbedo( const float3 I ) const
	{
		return float3( 10 );
	}
	float size;
	mat4 T, invT;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Torus primitive - Inigo Quilez, ShaderToy 4sBGDy
// -----------------------------------------------------------
class Torus
{
public:
	Torus() = default;
	Torus::Torus( int idx, float a, float b ) : objIdx( idx )
	{
		rc2 = a * a, rt2 = b * b;
		r2 = sqrf( a + b );
	}
	void Intersect( Ray& ray ) const
	{
		// via: https://www.shadertoy.com/view/4sBGDy
		float3 O = ray.O - float3( 0, 0, 1.5f ), D = ray.D;
		// extension rays need double precision for the quadratic solver!
		double po = 1, m = dot( O, O ), k3 = dot( O, D ), k32 = k3 * k3;
		// bounding sphere test
		double v = k32 - m + r2;
		if (v < 0) return;
		// setup torus intersection
		double k = (m - rt2 - rc2) * 0.5, k2 = k32 + rc2 * D.z * D.z + k;
		double k1 = k * k3 + rc2 * O.z * D.z, k0 = k * k + rc2 * O.z * O.z - rc2 * rt2;
		// solve quadratic equation
		if (fabs( k3 * (k32 - k2) + k1 ) < 0.0001)
		{
			swap( k1, k3 );
			po = -1, k0 = 1 / k0, k1 = k1 * k0, k2 = k2 * k0, k3 = k3 * k0, k32 = k3 * k3;
		}
		double c2 = 2 * k2 - 3 * k32, c1 = k3 * (k32 - k2) + k1;
		double c0 = k3 * (k3 * (-3 * k32 + 4 * k2) - 8 * k1) + 4 * k0;
		c2 *= 0.33333333333, c1 *= 2, c0 *= 0.33333333333;
		double Q = c2 * c2 + c0, R = 3 * c0 * c2 - c2 * c2 * c2 - c1 * c1;
		double h = R * R - Q * Q * Q, z;
		if (h < 0)
		{
			const double sQ = sqrt( Q );
			z = 2 * sQ * cos( acos( R / (sQ * Q) ) * 0.33333333333 );
		}
		else
		{
		#if 1
			const double sQ = cbrtFast( sqrt( h ) + fabs( R ) ); // by Jeppe Vroegindeweij
		#else
			const double sQ = cbrt( sqrt( h ) + fabs( R ) ); // pow( sqrt( h ) + fabs( R ), 0.3333333 );
		#endif
			z = copysign( fabs( sQ + Q / sQ ), R );
		}
		z = c2 - z;
		double d1 = z - 3 * c2, d2 = z * z - 3 * c0;
		if (fabs( d1 ) < 1.0e-8)
		{
			if (d2 < 0) return;
			d2 = sqrt( d2 );
		}
		else
		{
			if (d1 < 0) return;
			d1 = sqrt( d1 * 0.5 ), d2 = c1 / d1;
		}
		double t = 1e20;
		h = d1 * d1 - z + d2;
		if (h > 0)
		{
			h = sqrt( h );
			double t1 = -d1 - h - k3, t2 = -d1 + h - k3;
			t1 = (po < 0) ? 2 / t1 : t1, t2 = (po < 0) ? 2 / t2 : t2;
			if (t1 > 0) t = t1;
			if (t2 > 0) t = min( t, t2 );
		}
		h = d1 * d1 - z - d2;
		if (h > 0)
		{
			h = sqrt( h );
			double t1 = d1 - h - k3, t2 = d1 + h - k3;
			t1 = (po < 0) ? 2 / t1 : t1, t2 = (po < 0) ? 2 / t2 : t2;
			if (t1 > 0) t = min( t, t1 );
			if (t2 > 0) t = min( t, t2 );
		}
		float ft = (float)t;
		if (ft > 0 && ft < ray.t) ray.t = ft, ray.objIdx = objIdx;
	}
	bool IsOccluded( const Ray& ray ) const
	{
		// via: https://www.shadertoy.com/view/4sBGDy
		float3 O = ray.O - float3( 0, 0, 1.5f ), D = ray.D;
		float po = 1, m = dot( O, O ), k3 = dot( O, D ), k32 = k3 * k3;
		// bounding sphere test
		float v = k32 - m + r2;
		if (v < 0.0) return false;
		// setup torus intersection
		float k = (m - rt2 - rc2) * 0.5f, k2 = k32 + rc2 * D.z * D.z + k;
		float k1 = k * k3 + rc2 * O.z * D.z, k0 = k * k + rc2 * O.z * O.z - rc2 * rt2;
		// solve quadratic equation
		if (fabs( k3 * (k32 - k2) + k1 ) < 0.01f)
		{
			swap( k1, k3 );
			po = -1, k0 = 1 / k0, k1 = k1 * k0, k2 = k2 * k0, k3 = k3 * k0, k32 = k3 * k3;
		}
		float c2 = 2 * k2 - 3 * k32, c1 = k3 * (k32 - k2) + k1;
		float c0 = k3 * (k3 * (-3 * k32 + 4 * k2) - 8 * k1) + 4 * k0;
		c2 *= 0.33333333333f, c1 *= 2, c0 *= 0.33333333333f;
		float Q = c2 * c2 + c0, R = 3 * c0 * c2 - c2 * c2 * c2 - c1 * c1;
		float h = R * R - Q * Q * Q, z = 0;
		if (h < 0)
		{
			const float sQ = sqrtf( Q );
			z = 2 * sQ * cosf( acosf( R / (sQ * Q) ) * 0.3333333f );
		}
		else
		{
		#if 1
			const float sQ = (float)cbrtFast( sqrtf( h ) + fabsf( R ) );
		#else
			const float sQ = cbrtf( sqrtf( h ) + fabs( R ) ); // powf( sqrtf( h ) + fabs( R ), 0.3333333f );
		#endif
			z = copysign( fabs( sQ + Q / sQ ), R );
		}
		z = c2 - z;
		float d1 = z - 3 * c2, d2 = z * z - 3 * c0;
		if (fabs( d1 ) < 1.0e-4f)
		{
			if (d2 < 0) return false;
			d2 = sqrtf( d2 );
		}
		else
		{
			if (d1 < 0.0) return false;
			d1 = sqrtf( d1 * 0.5f ), d2 = c1 / d1;
		}
		float t = 1e20f;
		h = d1 * d1 - z + d2;
		if (h > 0)
		{
			float t1 = -d1 - sqrtf( h ) - k3;
			t1 = (po < 0) ? 2 / t1 : t1;
			if (t1 > 0 && t1 < ray.t) return true;
		}
		h = d1 * d1 - z - d2;
		if (h > 0)
		{
			float t1 = d1 - sqrtf( h ) - k3;
			t1 = (po < 0) ? 2 / t1 : t1;
			if (t1 > 0 && t1 < ray.t) return true;
		}
		return false;
	}
	float3 GetNormal( const float3 I ) const
	{
		const float3 L = I - float3( 0, 0, 1.5f );
		return normalize( L * (dot( L, L ) - rt2 - rc2 * float3( 1, 1, -1 )) );
	}
	float3 Torus::GetAlbedo( const float3 I ) const
	{
		return float3( 1 ); // material.albedo;
	}
	float rt2, rc2, r2;
	int objIdx;
	// these are helper functions for the torus code
	// these function will find the cubic root up till a certain accuracy using the newtonian method
	float cbrtfFast( const float n ) const {
		float x1 = n / 10.0f, x2 = 1.0f;
		int turn = 0;
		while (fabs( x1 - x2 ) > 0.00000001 && turn++ < 100)
			x1 = x2, x2 = (2.0f / 3.0f * x1) + (n / (3.0f * x1 * x1));
		return x2;
	}
	double cbrtFast( const double n ) const {
		double x1 = n / 10.0f, x2 = 1.0f;
		int turn = 0;
		while (fabs( x1 - x2 ) > 0.00000001 && turn++ < 100)
			x1 = x2, x2 = (2.0f / 3.0f * x1) + (n / (3.0f * x1 * x1));
		return x2;
	}
};

// -----------------------------------------------------------
// Scene class
// We intersect this. The query is internally forwarded to the
// list of primitives, so that the nearest hit can be returned.
// For this hit (distance, obj id), we can query the normal and
// albedo.
// -----------------------------------------------------------
class Scene
{
public:
	Scene()
	{
		// we store all primitives in one continuous buffer
	#ifdef FOURLIGHTS
		for (int i = 0; i < 4; i++) quad[i] = Quad( 0, 0.5f );	// 0: four light sources
	#else
		quad = Quad( 0, 1 );									// 0: light source
	#endif
		sphere = Sphere( 1, float3( 0 ), 0.6f );				// 1: bouncing ball
		sphere2 = Sphere( 2, float3( 0, 2.5f, -3.07f ), 8 );	// 2: rounded corners
		cube = Cube( 3, float3( 2, 0, 2 ), float3( 1.2f ) );	// 3: cube
		plane[0] = Plane( 4, float3( 1, 0, 0 ), 3 );			// 4: left wall
		plane[1] = Plane( 5, float3( -1, 0, 0 ), 2.99f );		// 5: right wall
		plane[2] = Plane( 6, float3( 0, 1, 0 ), 1 );			// 6: floor
		plane[3] = Plane( 7, float3( 0, -1, 0 ), 2 );			// 7: ceiling
		plane[4] = Plane( 8, float3( 0, 0, 1 ), 3 );			// 8: front wall
		plane[5] = Plane( 9, float3( 0, 0, -1 ), 3.99f );		// 9: back wall
		torus = Torus( 10, 0.8f, 0.25f );						// 10: torus
		SetTime( 0 );
		// Note: once we have triangle support we should get rid of the class
		// hierarchy: virtuals reduce performance somewhat.
	}
	void SetTime( float t )
	{
		// default time for the scene is simply 0. Updating/ the time per frame
		// enables animation. Updating it per ray can be used for motion blur.
		animTime = t;
	#ifdef FOURLIGHTS
		// four light sources are stationary
		quad[0].T = mat4::Translate( -1, 1.5f, -1 ), quad[0].invT = quad[0].T.FastInvertedTransformNoScale();
		quad[1].T = mat4::Translate( 1, 1.5f, -1 ), quad[1].invT = quad[1].T.FastInvertedTransformNoScale();
		quad[2].T = mat4::Translate( 1, 1.5f, 1 ), quad[2].invT = quad[2].T.FastInvertedTransformNoScale();
		quad[3].T = mat4::Translate( -1, 1.5f, 1 ), quad[3].invT = quad[3].T.FastInvertedTransformNoScale();
	#else
		// light source animation: swing
		mat4 M1base = mat4::Translate( float3( 0, 2.6f, 2 ) );
		mat4 M1 = M1base * mat4::RotateZ( sinf( animTime * 0.6f ) * 0.1f ) * mat4::Translate( float3( 0, -0.9f, 0 ) );
		quad.T = M1, quad.invT = M1.FastInvertedTransformNoScale();
	#endif
		// sphere animation: bounce
		float tm = 1 - sqrf( fmodf( animTime, 2.0f ) - 1 );
		sphere.pos = float3( -1.8f, -0.4f + tm, 1 );
	}
	float3 GetLightPos() const
	{
	#ifndef FOURLIGHTS
		// light point position is the middle of the swinging quad
		float3 corner1 = TransformPosition( float3( -0.5f, 0, -0.5f ), quad.T );
		float3 corner2 = TransformPosition( float3( 0.5f, 0, 0.5f ), quad.T );
		return (corner1 + corner2) * 0.5f - float3( 0, 0.01f, 0 );
	#else
		// function is not valid when using four lights; we'll return the origin
		return float3( 0 );
	#endif
	}
	float3 RandomPointOnLight( const float r0, const float r1 ) const
	{
	#ifndef FOURLIGHTS
		// get a random position on the swinging quad
		const float size = quad.size;
		float3 corner1 = TransformPosition( float3( -size, 0, -size ), quad.T );
		float3 corner2 = TransformPosition( float3( size, 0, -size ), quad.T );
		float3 corner3 = TransformPosition( float3( -size, 0, size ), quad.T );
		return corner1 + r0 * (corner2 - corner1) + r1 * (corner3 - corner1);
	#else
		// select a random light and use that
		uint lightIdx = (uint)(r0 * 4);
		const Quad& q = quad[lightIdx];
		// renormalize r0 for reuse
		float stratum = lightIdx * 0.25f;
		float r2 = (r0 - stratum) / (1 - stratum);
		// get a random position on the selected quad
		const float size = q.size;
		float3 corner1 = TransformPosition( float3( -size, 0, -size ), q.T );
		float3 corner2 = TransformPosition( float3( size, 0, -size ), q.T );
		float3 corner3 = TransformPosition( float3( -size, 0, size ), q.T );
		return corner1 + r2 * (corner2 - corner1) + r1 * (corner3 - corner1);
	#endif
	}
	float3 RandomPointOnLight( uint& seed ) const
	{
		return RandomPointOnLight( RandomFloat( seed ), RandomFloat( seed ) );
	}
	void GetLightQuad( float3& v0, float3& v1, float3& v2, float3& v3, const uint idx = 0 )
	{
	#ifndef FOURLIGHTS
		// return the four corners of the swinging quad, clockwise, for solid angle sampling
		const Quad& q = quad;
	#else
		// return four corners of the specified light
		const Quad& q = quad[idx];
	#endif
		const float size = q.size;
		v0 = TransformPosition( float3( -size, 0, size ), q.T );
		v1 = TransformPosition( float3( size, 0, size ), q.T );
		v2 = TransformPosition( float3( size, 0, -size ), q.T );
		v3 = TransformPosition( float3( -size, 0, -size ), q.T );
	}
	float3 GetLightColor() const
	{
		return float3( 24, 24, 22 );
	}
	float3 GetAreaLightColor() const
	{
	#ifdef FOURLIGHTS
		return quad[0].GetAlbedo( float3( 0 ) ); // they're all the same color
	#else
		return quad.GetAlbedo( float3( 0 ) );
	#endif
	}
	float GetLightArea() const
	{
	#ifdef FOURLIGHTS
		return sqrf( quad[0].size * 2 ); // all the same size
	#else
		return sqrf( quad.size * 2 );
	#endif
	}
	constexpr float GetLightCount() const
	{
	#ifdef FOURLIGHTS
		return 4; // what did you expect
	#else
		return 1;
	#endif
	}
	void FindNearest( Ray& ray ) const
	{
		// room walls - ugly shortcut for more speed
		const float3 spos = sphere.pos;
		const float3 ro = ray.O, rd = ray.D;
		// TODO: the room is actually just an AABB; use slab test
		static const __m128 x4min = _mm_setr_ps( 3, 1, 3, 1e30f );
		static const __m128 x4max = _mm_setr_ps( -2.99f, -2, -3.99f, 1e30f );
		static const __m128 idmin = _mm_castsi128_ps( _mm_setr_epi32( 4, 6, 8, -1 ) );
		static const __m128 idmax = _mm_castsi128_ps( _mm_setr_epi32( 5, 7, 9, -1 ) );
		static const __m128 zero4 = _mm_setzero_ps();
		const __m128 selmask = _mm_cmpge_ps( ray.D4, zero4 );
		const __m128i idx4 = _mm_castps_si128( _mm_blendv_ps( idmin, idmax, selmask ) );
		const __m128 x4 = _mm_blendv_ps( x4min, x4max, selmask );
		const __m128 d4 = _mm_sub_ps( zero4, _mm_mul_ps( _mm_add_ps( ray.O4, x4 ), ray.rD4 ) );
		const __m128 mask4 = _mm_cmple_ps( d4, zero4 );
		const __m128 t4 = _mm_blendv_ps( d4, _mm_set1_ps( 1e34f ), mask4 );
		/* first: unconditional */  ray.t = t4.m128_f32[0], ray.objIdx = idx4.m128i_i32[0];
		if (t4.m128_f32[1] < ray.t) ray.t = t4.m128_f32[1], ray.objIdx = idx4.m128i_i32[1];
		if (t4.m128_f32[2] < ray.t) ray.t = t4.m128_f32[2], ray.objIdx = idx4.m128i_i32[2];
	#ifdef FOURLIGHTS
		// efficient four-quad intersection by Jesse Vrooman
		const __m128 t = _mm_div_ps( _mm_add_ps( _mm_set1_ps( ray.O.y ),
			_mm_set1_ps( -1.5 ) ), _mm_xor_ps( _mm_set1_ps( ray.D.y ), _mm_set1_ps( -0.0 ) ) );
		const __m128 Ix = _mm_add_ps( _mm_add_ps( _mm_set1_ps( ray.O.x ),
			_mm_set_ps( 1, -1, -1, 1 ) ), _mm_mul_ps( t, _mm_set1_ps( ray.D.x ) ) );
		const __m128 Iz = _mm_add_ps( _mm_add_ps( _mm_set1_ps( ray.O.z ),
			_mm_set_ps( 1, 1, -1, -1 ) ), _mm_mul_ps( t, _mm_set1_ps( ray.D.z ) ) );
		const static __m128 size = _mm_set1_ps( 0.25f );
		const static __m128 nsize = _mm_xor_ps( _mm_set1_ps( 0.25f ), _mm_set1_ps( -0.0 ) );
		const __m128 maskedT = _mm_and_ps( t, _mm_and_ps(
			_mm_and_ps( _mm_cmpgt_ps( Ix, nsize ), _mm_cmplt_ps( Ix, size ) ),
			_mm_and_ps( _mm_cmpgt_ps( Iz, nsize ), _mm_cmplt_ps( Iz, size ) ) ) );
		if (maskedT.m128_f32[3] > 0) ray.t = maskedT.m128_f32[3], ray.objIdx = 0;
		if (maskedT.m128_f32[2] > 0) ray.t = maskedT.m128_f32[2], ray.objIdx = 0;
		if (maskedT.m128_f32[1] > 0) ray.t = maskedT.m128_f32[1], ray.objIdx = 0;
		if (maskedT.m128_f32[0] > 0) ray.t = maskedT.m128_f32[0], ray.objIdx = 0;
	#else
		quad.Intersect( ray );
	#endif
		{
			// SIMD sphere intersection code by Jesse Vrooman
			const __m128 oc = _mm_sub_ps( ray.O4, sphere.pos4 );
			const float b = _mm_dp_ps( oc, ray.D4, 0x71 ).m128_f32[0];
			const float d = b * b - (_mm_dp_ps( oc, oc, 0x71 ).m128_f32[0] - 0.36f);
			if (d > 0)
			{
				const float t = -b - sqrtf( d );
				const bool hit = t < ray.t && t > 0;
				if (hit) { ray.t = t, ray.objIdx = 1; }
			};
		}
		{
			// SIMD sphere intersection code by Jesse Vrooman
			const static __m128 s4 = _mm_setr_ps( 0, 2.5f, -3.07f, 1 );
			const __m128 oc = _mm_sub_ps( ray.O4, s4 );
			const float b = _mm_dp_ps( oc, ray.D4, 0x71 ).m128_f32[0];
			const float d = b * b - (_mm_dp_ps( oc, oc, 0x71 ).m128_f32[0] - 64.0f);
			if (d > 0)
			{
				const float t = sqrtf( d ) - b;
				const bool hit = t < ray.t && t > 0;
				if (hit) { ray.t = t, ray.objIdx = 2; }
			};
		}
		cube.Intersect( ray );
		torus.Intersect( ray );
	}
	bool IsOccluded( const Ray& ray ) const
	{
		if (cube.IsOccluded( ray )) return true;
		const float3 oc = ray.O - sphere.pos;
		const float b = dot( oc, ray.D ), c = dot( oc, oc ) - (0.6f * 0.6f);
		const float d = b * b - c;
		if (d > 0)
		{
			const float t = -b - sqrtf( d );
			const bool hit = t < ray.t && t > 0;
			if (hit) return true;
		}
	#ifdef FOURLIGHTS
		for (int i = 0; i < 4; i++) if (quad[i].IsOccluded( ray )) return true;
	#else
		if (quad.IsOccluded( ray )) return true;
	#endif
		if (torus.IsOccluded( ray )) return true;
		return false; // skip planes and rounded corners
	}
	float3 GetNormal( const int objIdx, const float3 I, const float3 wo ) const
	{
		// we get the normal after finding the nearest intersection:
		// this way we prevent calculating it multiple times.
		if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
		float3 N;
	#ifdef FOURLIGHTS
		if (objIdx == 0) N = quad[0].GetNormal( I ); // they're all oriented the same
	#else
		if (objIdx == 0) N = quad.GetNormal( I );
	#endif
		else if (objIdx == 1) N = sphere.GetNormal( I );
		else if (objIdx == 2) N = sphere2.GetNormal( I );
		else if (objIdx == 3) N = cube.GetNormal( I );
		else if (objIdx == 10) N = torus.GetNormal( I );
		else
		{
			// faster to handle the 6 planes without a call to GetNormal
			N = float3( 0 );
			N[(objIdx - 4) / 2] = 1 - 2 * (float)(objIdx & 1);
		}
		if (dot( N, wo ) > 0) N = -N; // hit backside / inside
		return N;
	}
	float3 GetAlbedo( int objIdx, float3 I ) const
	{
		if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
	#ifdef FOURLIGHTS
		if (objIdx == 0) return quad[0].GetAlbedo( I ); // they're all the same
	#else
		if (objIdx == 0) return quad.GetAlbedo( I );
	#endif
		if (objIdx == 1) return sphere.GetAlbedo( I );
		if (objIdx == 2) return sphere2.GetAlbedo( I );
		if (objIdx == 3) return cube.GetAlbedo( I );
		if (objIdx == 10) return torus.GetAlbedo( I );
		return plane[objIdx - 4].GetAlbedo( I );
		// once we have triangle support, we should pass objIdx and the bary-
		// centric coordinates of the hit, instead of the intersection location.
	}
	float GetReflectivity( int objIdx, float3 I ) const
	{
		if (objIdx == 1 /* ball */) return 1;
		if (objIdx == 6 /* floor */) return 0.3f;
		return 0;
	}
	float GetRefractivity( int objIdx, float3 I ) const
	{
		return (objIdx == 3 || objIdx == 10) ? 1.0f : 0.0f;
	}
	float3 GetAbsorption( int objIdx )
	{
		return objIdx == 3 ? float3( 0.5f, 0, 0.5f ) : float3( 0 );
	}
	__declspec(align(64)) // start a new cacheline here
		float animTime = 0;
#ifdef FOURLIGHTS
	Quad quad[4];
#else
	Quad quad, dummyQuad1, dummyQuad2, dummyQuad3;
#endif
	Sphere sphere;
	Sphere sphere2;
	Cube cube;
	Plane plane[6];
	Torus torus;
};
}