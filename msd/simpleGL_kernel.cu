/*
	SimpleGL SDK example modified to prepare for a simple 
	Spring-mass-damper model implementation.
*/

#ifndef _SIMPLEGL_KERNEL_H_
#define _SIMPLEGL_KERNEL_H_

// Cuda utilities
#include "uint_util.hcu"
#include "float_util.hcu"
#include "float_util_device.hcu"

__global__ void msd_initialize_kernel( float4 *dataPtr, float3 offset, uint3 dims )
{
	// Index in position array
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if( idx<prod(dims) ){

		// 3D coordinate of current particle. 
		const uint3 co = idx_to_co( idx, dims );

		// Output
		float3 pos = uintd_to_floatd(co);
		pos /= uintd_to_floatd(dims);
		pos = pos*2.0f - make_float3(1.0f, 1.0f, 1.0f);

		// rotate
		float x = 0.52532198881f * pos.x - 0.85090352453f * pos.y;
		float y = 0.85090352453f * pos.x + 0.52532198881f * pos.y;
		pos.x = x;
		pos.y = y;

		// offset
		pos += offset;

		dataPtr[idx] = make_float4( pos.x, pos.y, pos.z, 1.0f );
	}
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to implement numerical integration. EXTEND WITH MSD SYSTEM.
//! @param pos  vertex positiond in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void msd_kernel( float4 *_old_pos, float4 *_cur_pos, float4 *_new_pos, uint3 dims )
{
	// Index in position array
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if( idx<prod(dims) ){

		// Time step size
		const float dt = 0.40f; //0.1f;

		// 3D coordinate of current particle. 
		// Can be offset to access neighbors. E.g.: upIdx = co_to_idx(co-make_uint3(0,1,0), dims). <-- Be sure to take speciel care for border cases!
		const uint3 co = idx_to_co( idx, dims );

		// Get the two previous positions
		const float3 old_pos = crop_last_dim(_old_pos[idx]);
		const float3 cur_pos = crop_last_dim(_cur_pos[idx]);

		const float3 zero_force = make_float3(0.0, 0.0, 0.0);
		
		const float k = .5;
		// const float l = 0.1;
		const float lambda = 0.01;
		const float gravity = -0.001f;


		// float3 up_force = zero_force;
		// float3 down_force = zero_force;
		// float3 left_force = zero_force;
		// float3 right_force = zero_force;
		// float3 front_force = zero_force;
		// float3 back_force = zero_force;

		// // UP
		// if(co.y != 0) {
		// 	const uint3 up_co = co - make_uint3(0,1,0);
		// 	const unsigned int up_idx = co_to_idx(up_co, dims);
		// 	const float3 up_pos = crop_last_dim(_cur_pos[up_idx]);
		// 	float distance = length(cur_pos - up_pos);
		// 	if(distance == 0.0f) distance = 0.00000001f; // avoiding division by zero
		// 	up_force = k * (l - distance) * (cur_pos - up_pos)/distance;
		// }
		// // DOWN
		// if(co.y != dims.y-1) {
		// 	const uint3 down_co = co + make_uint3(0,1,0);
		// 	const unsigned int down_idx = co_to_idx(down_co, dims);
		// 	const float3 down_pos = crop_last_dim(_cur_pos[down_idx]);
		// 	float distance = length(cur_pos - down_pos);
		// 	if(distance == 0.0f) distance = 0.00000001f; // avoiding division by zero
		// 	down_force = k * (l - distance) * (cur_pos - down_pos)/distance;
		// }
		// // LEFT
		// if(co.x != 0) {
		// 	const uint3 left_co = co - make_uint3(1,0,0);
		// 	const unsigned int left_idx = co_to_idx(left_co, dims);
		// 	const float3 left_pos = crop_last_dim(_cur_pos[left_idx]);
		// 	left_force = k * (l - length(cur_pos - left_pos)) * (cur_pos - left_pos)/length(cur_pos - left_pos);
		// }
		// // RIGHT
		// if(co.x != dims.x-1) {
		// 	const uint3 right_co = co + make_uint3(1,0,0);
		// 	const unsigned int right_idx = co_to_idx(right_co, dims);
		// 	const float3 right_pos = crop_last_dim(_cur_pos[right_idx]);
		// 	right_force = k * (l - length(cur_pos - right_pos)) * (cur_pos - right_pos)/length(cur_pos - right_pos);
		// }
		// // FRONT
		// if(co.z != 0) {
		// 	const uint3 front_co = co - make_uint3(0,0,1);
		// 	const unsigned int front_idx = co_to_idx(front_co, dims);
		// 	const float3 front_pos = crop_last_dim(_cur_pos[front_idx]);
		// 	front_force = k * (l - length(cur_pos - front_pos)) * (cur_pos - front_pos)/length(cur_pos - front_pos);
		// }
		// // BACK
		// if(co.z != dims.z-1) {
		// 	const uint3 back_co = co + make_uint3(0,0,1);
		// 	const unsigned int back_idx = co_to_idx(back_co, dims);
		// 	const float3 back_pos = crop_last_dim(_cur_pos[back_idx]);
		// 	back_force = k * (l - length(cur_pos - back_pos)) * (cur_pos - back_pos)/length(cur_pos - back_pos);
		// }


		float3 spring_acceleration = zero_force;

		for(int xi = -1; xi <= 1; xi++) {
			int x = co.x + xi;
			if(x < 0 || x > dims.x-1) continue;
			for(int yi = -1; yi <= 1; yi++) {
				int y = co.y + yi;
				if(y < 0 || y > dims.y-1) continue;
				for(int zi = -1; zi <= 1; zi++) {
					int z = co.z + zi;
					if(z < 0 || z > dims.z-1) continue;
					if(xi == 0 && yi == 0 && zi == 0) continue;

					const float l = length(make_float3(x, y, z) - uintd_to_floatd(co)) * 0.1f;
					const uint3 neighbor_co = make_uint3((uint)x, (uint)y, (uint)z);
					const unsigned int neighbor_idx = co_to_idx(neighbor_co, dims);
					const float3 neighbor_pos = crop_last_dim(_cur_pos[neighbor_idx]);
					float distance = length(cur_pos - neighbor_pos);
					if(distance == 0.0f) distance = 0.00000001f; // avoiding division by zero
					const float3 neighbor_force = k * (l - distance) * (cur_pos - neighbor_pos)/distance;
					spring_acceleration += neighbor_force;
				}
			}
		}


		// spring force acceleration
		// const float3 spring_acceleration = up_force + down_force + left_force + right_force + front_force + back_force;
		// const float3 spring_acceleration = down_force;

		// Accelerate (constant gravity)
		
		const float3 gravity_a = make_float3( 0.0f, gravity, 0.0f );

		// sum acceleration
		const float3 a = spring_acceleration + gravity_a;

		// Integrate acceleration (forward Euler) to find velocity
		// const float3 cur_v = (cur_pos-old_pos)/dt;
		// const float3 new_v = cur_v + dt*a; // v'=a

		// Integrate velocity (forward Euler) to find new particle position
		// float3 new_pos = cur_pos + dt*new_v; // pos'=v

		// VERLET INTEGRATION
		// float3 new_pos = 2 * cur_pos - old_pos + a * dt*dt; // no dampening
		float3 new_pos = (2 - lambda) * cur_pos - (1 - lambda) * old_pos + a * dt*dt; // dampening

		// Implement a "floor"
		if(new_pos.y < 0) new_pos.y = 0.0f;

		// Output
		_new_pos[idx] = make_float4( new_pos.x, new_pos.y, new_pos.z, 1.0f );
	}
}

#endif // #ifndef _SIMPLEGL_KERNEL_H_
